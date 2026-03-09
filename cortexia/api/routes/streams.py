"""WebSocket streaming for real-time face analysis."""

from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cortexia.api.deps import get_pipeline
from cortexia.api.routes.recognize import _face_analysis_to_schema
from cortexia.config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/streams", tags=["Streaming"])

MAX_CONNECTIONS = 5  # don't let too many WS clients kill the server


class ConnectionManager:
    """Manage WebSocket connections for streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> bool:
        if len(self.active_connections) >= MAX_CONNECTIONS:
            await websocket.close(code=4429, reason="Too many connections")
            return False
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("websocket_connected", total=len(self.active_connections))
        return True

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("websocket_disconnected", total=len(self.active_connections))

    async def broadcast(self, data: dict) -> None:
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/webcam")
async def webcam_stream(websocket: WebSocket):
    """Real-time webcam face analysis via WebSocket.

    Protocol:
    - Connect with ?token=<API_KEY> for authentication
    - Client sends base64-encoded JPEG frames as text messages
    - Server responds with JSON containing face analysis results
    - Send "PING" for keepalive, server responds with "PONG"
    - Send "STOP" to disconnect gracefully

    Example client message:
        {"frame": "<base64-encoded-jpeg>"}

    Example server response:
        {
            "faces": [...],
            "face_count": 2,
            "fps": 12.5,
            "timestamp": "2025-01-01T12:00:00"
        }
    """
    # Authenticate before accepting
    settings = get_settings()
    token = websocket.query_params.get("token")
    if settings.app_env != "development" and (
        not token or token != settings.api_key
    ):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    connected = await manager.connect(websocket)
    if not connected:
        return

    pipeline = get_pipeline()

    frame_count = 0
    fps_start = time.perf_counter()

    try:
        while True:
            # Receive frame from client
            raw = await websocket.receive_text()

            if raw.strip().upper() == "PING":
                await websocket.send_text("PONG")
                continue

            # Handle stop command (raw string or JSON)
            if raw.strip().upper() == "STOP":
                break

            try:
                msg = json.loads(raw)
                # Support both "frame" and "image" keys from client
                frame_b64 = msg.get("image") or msg.get("frame", "")
                # Handle JSON stop command: {"type": "STOP"}
                if msg.get("type", "").upper() == "STOP":
                    break
            except json.JSONDecodeError:
                frame_b64 = raw  # Assume raw base64

            if not frame_b64:
                continue

            # Decode base64 frame
            try:
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid frame data"})
                continue

            if frame is None:
                continue

            # Process frame through Trust Pipeline
            result = pipeline.process_image(frame)

            # Calculate FPS
            frame_count += 1
            elapsed = time.perf_counter() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            if frame_count % 100 == 0:
                fps_start = time.perf_counter()
                frame_count = 0

            # Build response
            face_schemas = [_face_analysis_to_schema(fa) for fa in result.faces]

            response = {
                "type": "ANALYSIS",
                "faces": [f.model_dump() for f in face_schemas],
                "face_count": result.face_count,
                "known_count": result.known_count,
                "spoof_count": result.spoof_count,
                "fps": round(fps, 1),
                "processing_time_ms": round(result.total_processing_time_ms, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("webcam_stream_client_disconnected")
    except Exception as e:
        logger.error("webcam_stream_error", error=str(e))
    finally:
        manager.disconnect(websocket)
