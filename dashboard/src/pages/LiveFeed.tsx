/**
 * LiveFeed page — WebSocket-based real-time face recognition.
 * Streams webcam frames to the server and renders bounding boxes.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Maximize2, Minimize2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useAppStore } from "@/lib/store";
import type { FaceAnalysis } from "@/lib/api";

interface StreamFace extends FaceAnalysis {
  // Extended with rendering metadata
}

export default function LiveFeedPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animFrameRef = useRef<number>(0);

  const [faces, setFaces] = useState<StreamFace[]>([]);
  const [fullscreen, setFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processingMs, setProcessingMs] = useState(0);

  const { isStreaming, setStreaming, setFps } = useAppStore();

  const startStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Connect WebSocket
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(
        `${wsProtocol}//${window.location.host}/api/v1/streams/webcam?token=${import.meta.env.VITE_API_KEY || ""}`
      );
      wsRef.current = ws;

      ws.onopen = () => {
        setStreaming(true);
        setError(null);
        sendFrames();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "ANALYSIS") {
            setFaces(data.faces || []);
            setProcessingMs(data.processing_time_ms || 0);
            setFps(Math.round(1000 / Math.max(data.processing_time_ms || 33, 1)));
            drawOverlay(data.faces || []);
          }
        } catch {
          // Ignore parse errors
        }
      };

      ws.onclose = () => {
        setStreaming(false);
        setFps(0);
      };

      ws.onerror = () => {
        setError("WebSocket connection failed. Is the API server running?");
        setStreaming(false);
      };
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Camera access denied"
      );
    }
  }, [setStreaming, setFps]);

  const stopStream = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "STOP" }));
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }

    cancelAnimationFrame(animFrameRef.current);
    setStreaming(false);
    setFps(0);
    setFaces([]);
  }, [setStreaming, setFps]);

  const sendFrames = useCallback(() => {
    const FRAME_INTERVAL = 66; // ~15 fps
    let lastSendTime = 0;

    const send = () => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (!videoRef.current || !canvasRef.current) return;

      const now = performance.now();
      if (now - lastSendTime < FRAME_INTERVAL) {
        animFrameRef.current = requestAnimationFrame(send);
        return;
      }
      lastSendTime = now;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
      const base64 = dataUrl.split(",")[1];

      wsRef.current.send(
        JSON.stringify({ type: "FRAME", image: base64 })
      );

      animFrameRef.current = requestAnimationFrame(send);
    };

    animFrameRef.current = requestAnimationFrame(send);
  }, []);

  const drawOverlay = useCallback((detectedFaces: StreamFace[]) => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    for (const face of detectedFaces) {
      const { x1, y1, x2, y2 } = face.bbox;
      const w = x2 - x1;
      const h = y2 - y1;

      // Determine color based on trust/liveness
      let color = "#5c7cfa"; // default blue
      if (face.liveness?.verdict?.toUpperCase() === "SPOOF") {
        color = "#ff6b6b"; // red for spoof
      } else if (face.recognition?.is_known) {
        color = "#51cf66"; // green for known
      } else if (face.trust_score > 0.7) {
        color = "#fcc419"; // yellow for high-trust unknown
      }

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);

      // Corner accents
      const corner = Math.min(w, h) * 0.15;
      ctx.lineWidth = 3;
      // Top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + corner);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + corner, y1);
      ctx.stroke();
      // Top-right
      ctx.beginPath();
      ctx.moveTo(x2 - corner, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + corner);
      ctx.stroke();
      // Bottom-left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - corner);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + corner, y2);
      ctx.stroke();
      // Bottom-right
      ctx.beginPath();
      ctx.moveTo(x2 - corner, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - corner);
      ctx.stroke();

      // Label background
      const label = face.recognition?.is_known
        ? face.recognition.identity_name
        : "Unknown";
      const confidence = face.recognition?.confidence
        ? `${(face.recognition.confidence * 100).toFixed(0)}%`
        : "";
      const text = confidence ? `${label} · ${confidence}` : label;

      ctx.font = "bold 14px Inter, system-ui, sans-serif";
      const metrics = ctx.measureText(text);
      const padding = 6;
      const labelH = 22;

      ctx.fillStyle = color + "CC";
      ctx.fillRect(
        x1,
        y1 - labelH - 2,
        metrics.width + padding * 2,
        labelH
      );

      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 + padding, y1 - 8);

      // Trust score bar
      const barW = w;
      const barH = 3;
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(x1, y2 + 4, barW, barH);
      ctx.fillStyle = color;
      ctx.fillRect(x1, y2 + 4, barW * face.trust_score, barH);
    }
  }, []);

  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Live Feed</h2>
          <p className="text-sm text-muted-foreground">
            Real-time face detection, recognition & liveness analysis
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={isStreaming ? "destructive" : "default"}
            onClick={isStreaming ? stopStream : startStream}
          >
            {isStreaming ? (
              <>
                <CameraOff className="mr-2 h-4 w-4" /> Stop
              </>
            ) : (
              <>
                <Camera className="mr-2 h-4 w-4" /> Start Camera
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setFullscreen(!fullscreen)}
          >
            {fullscreen ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>

      {error && (
        <Card className="border-destructive">
          <CardContent className="p-4 text-sm text-destructive">
            {error}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Video feed */}
        <div className={fullscreen ? "lg:col-span-3" : "lg:col-span-2"}>
          <Card className="overflow-hidden">
            <div className="relative aspect-video bg-black">
              <video
                ref={videoRef}
                className="absolute inset-0 h-full w-full object-contain"
                muted
                playsInline
              />
              <canvas ref={canvasRef} className="hidden" />
              <canvas
                ref={overlayRef}
                className="absolute inset-0 h-full w-full object-contain"
              />
              {!isStreaming && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <Camera className="mx-auto mb-3 h-12 w-12 opacity-30" />
                    <p>Click "Start Camera" to begin</p>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Detection panel */}
        {!fullscreen && (
          <div className="space-y-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Detected Faces</CardTitle>
              </CardHeader>
              <CardContent>
                {faces.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {isStreaming ? "No faces detected" : "Stream not active"}
                  </p>
                ) : (
                  <div className="space-y-3">
                    {faces.map((face, i) => (
                      <div
                        key={i}
                        className="rounded-lg border bg-muted/50 p-3 space-y-2"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">
                            {face.recognition?.is_known
                              ? face.recognition.identity_name
                              : "Unknown"}
                          </span>
                          <Badge
                            variant={
                              face.liveness?.verdict?.toUpperCase() === "LIVE"
                                ? "success"
                                : face.liveness?.verdict?.toUpperCase() === "SPOOF"
                                ? "destructive"
                                : "secondary"
                            }
                          >
                            {face.liveness?.verdict || "N/A"}
                          </Badge>
                        </div>
                        {face.recognition?.confidence && (
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>Confidence:</span>
                            <div className="flex-1 rounded-full bg-muted h-1.5">
                              <div
                                className="h-full rounded-full bg-cortexia-500"
                                style={{
                                  width: `${face.recognition.confidence * 100}%`,
                                }}
                              />
                            </div>
                            <span>
                              {(face.recognition.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                        <div className="flex gap-2 text-xs text-muted-foreground">
                          {face.attributes?.age && (
                            <span>Age: ~{face.attributes.age}</span>
                          )}
                          {face.attributes?.gender && (
                            <span>· {face.attributes.gender}</span>
                          )}
                          {face.attributes?.emotion && (
                            <span>· {face.attributes.emotion}</span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-muted-foreground">Trust:</span>
                          <div className="flex-1 rounded-full bg-muted h-1.5">
                            <div
                              className="h-full rounded-full bg-emerald-500"
                              style={{
                                width: `${face.trust_score * 100}%`,
                              }}
                            />
                          </div>
                          <span className="font-mono text-muted-foreground">
                            {(face.trust_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Performance</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <div className="flex justify-between">
                  <span>Processing Time</span>
                  <span className="font-mono">{processingMs.toFixed(0)} ms</span>
                </div>
                <div className="flex justify-between">
                  <span>Faces Tracked</span>
                  <span className="font-mono">{faces.length}</span>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
