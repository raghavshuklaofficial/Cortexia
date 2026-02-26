"""
CORTEXIA CLI — Command-line interface for headless operations.

Usage:
    cortexia serve        Start the API server
    cortexia enroll       Enroll a new identity
    cortexia recognize    Recognize faces in an image
    cortexia info         Show system information
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="cortexia")
def cli():
    """CORTEXIA — Neural Face Intelligence Platform."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, help="Bind port")
@click.option("--workers", default=1, help="Number of workers")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the CORTEXIA API server."""
    import uvicorn

    from cortexia.utils.logging import setup_logging

    setup_logging(log_level="INFO")

    console.print(f"[bold green]CORTEXIA[/] v1.0.0 starting on {host}:{port}")
    console.print(f"  Docs:  http://{host}:{port}/docs")
    console.print(f"  Redoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "cortexia.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


@cli.command()
@click.option("--name", required=True, help="Identity name")
@click.option("--images", required=True, type=click.Path(exists=True), help="Image directory or file")
def enroll(name: str, images: str):
    """Enroll a new identity from image files."""
    import asyncio
    from pathlib import Path

    import cv2
    import numpy as np

    async def _enroll():
        from cortexia.core.trust_pipeline import TrustPipeline

        pipeline = TrustPipeline()
        pipeline.initialize()

        image_path = Path(images)
        image_files = []
        if image_path.is_dir():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(image_path.glob(ext))
        else:
            image_files = [image_path]

        console.print(f"[bold]Enrolling[/] '{name}' with {len(image_files)} image(s)...")

        embeddings = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                console.print(f"  [red]Skip[/] {img_path.name} (unreadable)")
                continue

            result = pipeline.process_image(img)
            if result.face_count == 0:
                console.print(f"  [red]Skip[/] {img_path.name} (no face detected)")
                continue

            fa = result.faces[0]
            if fa.embedding is not None:
                embeddings.append(fa.embedding)
                console.print(f"  [green]OK[/]   {img_path.name} (confidence: {fa.face.confidence:.3f})")

        console.print(f"\n[bold green]Enrolled[/] {name} with {len(embeddings)} face embedding(s)")

    asyncio.run(_enroll())


@cli.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Image to analyze")
def recognize(image: str):
    """Recognize faces in an image."""
    import cv2

    from cortexia.core.trust_pipeline import TrustPipeline

    pipeline = TrustPipeline()
    pipeline.initialize()

    img = cv2.imread(image)
    if img is None:
        console.print("[red]Error:[/] Cannot read image.")
        sys.exit(1)

    result = pipeline.process_image(img)

    table = Table(title=f"CORTEXIA Analysis — {result.face_count} face(s) detected")
    table.add_column("#", style="dim")
    table.add_column("Identity", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Trust Score", justify="right")
    table.add_column("Liveness", justify="center")
    table.add_column("Age", justify="right")
    table.add_column("Gender")
    table.add_column("Emotion")

    for i, fa in enumerate(result.faces, 1):
        identity = fa.recognition.identity_name if fa.recognition else "—"
        conf = f"{fa.recognition.confidence:.2%}" if fa.recognition else "—"
        trust = f"{fa.trust_score:.2%}"
        liveness = fa.liveness.verdict.value if fa.liveness else "—"
        age = str(fa.attributes.age) if fa.attributes and fa.attributes.age else "—"
        gender = fa.attributes.gender if fa.attributes and fa.attributes.gender else "—"
        emotion = fa.attributes.emotion.value if fa.attributes and fa.attributes.emotion else "—"

        table.add_row(str(i), identity, conf, trust, liveness, age, gender, emotion)

    console.print(table)
    console.print(f"\nProcessing time: {result.total_processing_time_ms:.0f}ms")


@cli.command()
def info():
    """Show CORTEXIA system information."""
    table = Table(title="CORTEXIA System Information")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    from cortexia import __version__
    from cortexia.config import get_settings

    settings = get_settings()

    table.add_row("Version", __version__)
    table.add_row("Environment", settings.app_env)
    table.add_row("Detection Backend", settings.model_backend)
    table.add_row("Embedding Dimension", str(settings.embedding_dim))
    table.add_row("Trust Pipeline", "Enabled" if settings.trust_pipeline_enabled else "Disabled")
    table.add_row("Anti-Spoofing", "Enabled" if settings.antispoof_enabled else "Disabled")
    table.add_row("Attributes", "Enabled" if settings.attributes_enabled else "Disabled")
    table.add_row("Recognition Threshold", str(settings.recognition_threshold))
    table.add_row("Anti-Spoof Threshold", str(settings.antispoof_threshold))

    # Check GPU
    try:
        import onnxruntime

        gpu = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        table.add_row("GPU Available", "Yes" if gpu else "No (CPU only)")
    except ImportError:
        table.add_row("GPU Available", "Unknown (onnxruntime not installed)")

    console.print(table)


if __name__ == "__main__":
    cli()
