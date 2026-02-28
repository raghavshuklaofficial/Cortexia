"""Validation helpers for image uploads."""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_SIZE = 100  # bytes

_IMAGE_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"BM", "BMP"),
    (b"RIFF", "WebP"),
]


def _check_image_signature(data: bytes) -> bool:
    return any(data.startswith(sig) for sig, _ in _IMAGE_SIGNATURES)


async def validate_image_upload(
    file: UploadFile, max_size: int = MAX_IMAGE_SIZE
) -> bytes:
    """Read the upload, enforce size limits and check magic bytes."""
    content = await file.read()

    if len(content) < MIN_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="File too small to be a valid image.")

    if len(content) > max_size:
        max_mb = max_size // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_mb}MB.",
        )

    if not _check_image_signature(content):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Accepted formats: JPEG, PNG, BMP, WebP.",
        )

    return content
