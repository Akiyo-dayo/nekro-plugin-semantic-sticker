from __future__ import annotations

import hashlib
import os
import re
import uuid
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import SemanticStickerConfig
from .models import FileSnapshot, StoredAsset, UploadPayload, ValidatedImage


class UploadValidationError(ValueError):
    pass


_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "GIF": ("image/gif", "gif"),
    "WEBP": ("image/webp", "webp"),
}


class ImageStore:
    def __init__(self, data_root: Path, config: SemanticStickerConfig) -> None:
        self.data_root = Path(data_root).resolve()
        self.assets_dir = self.data_root / "assets"
        self.thumbnails_dir = self.data_root / "thumbnails"
        self.temp_dir = self.data_root / "temp"
        self.backups_dir = self.data_root / "backups"
        self.config = config

    def validate(self, upload: UploadPayload) -> ValidatedImage:
        content = bytes(upload.content)
        if not content:
            raise UploadValidationError("image bytes are empty")
        if len(content) > self.config.MAX_UPLOAD_BYTES:
            raise UploadValidationError("image bytes exceed configured limit")
        self._validate_filename(upload.filename)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    detected_format = (image.format or "").upper()
                    image.verify()
                with Image.open(BytesIO(content)) as image:
                    width, height = image.size
                    frame_count = int(getattr(image, "n_frames", 1))
        except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise UploadValidationError("Pillow decompression bomb rejected") from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise UploadValidationError("malformed or unsupported image") from exc

        if detected_format not in _FORMATS:
            raise UploadValidationError(f"unsupported image format: {detected_format or 'unknown'}")
        mime_type, extension = _FORMATS[detected_format]
        if upload.content_type and upload.content_type.casefold() != mime_type:
            raise UploadValidationError("request MIME does not match detected image MIME")
        if width > self.config.MAX_WIDTH:
            raise UploadValidationError("image width exceeds configured limit")
        if height > self.config.MAX_HEIGHT:
            raise UploadValidationError("image height exceeds configured limit")
        if width * height > self.config.MAX_IMAGE_PIXELS:
            raise UploadValidationError("image pixels exceed configured limit")
        if frame_count > self.config.MAX_ANIMATION_FRAMES:
            raise UploadValidationError("image frames exceed configured limit")

        digest = hashlib.sha256(content).hexdigest()
        return ValidatedImage(
            original_bytes=content,
            sha256=digest,
            detected_format=detected_format,
            detected_extension=extension,
            mime_type=mime_type,
            byte_size=len(content),
            width=width,
            height=height,
            frame_count=frame_count,
            animated=frame_count > 1,
            asset_name=self._safe_asset_name(digest, extension),
        )

    def install(self, image: ValidatedImage) -> StoredAsset:
        self._ensure_directories()
        asset_path = self.assets_dir / image.asset_name
        thumbnail_path = self.thumbnails_dir / f"{image.sha256}.webp"
        if asset_path.exists():
            if self._sha256_file(asset_path) != image.sha256:
                raise UploadValidationError("content-addressed asset integrity mismatch")
        else:
            self._atomic_write(asset_path, image.original_bytes)
        if not thumbnail_path.exists():
            self._atomic_write(thumbnail_path, self._thumbnail_bytes(image.original_bytes))
        return StoredAsset(
            sha256=image.sha256,
            asset_path=str(asset_path),
            thumbnail_path=str(thumbnail_path),
            detected_format=image.detected_format,
            detected_extension=image.detected_extension,
            mime_type=image.mime_type,
            byte_size=image.byte_size,
            width=image.width,
            height=image.height,
            frame_count=image.frame_count,
            animated=image.animated,
        )

    def delete(self, asset: StoredAsset) -> None:
        for raw_path, managed_root in (
            (asset.asset_path, self.assets_dir),
            (asset.thumbnail_path, self.thumbnails_dir),
        ):
            path = Path(raw_path).resolve()
            root = managed_root.resolve()
            if not path.is_relative_to(root):
                raise UploadValidationError("refusing to delete an unmanaged file")
            path.unlink(missing_ok=True)

    def snapshot(self) -> FileSnapshot:
        groups = (
            (self.assets_dir, "assets"),
            (self.thumbnails_dir, "thumbnails"),
            (self.temp_dir, "temp_files"),
        )
        values: dict[str, list[str]] = {"assets": [], "thumbnails": [], "temp_files": []}
        total_bytes = 0
        for directory, key in groups:
            if not directory.exists():
                continue
            for path in sorted(item for item in directory.iterdir() if item.is_file()):
                values[key].append(str(path))
                total_bytes += path.stat().st_size
        return FileSnapshot(total_bytes=total_bytes, **values)

    def _ensure_directories(self) -> None:
        for directory in (self.assets_dir, self.thumbnails_dir, self.temp_dir, self.backups_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, destination: Path, content: bytes) -> None:
        temp_path = self.temp_dir / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temp_path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _thumbnail_bytes(content: bytes) -> bytes:
        output = BytesIO()
        with Image.open(BytesIO(content)) as image:
            image.seek(0)
            thumbnail = image.convert("RGBA")
            thumbnail.thumbnail((512, 512), Image.Resampling.LANCZOS)
            thumbnail.save(output, format="WEBP", method=6)
        return output.getvalue()

    @staticmethod
    def _validate_filename(filename: str | None) -> None:
        if not filename:
            return
        if filename in {".", ".."} or "/" in filename or "\\" in filename:
            raise UploadValidationError("upload filename contains a path component")

    @staticmethod
    def _safe_asset_name(digest: str, extension: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise UploadValidationError("invalid content hash")
        if extension not in {"png", "jpg", "gif", "webp"}:
            raise UploadValidationError("invalid detected extension")
        return f"{digest}.{extension}"