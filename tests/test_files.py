from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from nekro_plugin_semantic_sticker.config import SemanticStickerConfig
from nekro_plugin_semantic_sticker.models import UploadPayload


def image_bytes(format_name: str, *, size: tuple[int, int] = (8, 8), frames: int = 1) -> bytes:
    output = BytesIO()
    images = [Image.new("RGBA", size, (index * 20 % 255, 40, 80, 255)) for index in range(frames)]
    if format_name == "JPEG":
        images[0].convert("RGB").save(output, format="JPEG")
    elif format_name in {"GIF", "WEBP"} and frames > 1:
        images[0].save(output, format=format_name, save_all=True, append_images=images[1:], duration=20, loop=0)
    else:
        images[0].save(output, format=format_name)
    return output.getvalue()


@pytest.fixture
def config() -> SemanticStickerConfig:
    return SemanticStickerConfig()


@pytest.fixture
def image_store(tmp_path: Path, config: SemanticStickerConfig):
    from nekro_plugin_semantic_sticker.files import ImageStore

    return ImageStore(tmp_path, config)


@pytest.mark.parametrize(
    ("format_name", "content_type", "extension"),
    [("PNG", "image/png", "png"), ("JPEG", "image/jpeg", "jpg"), ("GIF", "image/gif", "gif"), ("WEBP", "image/webp", "webp")],
)
def test_validate_detects_format_and_normalizes_extension(image_store, format_name: str, content_type: str, extension: str) -> None:
    content = image_bytes(format_name)
    validated = image_store.validate(UploadPayload(content=content, filename="sticker.wrong", content_type=content_type))
    digest = hashlib.sha256(content).hexdigest()
    assert validated.sha256 == digest
    assert validated.detected_format == format_name
    assert validated.detected_extension == extension
    assert validated.mime_type == content_type
    assert validated.asset_name == f"{digest}.{extension}"
    assert validated.original_bytes == content


def test_rejects_mime_spoofing_and_path_traversal(image_store) -> None:
    from nekro_plugin_semantic_sticker.files import UploadValidationError

    with pytest.raises(UploadValidationError, match="MIME"):
        image_store.validate(UploadPayload(content=image_bytes("PNG"), filename="x.png", content_type="image/jpeg"))
    with pytest.raises(UploadValidationError, match="filename"):
        image_store.validate(UploadPayload(content=image_bytes("PNG"), filename="../../x.png", content_type="image/png"))


@pytest.mark.parametrize("content", [b"", b"not-an-image", b"BM" + b"x" * 100])
def test_rejects_empty_malformed_and_unsupported_images(image_store, content: bytes) -> None:
    from nekro_plugin_semantic_sticker.files import UploadValidationError

    with pytest.raises(UploadValidationError):
        image_store.validate(UploadPayload(content=content, filename="x.bin", content_type=None))


def test_enforces_byte_dimension_pixel_and_frame_limits(tmp_path: Path) -> None:
    from nekro_plugin_semantic_sticker.files import ImageStore, UploadValidationError

    byte_store = ImageStore(tmp_path / "bytes", SemanticStickerConfig(MAX_UPLOAD_BYTES=10))
    with pytest.raises(UploadValidationError, match="bytes"):
        byte_store.validate(UploadPayload(content=image_bytes("PNG"), filename="x.png", content_type="image/png"))

    width_store = ImageStore(tmp_path / "width", SemanticStickerConfig(MAX_WIDTH=4))
    with pytest.raises(UploadValidationError, match="width"):
        width_store.validate(UploadPayload(content=image_bytes("PNG", size=(8, 2)), filename="x.png", content_type="image/png"))

    height_store = ImageStore(tmp_path / "height", SemanticStickerConfig(MAX_HEIGHT=4))
    with pytest.raises(UploadValidationError, match="height"):
        height_store.validate(UploadPayload(content=image_bytes("PNG", size=(2, 8)), filename="x.png", content_type="image/png"))

    pixel_store = ImageStore(tmp_path / "pixels", SemanticStickerConfig(MAX_IMAGE_PIXELS=8))
    with pytest.raises(UploadValidationError, match="pixels"):
        pixel_store.validate(UploadPayload(content=image_bytes("PNG", size=(4, 4)), filename="x.png", content_type="image/png"))

    frame_store = ImageStore(tmp_path / "frames", SemanticStickerConfig(MAX_ANIMATION_FRAMES=2))
    with pytest.raises(UploadValidationError, match="frames"):
        frame_store.validate(UploadPayload(content=image_bytes("GIF", frames=3), filename="x.gif", content_type="image/gif"))


def test_rejects_pillow_decompression_bomb(image_store, monkeypatch: pytest.MonkeyPatch) -> None:
    from nekro_plugin_semantic_sticker.files import UploadValidationError

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(UploadValidationError, match="decompression bomb"):
        image_store.validate(UploadPayload(content=image_bytes("PNG", size=(4, 4)), filename="x.png", content_type="image/png"))


def test_install_is_content_addressed_atomic_and_idempotent(image_store) -> None:
    validated = image_store.validate(
        UploadPayload(content=image_bytes("GIF", frames=2), filename="user secret.gif", content_type="image/gif")
    )
    first = image_store.install(validated)
    second = image_store.install(validated)

    assert first == second
    assert Path(first.asset_path).name == validated.asset_name
    assert Path(first.thumbnail_path).name == f"{validated.sha256}.webp"
    assert "user secret" not in first.asset_path
    assert Path(first.asset_path).read_bytes() == validated.original_bytes
    assert Path(first.thumbnail_path).is_file()
    assert image_store.backups_dir.is_dir()
    assert image_store.snapshot().temp_files == []
    assert image_store.snapshot().assets == [first.asset_path]


def test_delete_is_managed_and_idempotent(image_store) -> None:
    validated = image_store.validate(UploadPayload(content=image_bytes("PNG"), filename="x.png", content_type="image/png"))
    stored = image_store.install(validated)
    image_store.delete(stored)
    image_store.delete(stored)
    snapshot = image_store.snapshot()
    assert snapshot.assets == []
    assert snapshot.thumbnails == []
    assert snapshot.temp_files == []
    assert snapshot.total_bytes == 0

def test_existing_content_addressed_asset_must_match_hash(image_store) -> None:
    from nekro_plugin_semantic_sticker.files import UploadValidationError

    validated = image_store.validate(UploadPayload(content=image_bytes("PNG"), filename="x.png", content_type="image/png"))
    stored = image_store.install(validated)
    Path(stored.asset_path).write_bytes(b"corrupted")
    with pytest.raises(UploadValidationError, match="integrity"):
        image_store.install(validated)