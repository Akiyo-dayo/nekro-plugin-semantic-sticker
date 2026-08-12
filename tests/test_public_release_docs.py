from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_documents_supported_environment_and_limits() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_fragments = [
        "NekroAgent v2.3.3",
        "onebot_v11",
        "CHAT_PROXY",
        "VECTOR_DIMENSION",
        "Akiyo.semantic_sticker",
        "client_max_body_size",
        "?token=",
        "local_path",
        "data URL",
        "JSON object",
        "dimensions",
        "完整重启",
        "内部接口",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in readme]
    assert not missing, f"README is missing public compatibility or security notes: {missing}"


def test_mit_license_is_present() -> None:
    license_path = ROOT / "LICENSE"
    assert license_path.is_file(), "repository root is missing LICENSE"
    license_text = license_path.read_text(encoding="utf-8")
    required_fragments = [
        "MIT License",
        "Copyright (c) 2026 Akiyo",
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in license_text]
    assert not missing, f"LICENSE is not the expected MIT text: {missing}"
