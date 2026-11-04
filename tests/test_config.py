"""清单与配置测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docrag.config import Settings
from docrag.ingest.manifest import Manifest


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path, "modelA@1024")
    manifest.put("a.docx", "hash1", 12)
    manifest.save()

    loaded = Manifest(path, "modelA@1024")
    entry = loaded.get("a.docx")
    assert entry is not None
    assert entry.doc_id == "hash1"
    assert entry.chunks == 12
    assert loaded.embedding_changed is False


def test_manifest_detects_embedding_change(tmp_path):
    path = tmp_path / "manifest.json"
    Manifest(path, "modelA@1024").save()
    assert Manifest(path, "modelB@768").embedding_changed is True


def test_manifest_remove(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path, "m")
    manifest.put("a.docx", "h", 1)
    manifest.remove("a.docx")
    assert manifest.get("a.docx") is None


def test_corrupted_manifest_falls_back_to_empty(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    assert Manifest(path, "m").files == {}


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 400
    assert settings.llm_provider == "deepseek"
    assert settings.model == "deepseek-flash"
    assert settings.profile.model == "deepseek-flash"


def test_provider_profiles():
    deepseek = Settings(_env_file=None, llm_provider="deepseek")
    mimo = Settings(_env_file=None, llm_provider="mimo")
    assert deepseek.base_url == "https://api.deepseek.com"
    assert mimo.base_url == "https://token-plan-cn.xiaomimimo.com/v1"
    assert mimo.model == "mimo-v2.6-pro"
    assert mimo.profile.max_output_field == "max_completion_tokens"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=200, chunk_overlap=200)


def test_missing_api_key_raises():
    settings = Settings(_env_file=None, deepseek_api_key=None, llm_provider="deepseek")
    with pytest.raises(RuntimeError):
        _ = settings.api_key
