import pytest

from config import REPO_ROOT, ConfigurationError, Settings


def test_paths_default_relative_to_repo_not_cwd(monkeypatch, tmp_path):
    for name in ("DATA_DIR", "CORPUS_PATH", "CHROMA_PERSIST_DIR", "TRACE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    s = Settings.from_env()
    assert s.chroma_dir == REPO_ROOT / "vectorstore" / "chroma_db"
    assert s.corpus_path == REPO_ROOT / "data" / "processed" / "corpus_final.jsonl"


def test_env_overrides_and_empty_trace_dir_disables_traces(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("TRACE_DIR", "")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://a.example/, https://b.example")
    s = Settings.from_env()
    assert s.chroma_dir == tmp_path / "c" and s.trace_dir is None and s.rate_limit_per_minute == 3
    assert s.cors_allowed_origins == ["https://a.example", "https://b.example"]


@pytest.mark.parametrize("name,value", [("RATE_LIMIT_PER_MINUTE", "lots"), ("DEMO_MODE", "yes")])
def test_invalid_values_fail_fast(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_env()
