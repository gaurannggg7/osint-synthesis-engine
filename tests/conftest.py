import dataclasses

import pytest

from config import Settings


@pytest.fixture
def settings(tmp_path):
    """Settings isolated from the developer's real .env / data / index."""
    base = Settings.from_env()
    return dataclasses.replace(
        base,
        data_dir=tmp_path / "data",
        corpus_path=tmp_path / "data" / "corpus_final.jsonl",
        chroma_dir=tmp_path / "chroma",
        trace_dir=None,
        groq_api_key="test-key-not-real",
        cors_allowed_origins=["https://app.example.com"],
        rate_limit_per_minute=0,
        trust_proxy_headers=False,
        demo_mode="off",
        contact_user_agent=None,
    )


CHUNKS = [
    {"doc_id": "doc-a", "source": "SEC_EDGAR", "title": "Filing A", "url": "https://x/a", "text": "Alpha text. " * 5},
    {"doc_id": "doc-b", "source": "OFAC_SDN", "title": "Entry B", "url": "https://x/b", "text": "Bravo text."},
    {"doc_id": "doc-a", "source": "SEC_EDGAR", "title": "Filing A", "url": "https://x/a", "text": "Alpha second chunk."},
]


class FakeResponse:
    def __init__(self, content, usage=None, extras=None):
        self.content = content
        self.response_metadata = {"token_usage": usage or {"prompt_tokens": 10, "completion_tokens": 20,
                                                           "total_tokens": 30}}
        self.additional_kwargs = extras or {}


class FakeLLM:
    def __init__(self, content="Claim one [1]. Claim two [3].", error=None):
        self.content, self.error, self.prompts = content, error, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return FakeResponse(self.content)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = CHUNKS if chunks is None else chunks
        self.calls = []

    def __call__(self, query, k):
        self.calls.append((query, k))
        return self.chunks[:k]
