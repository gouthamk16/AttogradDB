"""Provider selection and fallback. These load the real model, so keep them few."""

import warnings

import onnxruntime as ort
import pytest

from attogradDB.embedding import QwenEmbedding


def test_cpu_is_always_last_and_present():
    providers = QwenEmbedding.available_providers()
    assert providers[-1] == "CPUExecutionProvider"
    assert len(providers) == len(set(providers))


def test_coreml_is_not_auto_selected():
    """It ships in the default macOS wheel; enabling it unasked could slow Macs down."""
    assert "CoreMLExecutionProvider" not in QwenEmbedding.available_providers()


def test_only_providers_the_build_offers_are_requested():
    offered = set(ort.get_available_providers())
    assert set(QwenEmbedding.available_providers()) <= offered | {"CPUExecutionProvider"}


def test_azure_provider_is_never_selected():
    """It is remote inference and present in the default build; never send text to it."""
    assert "AzureExecutionProvider" not in QwenEmbedding.available_providers()


def test_unavailable_provider_falls_back_to_cpu():
    """The guarantee that makes GPU support safe on machines without one."""
    embedder = QwenEmbedding(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = embedder.embed_query("does this still work")

    assert vector.shape == (1024,)
    assert embedder.provider == "CPUExecutionProvider"


def test_nonsense_provider_falls_back_to_cpu():
    embedder = QwenEmbedding(providers=["NoSuchExecutionProvider"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert embedder.embed_document("still fine").shape == (1024,)
    assert embedder.provider == "CPUExecutionProvider"


def test_provider_is_none_before_load():
    assert QwenEmbedding().provider is None


def test_query_and_document_paths_differ(monkeypatch):
    embedder = QwenEmbedding()
    seen = []
    monkeypatch.setattr(embedder, "_embed", lambda text: seen.append(text))
    embedder.embed_query("abc")
    embedder.embed_document("abc")
    assert seen[0] != seen[1] and seen[1] == "abc"


@pytest.mark.parametrize("dim", [1024])
def test_embeddings_are_full_native_dimension(dim):
    assert QwenEmbedding().embed_document("hello").shape == (dim,)
