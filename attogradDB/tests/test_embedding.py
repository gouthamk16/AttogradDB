"""Profile selection, provider fallback, and batching. These load the real model."""

import warnings

import numpy as np
import pytest

from attogradDB.embedding import NATIVE_DIM, PROFILES, QwenEmbedding, gpu_available


def test_default_profile_is_int8_on_cpu():
    """int8 on DirectML measured 5x slower than on CPU, so they must not be paired."""
    embedder = QwenEmbedding()
    assert embedder.profile == "int8-cpu"
    assert PROFILES["int8-cpu"]["providers"] == ["CPUExecutionProvider"]


def test_gpu_is_never_selected_automatically():
    """Auto-switching would embed queries with a different build than stored documents."""
    assert QwenEmbedding().profile == "int8-cpu"
    assert "Dml" not in " ".join(PROFILES["int8-cpu"]["providers"])


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="Unknown profile"):
        QwenEmbedding(profile="fp32-tpu")


def test_int8_profile_does_not_batch():
    """Padding shifts int8 output (0.9462 against unpadded), so it stays unbatched."""
    assert QwenEmbedding(profile="int8-cpu").batch_size == 1


def test_coreml_is_not_in_any_profile():
    """It ships in the default macOS wheel; enabling it unasked could slow Macs down."""
    for config in PROFILES.values():
        assert "CoreMLExecutionProvider" not in config["providers"]


def test_azure_provider_is_never_used():
    """It is remote inference and present in some default builds; never send text to it."""
    for config in PROFILES.values():
        assert "AzureExecutionProvider" not in config["providers"]


def test_runs_on_cpu_when_the_requested_provider_is_unavailable():
    embedder = QwenEmbedding(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert embedder.embed_query("does this still work").shape == (NATIVE_DIM,)
    assert embedder.provider == "CPUExecutionProvider"


def test_runs_on_cpu_when_the_provider_does_not_exist():
    embedder = QwenEmbedding(providers=["NoSuchExecutionProvider"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert embedder.embed_document("still fine").shape == (NATIVE_DIM,)
    assert embedder.provider == "CPUExecutionProvider"


def test_provider_is_none_before_load():
    assert QwenEmbedding().provider is None


def test_query_and_document_differ():
    embedder = QwenEmbedding()
    q = embedder.embed_query("redis")
    d = embedder.embed_document("redis")
    assert not np.allclose(q, d), "the instruction prefix must change the vector"


def test_embed_documents_matches_one_at_a_time():
    embedder = QwenEmbedding()
    texts = ["first document", "a rather longer second document to force padding"]
    batched = embedder.embed_documents(texts)
    solo = np.stack([embedder.embed_document(t) for t in texts])

    assert batched.shape == (2, NATIVE_DIM)
    for b, s in zip(batched, solo):
        b, s = b / np.linalg.norm(b), s / np.linalg.norm(s)
        assert float(b @ s) > 0.999, "batching must not change the vectors"


def test_embed_documents_on_empty_input():
    assert QwenEmbedding().embed_documents([]).shape == (0, NATIVE_DIM)


@pytest.mark.skipif(not gpu_available(), reason="no CUDA or DirectML provider installed")
def test_gpu_profile_batches_and_agrees_with_itself():
    embedder = QwenEmbedding(profile="fp16-gpu")
    assert embedder.batch_size > 1
    texts = ["first document", "a rather longer second document to force padding"]
    batched = embedder.embed_documents(texts)
    solo = np.stack([embedder.embed_document(t) for t in texts])
    for b, s in zip(batched, solo):
        b, s = b / np.linalg.norm(b), s / np.linalg.norm(s)
        assert float(b @ s) > 0.999
