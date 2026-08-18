import warnings

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

MODEL_REPO = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
MODEL_FILE = "onnx/model_int8.onnx"
NATIVE_DIM = 1024
MAX_TOKENS = 8192

# The model is trained asymmetrically: queries carry an instruction, documents do not.
# Qwen's own docs put the difference at 1-5% retrieval quality.
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages\nQuery: "


class QwenEmbedding:
    """Qwen3-Embedding-0.6B (Apache-2.0) through ONNX Runtime.

    One text at a time, deliberately. Batching gave no throughput gain on CPU
    (~50ms per chunk at every batch size) and padding perturbs the result -- a padded
    text scored only 0.95 against its unpadded self, so identical documents would
    embed differently depending on what they were batched with.

    The model and tokenizer load on first use, not on construction, so opening a
    store does not trigger a 600MB download.
    """

    def __init__(self, providers: list[str] | None = None):
        self._requested_providers = providers
        self._session = None
        self._tokenizer = None
        self._kv_inputs: list[str] = []
        self._kv_shape = (1, 0, 0, 0)
        self._kv_dtype = np.float32

    @staticmethod
    def available_providers() -> list[str]:
        """Opted-into accelerators first, CPU last as the always-present fallback.

        Only CUDA and DirectML are auto-preferred, because neither ships in the
        default wheel -- you get them by deliberately installing onnxruntime-gpu or
        onnxruntime-directml, so their presence means acceleration was asked for.

        CoreML is deliberately NOT auto-preferred even though it does ship in the
        macOS wheel. Enabling it silently would change behaviour for every Mac user
        without being asked, and CoreML can run slower than CPU on a model it has to
        partition heavily -- which this export, with its 56 KV-cache inputs, is a
        strong candidate for. Mac users who measure a win can opt in explicitly:

            QwenEmbedding(providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        """
        available = ort.get_available_providers()
        preferred = ("CUDAExecutionProvider", "DmlExecutionProvider")
        return [p for p in preferred if p in available] + ["CPUExecutionProvider"]

    @staticmethod
    def _open_session(model_path: str, providers: list[str]) -> ort.InferenceSession:
        """Open a session, degrading to CPU if the accelerator cannot be used.

        onnxruntime does fall back on its own -- verified against providers missing
        from the build, unknown provider names, and an empty list. But a provider
        that is present in the build and fails at runtime init (onnxruntime-gpu on a
        machine with no driver) is a path we cannot test here, and the fallback lives
        in onnxruntime's Python layer where it has changed before. So the guarantee
        is made here instead of assumed.
        """
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as err:
            if providers == ["CPUExecutionProvider"]:
                raise
            warnings.warn(
                f"onnxruntime could not start with {providers}: {err}. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @property
    def provider(self) -> str | None:
        """The provider actually in use, or None before the model has loaded."""
        return self._session.get_providers()[0] if self._session else None

    def _load(self) -> None:
        if self._session is not None:
            return
        path = snapshot_download(MODEL_REPO, allow_patterns=["*.json", "*.txt", MODEL_FILE])
        self._tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._session = self._open_session(
            f"{path}/{MODEL_FILE}", self._requested_providers or self.available_providers()
        )
        self._kv_inputs = [
            i.name for i in self._session.get_inputs() if i.name.startswith("past_key_values")
        ]
        # Shape is (batch, heads, past_len, head_dim); read it off rather than hardcoding.
        spec = next(i for i in self._session.get_inputs() if i.name in self._kv_inputs)
        _, heads, _, head_dim = spec.shape
        self._kv_shape = (1, heads, 0, head_dim)
        # fp16 builds reject float32 cache tensors, so follow whatever the export declares.
        self._kv_dtype = np.float16 if spec.type == "tensor(float16)" else np.float32

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(QUERY_INSTRUCTION + text)

    def _embed(self, text: str) -> np.ndarray:
        self._load()
        ids = np.array([self._tokenizer.encode(text).ids], dtype=np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "position_ids": np.arange(ids.shape[1], dtype=np.int64)[None, :],
        }
        # This export carries a generation KV cache; for embedding it stays empty.
        for name in self._kv_inputs:
            feed[name] = np.zeros(self._kv_shape, dtype=self._kv_dtype)

        hidden = self._session.run(["last_hidden_state"], feed)[0]
        # Last-token pooling: the model's embedding sits on the trailing EOS token.
        return hidden[0, -1].astype(np.float32)
