import warnings

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

MODEL_REPO = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
NATIVE_DIM = 1024
MAX_TOKENS = 8192
PAD_ID = 151643

# The model is trained asymmetrically: queries carry an instruction, documents do not.
# Qwen's own docs put the difference at 1-5% retrieval quality.
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages\nQuery: "

# Build and execution provider are one decision, not two. Measured on a 16-core laptop
# with an RTX 4060, ~65-token chunks:
#
#   build  provider  ms/chunk (batch 1)  chunks/sec (best batch)  batched == single
#   int8   CPU                       74                     13.5             0.9462
#   int8   DML                      373                      2.7             --
#   fp16   DML                      240                     52.3 (batch 16)  1.0000
#   fp16   CPU                      190                      5.3             --
#
# int8 on DML is 5x slower than on CPU, so pairing them would be a regression. fp16 only
# pays off batched on a GPU. Padding perturbs int8 (0.9462) but is exact on fp16, so
# batching is enabled for fp16 alone.
PROFILES = {
    "int8-cpu": {
        "files": ["onnx/model_int8.onnx"],
        "model": "onnx/model_int8.onnx",
        "providers": ["CPUExecutionProvider"],
        "batch_size": 1,
    },
    "fp16-gpu": {
        "files": ["onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"],
        "model": "onnx/model_fp16.onnx",
        "providers": [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
        "batch_size": 16,
    },
}
DEFAULT_PROFILE = "int8-cpu"


def gpu_available() -> bool:
    """Whether the installed onnxruntime offers an accelerator the fp16 profile can use."""
    offered = set(ort.get_available_providers())
    return bool(offered & {"CUDAExecutionProvider", "DmlExecutionProvider"})


class QwenEmbedding:
    """Qwen3-Embedding-0.6B (Apache-2.0) through ONNX Runtime.

    Defaults to `int8-cpu`, which runs the same everywhere including macOS. The
    `fp16-gpu` profile is ~3.9x faster at ingest but is opt-in, never auto-selected:
    the two builds produce different vectors (0.913 cosine between them), so a store
    whose documents were embedded under one profile must be queried under the same
    one. Picking a profile from whatever hardware happens to be present would
    silently degrade retrieval the moment someone installed a GPU package.

    The model loads on first use, so opening a store triggers no download.
    """

    def __init__(self, profile: str = DEFAULT_PROFILE, providers: list[str] | None = None):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}, expected one of {tuple(PROFILES)}")
        self.profile = profile
        self._config = PROFILES[profile]
        self._requested_providers = providers
        self._session = None
        self._tokenizer = None
        self._kv_inputs: list[str] = []
        self._kv_shape = (1, 0, 0, 0)
        self._kv_dtype = np.float32

    @property
    def batch_size(self) -> int:
        return self._config["batch_size"]

    @property
    def provider(self) -> str | None:
        """The provider actually in use, or None before the model has loaded."""
        return self._session.get_providers()[0] if self._session else None

    @staticmethod
    def _open_session(model_path: str, providers: list[str]) -> ort.InferenceSession:
        """Open a session, degrading to CPU if the accelerator cannot be used.

        onnxruntime does fall back on its own -- verified against providers missing
        from the build, unknown provider names, and an empty list. But a provider
        present in the build that fails at runtime init (onnxruntime-gpu with no
        driver) cannot be tested without that hardware, and this fallback lives in
        onnxruntime's Python layer, which has changed between versions. So the
        guarantee is made here rather than assumed.
        """
        available = [p for p in providers if p in ort.get_available_providers()]
        try:
            return ort.InferenceSession(model_path, providers=available or ["CPUExecutionProvider"])
        except Exception as err:
            if available == ["CPUExecutionProvider"]:
                raise
            warnings.warn(
                f"onnxruntime could not start with {available}: {err}. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def _load(self) -> None:
        if self._session is not None:
            return
        path = snapshot_download(
            MODEL_REPO, allow_patterns=["*.json", "*.txt", *self._config["files"]]
        )
        self._tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
        self._tokenizer.enable_truncation(MAX_TOKENS)
        if self.batch_size > 1:
            # Left padding puts the EOS token last on every row, so pooling stays uniform.
            self._tokenizer.enable_padding(pad_id=PAD_ID, pad_token="<|endoftext|>",
                                           direction="left")

        self._session = self._open_session(
            f"{path}/{self._config['model']}",
            self._requested_providers or self._config["providers"],
        )
        self._kv_inputs = [
            i.name for i in self._session.get_inputs() if i.name.startswith("past_key_values")
        ]
        spec = next(i for i in self._session.get_inputs() if i.name in self._kv_inputs)
        _, heads, _, head_dim = spec.shape
        self._kv_heads, self._kv_head_dim = heads, head_dim
        # fp16 builds reject float32 cache tensors, so follow whatever the export declares.
        self._kv_dtype = np.float16 if spec.type == "tensor(float16)" else np.float32

    def embed_query(self, text: str) -> np.ndarray:
        return self._run([QUERY_INSTRUCTION + text])[0]

    def embed_document(self, text: str) -> np.ndarray:
        return self._run([text])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed many documents, batching where the profile supports it.

        int8 runs one at a time: batching gains nothing on CPU and padding shifts its
        output. fp16 on a GPU batches exactly, and that is where the speedup lives.
        """
        self._load()
        if not texts:
            return np.empty((0, NATIVE_DIM), dtype=np.float32)
        if self.batch_size == 1:
            return np.stack([self._run([text])[0] for text in texts])
        size = self.batch_size
        chunks = [self._run(texts[i : i + size]) for i in range(0, len(texts), size)]
        return np.concatenate(chunks)

    def _run(self, texts: list[str]) -> np.ndarray:
        self._load()
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        # Positions must count real tokens only, or left padding shifts every position.
        feed = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": np.clip(mask.cumsum(-1) - 1, 0, None).astype(np.int64),
        }
        # This export carries a generation KV cache; for embedding it stays empty.
        empty = np.zeros((ids.shape[0], self._kv_heads, 0, self._kv_head_dim), self._kv_dtype)
        for name in self._kv_inputs:
            feed[name] = empty

        hidden = self._session.run(["last_hidden_state"], feed)[0]
        # Last-token pooling: the model's embedding sits on the trailing EOS token.
        return hidden[:, -1].astype(np.float32)
