class TextSplitter:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        """
        :chunk_size: The maximum number of characters in each chunk.
        :chunk_overlap: The number of characters overlapping between chunks.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0 <= chunk_overlap < chunk_size:
            # A non-advancing step makes split_text loop forever.
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} "
                f"for chunk_size {chunk_size}"
            )

        self.text = None
        self.docs = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __repr__(self) -> str:
        return f"TextSplitter(docs={self.docs})"

    def split_text(self, text: str) -> None:
        self.text = text
        self.docs = []

        start = 0
        text_length = len(self.text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = self.text[start:end].strip()
            if chunk:
                self.docs.append(chunk)
            start += self.chunk_size - self.chunk_overlap

    def get_docs(self) -> list[str]:
        """Return the chunks produced by the last split_text call."""
        return self.docs
