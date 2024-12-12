## Implementation of a class that orders the given text into a collection of docs

class TextSplitter:
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        """
        :chunk_size: The maximum number of characters in each chunk.
        :chunk_overlap: The number of characters overlapping between chunks.
        """
        self.text = None  
        self.docs = []    
        self.chunk_size = chunk_size  
        self.chunk_overlap = chunk_overlap  
    
    def __repr__(self) -> str:
        return f"TextSplitter(docs={self.docs})"
    
    def split_text(self, text: str) -> None:
        # Reset the state
        self.text = text
        self.docs = []
        
        # Split text into chunks of a specified size with overlap
        start = 0
        text_length = len(self.text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = self.text[start:end].strip()  # remove leading/trailing whitespace
            if chunk: 
                self.docs.append(chunk)
            start += self.chunk_size - self.chunk_overlap  # Move start point
    
    def get_docs(self, extract: bool = False) -> list:
        """
        Returns the split text chunks. 
        If extract=True, returns a list of docs.
        If extract=False (default), returns the class instance.
        """
        return self if not extract else self.docs