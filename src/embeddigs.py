from sentence_transformers import SentenceTransformer
import numpy as np

DEFAULT_MODEL_NAME = "embaas/sentence-transformers-multilingual-e5-large"

class EmbeddingModel:
    """A class for generating text embeddings and computing similarity."""
    
    
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       Defaults to multilingual-e5-large if not specified.
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embeddings, where each embedding is a list of floats.
        """
        return self.model.encode(texts).tolist()
    
    def compute_cosine_similarity(self, embeddings1: list[list[float]], embeddings2: list[list[float]]) -> list[float]:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embeddings1: First embedding or list of embeddings.
            embeddings2: Second embedding or list of embeddings.
            
        Returns:
            Cosine similarity score(s).
        """
        return np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))