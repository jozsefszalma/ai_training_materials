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
    
    def print_architecture(self):
        """
        Print the model architecture showing all layers.
        """
        print(f"Model: {self.model_name}")
        print("\n" + "="*80)
        print(self.model)
        print("="*80)
    
    def print_layers(self):
        """
        Print detailed information about each layer in the model.
        """
        print(f"Model: {self.model_name}\n")
        
        # Access the underlying transformer model
        if hasattr(self.model, '_modules'):
            for name, module in self.model._modules.items():
                print(f"\n{name}:")
                print(f"  Type: {type(module).__name__}")
                if hasattr(module, '_modules'):
                    for subname, submodule in module._modules.items():
                        print(f"    {subname}: {type(submodule).__name__}")
    
    def get_model_summary(self):
        """
        Print model summary with parameter counts.
        """
        print(f"Model: {self.model_name}\n")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nArchitecture:")
        print("-" * 80)
        
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                num_params = sum(p.numel() for p in module.parameters(recurse=False))
                if num_params > 0:
                    print(f"{name}: {type(module).__name__} ({num_params:,} params)")