#%%
from src.embeddigs import EmbeddingModel

#%%
# Initialize the embedding model (uses default model)
embedding_model = EmbeddingModel()

#%%
# Define diverse text examples to demonstrate different similarity patterns
texts = [
    # Similar texts (should have high positive similarity)
    "The weather in Paris is pleasant in August.",
    "Paris has nice weather during August.",
    
    # Negation example
    "The weather in Paris is not pleasant in August.",
    
    # Opposite sentiment (same topic, different sentiment - moderate similarity)
    "The weather in Paris is terrible in August.",
    
    # Orthogonal/unrelated texts (should have similarity close to 0)
    "Quantum computing uses qubits for parallel computation.",
    "The recipe requires three eggs and two cups of flour.",
    
    # Contradictory concepts (might produce lower or negative similarity)
    "The company's profits increased dramatically.",
    "The company faced severe financial losses.",
    
    # Multilingual examples - same concept in different languages
    "I love reading books in the evening.",  # English
    "Ich liebe es, abends Bücher zu lesen.",  # German
    "Szeretek este könyvet olvasni.",  # Hungarian
]

# Generate embeddings for all texts
embeddings = embedding_model.embed_texts(texts)

#%%
# Display each text with its embedding dimensions
print("=" * 80)
print("TEXT EMBEDDINGS")
print("=" * 80)
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    print(f"\nText {i}: \"{text}\"")
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Last 5 values: {embedding[-5:]}")

#%%
# Calculate and display cosine similarities between interesting pairs
print("\n" + "=" * 80)
print("COSINE SIMILARITY ANALYSIS")
print("=" * 80)

# Define interesting pairs to compare
comparison_pairs = [
    (0, 1, "Similar texts (same topic, similar sentiment)"),
    (0, 2, "Negation (pleasant vs. not pleasant)"),
    (0, 3, "Same topic, opposite sentiment (pleasant vs. terrible)"),
    (0, 4, "Completely unrelated topics (orthogonal)"),
    (4, 5, "Two unrelated topics"),
    (6, 7, "Opposite financial outcomes"),
    (1, 4, "Different topics (weather vs. quantum computing)"),
    (5, 7, "Unrelated topics (recipe vs. financial losses)"),
    (8, 9, "Same concept: English vs. German"),
    (8, 10, "Same concept: English vs. Hungarian"),
    (9, 10, "Same concept: German vs. Hungarian"),
]

for idx1, idx2, description in comparison_pairs:
    similarity = embedding_model.compute_cosine_similarity(embeddings[idx1], embeddings[idx2])
    print(f"\nPair ({idx1}, {idx2}): {description}")
    print(f"  Text {idx1}: \"{texts[idx1][:50]}...\"" if len(texts[idx1]) > 50 else f"  Text {idx1}: \"{texts[idx1]}\"")
    print(f"  Text {idx2}: \"{texts[idx2][:50]}...\"" if len(texts[idx2]) > 50 else f"  Text {idx2}: \"{texts[idx2]}\"")
    print(f"  Cosine Similarity: {similarity:.4f}")

#%%