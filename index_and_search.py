import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(path):
    with open(path) as f:
        return json.load(f)

def build_index(data, text_field="Description"):
    texts = [item[text_field] for item in data]
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, texts

def search(query, index, data, top_k=3):
    q_embedding = model.encode([query])
    D, I = index.search(np.array(q_embedding), top_k)
    return [data[i] for i in I[0]]

if __name__ == "__main__":
    # Choose which file to load
    db_path = "player_abilities.json"
    db = load_data(db_path)

    # Build index on 'Description' field
    index, _, _ = build_index(db, text_field="Description")

    # Sample search
    while True:
        query = input("\n🔍 Enter your search prompt: ")
        if query.lower() in ["exit", "quit"]:
            break
        results = search(query, index, db)
        print("\n📌 Results:")
        for r in results:
            print(f"- {r['Name']}: {r['Description']}")
