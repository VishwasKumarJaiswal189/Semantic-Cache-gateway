import os
import json
import uuid
import chromadb
import redis
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

r = redis.from_url(REDIS_URL, decode_responses=True)

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)

# Cosine distance = 1 - cosine_similarity (0.12 distance = 0.88 similarity)
MAX_DISTANCE_THRESHOLD = 0.12

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def query_cache(prompt: str):
    if collection.count() == 0:
        return None, 1.0

    prompt_vector = get_embedding(prompt)
    results = collection.query(
        query_embeddings=[prompt_vector],
        n_results=1
    )

    if not results["ids"][0]:
        return None, 1.0

    best_distance = results["distances"][0][0]
    matched_id = results["ids"][0][0]

    if best_distance <= MAX_DISTANCE_THRESHOLD:
        cached_meta = r.get(f"cache:{matched_id}")
        if cached_meta:
            return json.loads(cached_meta), best_distance

    return None, best_distance

def store_in_cache(prompt: str, response_text: str, prompt_vector: list[float]):
    cache_id = str(uuid.uuid4())

    collection.add(
        ids=[cache_id],
        embeddings=[prompt_vector],
        documents=[prompt]
    )

    r.set(
        f"cache:{cache_id}",
        json.dumps({"prompt": prompt, "response": response_text})
    )