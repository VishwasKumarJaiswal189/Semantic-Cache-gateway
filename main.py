import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from cache_engine import query_cache, store_in_cache, get_embedding, client, LLM_MODEL, r

app = FastAPI(title="Semantic Cache Gateway")
app.mount("/static", StaticFiles(directory="static"), name="static")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(req: PromptRequest):
    start_time = time.perf_counter()
    
    try:
        cached_data, distance = query_cache(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache/Embedding error: {str(e)}")

    similarity = round(max(0.0, 1 - distance), 4)

    if cached_data:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        
        # Calculate latency saved compared to average LLM miss latency
        miss_count = int(r.get("metrics:misses") or 0)
        total_miss_ms = float(r.get("metrics:total_miss_ms") or 1800.0)
        avg_miss_ms = (total_miss_ms / miss_count) if miss_count > 0 else 1800.0
        saved_ms = max(0.0, round(avg_miss_ms - latency_ms, 1))

        r.incr("metrics:hits")
        r.incrbyfloat("metrics:total_hit_ms", latency_ms)
        r.incrbyfloat("metrics:saved_ms", saved_ms)
        r.incrbyfloat("metrics:saved_dollars", 0.005)

        return {
            "source": "cache",
            "similarity_score": similarity,
            "latency_ms": latency_ms,
            "latency_saved_ms": saved_ms,
            "matched_prompt": cached_data["prompt"],
            "response": cached_data["response"]
        }

    # Cache Miss -> Call OpenRouter Chat Model
    try:
        chat_completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": req.prompt}],
            temperature=0.3
        )
        llm_output = chat_completion.choices[0].message.content
        vector = get_embedding(req.prompt)
        store_in_cache(req.prompt, llm_output, vector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
    r.incr("metrics:misses")
    r.incrbyfloat("metrics:total_miss_ms", latency_ms)

    return {
        "source": "model",
        "similarity_score": similarity,
        "latency_ms": latency_ms,
        "latency_saved_ms": 0.0,
        "response": llm_output
    }

@app.get("/analytics")
async def get_analytics():
    hits = int(r.get("metrics:hits") or 0)
    misses = int(r.get("metrics:misses") or 0)
    total = hits + misses

    total_hit_ms = float(r.get("metrics:total_hit_ms") or 0.0)
    total_miss_ms = float(r.get("metrics:total_miss_ms") or 0.0)
    saved_ms = float(r.get("metrics:saved_ms") or 0.0)
    saved_dollars = float(r.get("metrics:saved_dollars") or 0.0)

    avg_hit_ms = round(total_hit_ms / hits, 1) if hits > 0 else 0.0
    avg_miss_ms = round(total_miss_ms / misses, 1) if misses > 0 else 0.0

    return {
        "total_calls": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round((hits / total * 100), 1) if total > 0 else 0.0,
        "avg_cache_latency_ms": avg_hit_ms,
        "avg_llm_latency_ms": avg_miss_ms,
        "total_time_saved_sec": round(saved_ms / 1000, 2),
        "estimated_cost_saved_usd": round(saved_dollars, 4)
    }