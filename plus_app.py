import os
import sys
import time
import io
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Local imports
from plus_extractor import PlusMultimodalExtractor
from plus_graph import PlusIntentGraphEngine
from plus_router import VLMGatewayRouter

from contextlib import asynccontextmanager

# Global instances & metrics state
extractor: Optional[PlusMultimodalExtractor] = None
graph_engine: Optional[PlusIntentGraphEngine] = None
router_gateway: Optional[VLMGatewayRouter] = None

metrics_history: List[Dict[str, Any]] = []
total_inspections: int = 0
total_blocked: int = 0
total_passed: int = 0
latency_records: List[float] = []

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

def init_models():
    global extractor, graph_engine, router_gateway
    if extractor is None:
        print("[PolyGuard-VLM Plus] Initializing Core Security Models...")
        extractor = PlusMultimodalExtractor(lazy_load_clip=True)
        graph_engine = PlusIntentGraphEngine(embedding_dim=512, distance_threshold=0.45)
        count = graph_engine.seed_benchmark_dataset(extractor=extractor, num_samples=25)
        print(f"[PolyGuard-VLM Plus] Intent Graph pre-seeded with {count} adversarial vectors.")
        router_gateway = VLMGatewayRouter(provider="mock")
        print("[PolyGuard-VLM Plus] Security Gateway Ready!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield

app = FastAPI(
    title="PolyGuard-VLM Plus Guardrail API",
    description="Multilingual VLM Security Guardrail with OpenCLIP, Graph Engine, and Router Proxy",
    version="2.0.0",
    lifespan=lifespan
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SafetyInspectionResponse(BaseModel):
    is_safe: bool = Field(..., description="True if prompt passed safety inspection, False if adversarial")
    jailbreak_risk_score: float = Field(..., description="Out-of-Distribution Anomaly Score S_OOD (0.0 to 1.0)")
    latency_ms: float = Field(..., description="Guardrail inspection latency in milliseconds")
    language_detected: str = Field(default="Auto-CrossLingual-LaBSE", description="Language detection indicator")
    action_taken: str = Field(..., description="PASSED_TO_VLM or BLOCKED_ADVERSARIAL_INTENT")
    vlm_response: Optional[Any] = Field(default=None, description="Upstream VLM response payload if safe")


@app.get("/")
def read_root():
    """Serves real-time telemetry web dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PolyGuard-VLM Plus API is running. Access /v1/guardrail/inspect for endpoints."}


@app.post("/v1/guardrail/inspect", response_model=SafetyInspectionResponse)
async def inspect_guardrail(
    prompt: str = Form(..., description="Text prompt in any language"),
    image_file: Optional[UploadFile] = File(None, description="Optional image input")
):
    global total_inspections, total_blocked, total_passed, latency_records
    start_time = time.perf_counter()

    if extractor is None or graph_engine is None:
        raise HTTPException(status_code=500, detail="Security models not initialized.")

    # Process image input if provided
    pil_image = None
    if image_file:
        try:
            image_bytes = await image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image upload: {e}")

    # Extract multimodal embedding
    embedding, meta = extractor.encode_multimodal(prompt, pil_image)
    visual_anomaly = meta.get("visual_anomaly_factor", 0.0)

    # Calculate risk score
    risk_score, details = graph_engine.compute_plus_risk_score(embedding, visual_anomaly_factor=visual_anomaly)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Decision threshold (0.60)
    IS_SAFE_THRESHOLD = 0.60
    is_safe = risk_score <= IS_SAFE_THRESHOLD

    total_inspections += 1
    latency_records.append(elapsed_ms)

    vlm_out = None
    if is_safe:
        total_passed += 1
        action_taken = "PASSED_TO_VLM"
        if router_gateway:
            vlm_res = await router_gateway.route_safe_request(prompt, pil_image)
            vlm_out = vlm_res.get("vlm_response", "Safe")
    else:
        total_blocked += 1
        action_taken = "BLOCKED_ADVERSARIAL_INTENT"

    log_entry = {
        "timestamp": time.time(),
        "prompt": prompt[:120],
        "is_safe": is_safe,
        "jailbreak_risk_score": float(risk_score),
        "latency_ms": float(elapsed_ms),
        "action_taken": action_taken
    }
    metrics_history.insert(0, log_entry)
    if len(metrics_history) > 50:
        metrics_history.pop()

    resp_data = SafetyInspectionResponse(
        is_safe=is_safe,
        jailbreak_risk_score=float(risk_score),
        latency_ms=float(elapsed_ms),
        language_detected="Auto-CrossLingual-LaBSE",
        action_taken=action_taken,
        vlm_response=vlm_out
    )

    if not is_safe:
        return JSONResponse(status_code=403, content=resp_data.dict())

    return resp_data


@app.get("/v1/guardrail/metrics")
def get_metrics():
    avg_lat = float(np.mean(latency_records)) if latency_records else 0.0
    p95_lat = float(np.percentile(latency_records, 95)) if latency_records else 0.0
    return {
        "total_inspections": total_inspections,
        "total_passed": total_passed,
        "total_blocked": total_blocked,
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat
    }


@app.get("/v1/guardrail/history")
def get_history():
    return metrics_history[:20]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
