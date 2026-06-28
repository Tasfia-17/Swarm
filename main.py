"""
SwarmPilot FastAPI backend
Endpoints:
  POST /debate          — run one debate frame
  GET  /metrics         — last debate metrics
  GET  /scenes          — list available scenes
  GET  /                — serve frontend
"""
import asyncio
import time
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simulator.frame_gen import get_frame, get_prev_frame, list_scenes
from vision.processor import process as vision_process
from debate.orchestrator import run_debate

app = FastAPI(title="SwarmPilot")

_metrics: dict = {}
_history: list[str] = []
_frame_idx: int = 0


class DebateRequest(BaseModel):
    scene: str | None = None


@app.post("/debate")
async def debate(req: DebateRequest):
    global _frame_idx, _history

    frame, scene_meta = get_frame(req.scene, _frame_idx)
    prev = get_prev_frame() if _frame_idx > 0 else None
    _frame_idx += 1

    # Vision: 3 images
    vision = vision_process(frame, prev)
    images = [vision["rgb"], vision["flow"], vision["depth"]]

    scene_desc = (
        f"Scene: {scene_meta['name']}. "
        f"Speed: {scene_meta['speed']} km/h. "
        f"Hazard: {scene_meta.get('hazard') or 'none'}."
    )

    result = await run_debate(
        scene_desc=scene_desc,
        hazard=scene_meta.get("hazard"),
        images=images,
        history=_history[-8:],
    )
    result["frame_b64"] = vision["rgb"]
    result["flow_b64"] = vision["flow"]
    result["depth_b64"] = vision["depth"]

    _history.append(f"Frame {_frame_idx}: {result['final_action']} ({result['final_confidence']:.2f})")
    if len(_history) > 20:
        _history.pop(0)

    global _metrics
    _metrics = {
        "total_ms": result["total_ms"],
        "avg_agent_ms": result["avg_agent_ms"],
        "rounds": len(result["rounds"]),
        "converged": result["converged"],
        "final_action": result["final_action"],
        "scene": result["scene"],
        "frame": _frame_idx,
    }

    return result


@app.get("/metrics")
def metrics():
    return _metrics


@app.get("/scenes")
def scenes():
    return {"scenes": list_scenes()}


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>SwarmPilot — frontend not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
