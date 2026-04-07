"""
app.py — FastAPI server implementing the OpenEnv HTTP API.

Endpoints:
  POST /reset  — start a new episode
  POST /step   — submit an action, receive obs + reward
  GET  /state  — inspect current state without acting
  GET  /health — liveness probe
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ResetRequest, StepRequest, StepResult, StateResponse
from env import SupplyChainEnv

app = FastAPI(
    title="Supply Chain OpenEnv",
    description="Real-world supply chain management environment for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SupplyChainEnv()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Start a new episode for the requested task."""
    try:
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        return {"observation": obs, "task_id": req.task_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest) -> StepResult:
    """Submit an action and advance the environment one step."""
    try:
        result = env.step(task_id=req.task_id, action=req.action)
        return result
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> StateResponse:
    """Return current environment state without advancing it."""
    return env.state()
