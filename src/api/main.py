"""CyberSentry detection API.

Run:  uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from .service import get_detector
from src.agents.investigation import InvestigationAgent

app = FastAPI(title="CyberSentry Detection API", version="0.1.0")

_investigation_agent = InvestigationAgent()


@app.get("/health")
def health() -> dict:
    d = get_detector()
    return {"status": "ok", "n_features": len(d.features), "classes": d.classes}


@app.post("/detect")
def detect(flow: dict = Body(..., description="One network flow: feature name -> value")) -> dict:
    """Score one flow. Feature names are the CIC-IDS2017 column names; missing ones use training medians."""
    if not flow:
        raise HTTPException(status_code=422, detail="Empty flow")
    return get_detector().detect(flow)


@app.post("/detect/batch")
def detect_batch(flows: list[dict] = Body(...)) -> list[dict]:
    if len(flows) > 1000:
        raise HTTPException(status_code=413, detail="At most 1000 flows per request")
    d = get_detector()
    return [d.detect(f) for f in flows]


class InvestigateRequest(BaseModel):
    flow: dict = {}
    event_id: str | None = None

    model_config = {"extra": "forbid"}


@app.post("/investigate")
def investigate(body: InvestigateRequest) -> dict:
    """Investigate one network flow.

    Runs the detector internally, then applies the Investigation Agent to
    produce a structured evidence-based report.  Supply feature values under
    ``flow`` using the same CIC-IDS2017 column names accepted by ``/detect``.
    Missing features are imputed with training medians.

    An optional ``event_id`` string is echoed back in the report to aid
    correlation; it is never invented or modified by the agent.
    """
    if not body.flow:
        raise HTTPException(status_code=422, detail="'flow' must be a non-empty feature dict")
    try:
        detection = get_detector().detect(body.flow)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detector error: {exc}") from exc
    report = _investigation_agent.investigate(
        detection=detection,
        flow=body.flow,
        event_id=body.event_id,
    )
    return report
