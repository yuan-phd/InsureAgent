"""
api/main_kube.py

Lightweight FastAPI mock for Kubernetes demo.
No model loading. Returns realistic mock responses.

Usage:
    uvicorn api.main_kube:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(
    title="InsureAgent API (Demo)",
    description="Agentic LLM Insurance Claims Processing",
    version="1.0.0",
)


class ClaimRequest(BaseModel):
    user_id: str
    claim_text: str
    claimed_amount: float


class ClaimResponse(BaseModel):
    user_id: str
    verdict: str
    payout: float
    reasoning: str
    risk_level: str
    risk_confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process_claim", response_model=ClaimResponse)
def process_claim(request: ClaimRequest):
    """Process an insurance claim and return a verdict."""

    # Realistic mock logic based on claim text keywords
    claim_lower = request.claim_text.lower()

    denied_keywords = ["flood", "lapsed", "cancelled", "not covered"]
    is_denied = any(k in claim_lower for k in denied_keywords)

    if is_denied:
        return ClaimResponse(
            user_id=request.user_id,
            verdict="DENIED",
            payout=0.0,
            reasoning="Claim type is not covered under this policy.",
            risk_level="high",
            risk_confidence=round(random.uniform(0.80, 0.95), 3),
        )

    payout = max(0.0, request.claimed_amount - 500)  # mock deductible $500
    return ClaimResponse(
        user_id=request.user_id,
        verdict="APPROVED",
        payout=round(payout, 2),
        reasoning=f"Claim is covered. After $500 deductible, payout is ${payout:.2f}.",
        risk_level="medium",
        risk_confidence=round(random.uniform(0.65, 0.85), 3),
    )