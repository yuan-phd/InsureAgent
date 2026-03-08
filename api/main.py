"""
api/main.py
InsureAgent FastAPI inference server.

Endpoints:
  POST /process_claim  — run full agent loop, return verdict + trace
  GET  /health         — health check
"""

from fastapi import FastAPI, HTTPException
from api.schemas import ClaimRequest, ClaimResponse
from api.inference import process_claim

app = FastAPI(title="InsureAgent API")


@app.post("/process_claim", response_model=ClaimResponse)
async def handle_claim(request: ClaimRequest):
    try:
        result = process_claim(
            claim_text=request.claim_text,
            user_id=request.user_id,
            claimed_amount=request.claimed_amount,
            model=request.model,
        )
        return ClaimResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}