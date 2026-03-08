from pydantic import BaseModel

class ClaimRequest(BaseModel):
    user_id: str
    claim_text: str
    claimed_amount: float
    model: str = "teacher"  # "teacher" or "student"

class ClaimResponse(BaseModel):
    verdict: str
    payout: float
    reasoning: str
    trace: list
    latency_ms: float
    model_used: str