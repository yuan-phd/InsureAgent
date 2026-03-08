"""
tests/locustfile.py
Load testing for InsureAgent FastAPI inference server.

Usage:
    # Start FastAPI server first:
    docker compose up

    # Run load test (web UI):
    locust -f tests/locustfile.py --host http://localhost:8000

    # Run headless (CI mode):
    locust -f tests/locustfile.py \
        --host http://localhost:8000 \
        --headless \
        --users 10 \
        --spawn-rate 2 \
        --run-time 60s \
        --only-summary

Target metrics to record in evaluation/locust_results.md:
    - QPS (requests per second at stable load)
    - P50 / P95 / P99 latency
    - Max concurrent users before degradation (>2x baseline latency)
    - Failure rate (should be 0% under normal load)
"""

import random
from locust import HttpUser, task, between


# ── SAMPLE CLAIMS ─────────────────────────────────────────────────────────────
# Representative mix of claim types, amounts, and expected outcomes

SAMPLE_CLAIMS = [
    # APPROVED cases
    {
        "user_id": "P-1001",
        "claim_text": "My car windshield was cracked by hail during a storm.",
        "claimed_amount": 1200,
        "model": "teacher",
    },
    {
        "user_id": "P-1021",
        "claim_text": "Strong winds from a storm knocked over my garden fence and damaged my shed roof.",
        "claimed_amount": 2800,
        "model": "teacher",
    },
    {
        "user_id": "P-1009",
        "claim_text": "Heavy flooding in my area caused water to enter my ground floor rooms.",
        "claimed_amount": 12000,
        "model": "teacher",
    },
    {
        "user_id": "P-1030",
        "claim_text": "An electrical fire in my garage destroyed my car and damaged the walls.",
        "claimed_amount": 22000,
        "model": "teacher",
    },
    {
        "user_id": "P-1012",
        "claim_text": "A collision on the motorway caused significant damage to the front of my car.",
        "claimed_amount": 8000,
        "model": "teacher",
    },
    # DENIED cases
    {
        "user_id": "P-1019",
        "claim_text": "My basement flooded after a heavy rainfall ruined my furniture.",
        "claimed_amount": 6000,
        "model": "teacher",
    },
    {
        "user_id": "P-1013",
        "claim_text": "A fire in my kitchen caused damage to the cabinets and appliances.",
        "claimed_amount": 7000,
        "model": "teacher",
    },
    {
        "user_id": "P-1022",
        "claim_text": "My car was stolen from the street outside my house last night.",
        "claimed_amount": 15000,
        "model": "teacher",
    },
]


# ── LOAD TEST USER ────────────────────────────────────────────────────────────

class InsureAgentUser(HttpUser):
    """
    Simulates a user submitting insurance claims via the API.

    wait_time: random wait between requests (1-3 seconds)
    Simulates realistic user behaviour rather than hammering the API.
    """
    wait_time = between(1, 3)

    @task(9)
    def process_claim(self):
        """Main task: submit a random claim. Weight 9 (90% of requests)."""
        claim = random.choice(SAMPLE_CLAIMS)
        with self.client.post(
            "/process_claim",
            json=claim,
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "verdict" not in data:
                    response.failure("Response missing 'verdict' field")
                elif data["verdict"] not in ("APPROVED", "DENIED", "NO_VERDICT"):
                    response.failure(f"Unexpected verdict: {data['verdict']}")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:100]}")

    @task(1)
    def health_check(self):
        """Health check endpoint. Weight 1 (10% of requests)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")