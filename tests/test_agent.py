"""
tests/test_agent.py
Integration tests for agent/loop.py using a mock LLM client.

Uses unittest.mock to replace OpenAI API calls — no API key required.
Tests the full agent loop: tool dispatch, trace structure, verdict extraction.
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.loop import run_agent


# ── MOCK LLM RESPONSES ────────────────────────────────────────────────────────

def make_mock_client(responses: list[str]) -> MagicMock:
    """
    Create a mock OpenAI client that returns responses in sequence.
    Each call to client.chat.completions.create() returns the next response.
    """
    client = MagicMock()
    mock_responses = []
    for text in responses:
        choice = MagicMock()
        choice.message.content = text
        completion = MagicMock()
        completion.choices = [choice]
        mock_responses.append(completion)
    client.chat.completions.create.side_effect = mock_responses
    return client


# ── TESTS ─────────────────────────────────────────────────────────────────────

class TestAgentLoop:

    def test_approved_claim_full_trace(self):
        """Full 3-step approved claim: lookup → check_rules → calculate_payout → verdict."""
        responses = [
            'Thought: Look up policy.\nAction: lookup_policy({"user_id": "P-1001"})',
            'Thought: Check rules.\nAction: check_rules({"claim_type": "hail", "plan_type": "Premium", "policy_covers": ["storm","hail","theft","fire","flood","collision"], "policy_status": "active", "claims_this_year": 1})',
            'Thought: Calculate payout.\nAction: calculate_payout({"claimed_amount": 1200, "deductible": 500, "max_single_claim": 15000, "max_annual_payout": 50000, "already_claimed_this_year": 0})',
            "Verdict: APPROVED\nPayout: $700\nReasoning: Hail damage is covered. After deductible of $500, payout is $700.",
        ]
        client = make_mock_client(responses)
        result = run_agent("Hail cracked my windshield.", "P-1001", 1200, client)
        trace = result["trace"]

        # Trace should contain assistant messages
        assistant_msgs = [m for m in trace if m["role"] == "assistant"]
        assert len(assistant_msgs) == 4

        # Final message should contain verdict
        final = assistant_msgs[-1]["content"]
        assert "Verdict: APPROVED" in final
        assert "Payout: $700" in final

    def test_denied_claim_stops_after_check_rules(self):
        """DENIED claim: lookup → check_rules → verdict (no calculate_payout)."""
        responses = [
            'Thought: Look up policy.\nAction: lookup_policy({"user_id": "P-1019"})',
            'Thought: Check rules.\nAction: check_rules({"claim_type": "flood", "plan_type": "Basic", "policy_covers": ["fire","theft"], "policy_status": "active", "claims_this_year": 0})',
            "Verdict: DENIED\nPayout: $0\nReasoning: Flood is not covered under Basic plan.",
        ]
        client = make_mock_client(responses)
        result = run_agent("Basement flooded.", "P-1019", 6000, client)
        trace = result["trace"]

        assistant_msgs = [m for m in trace if m["role"] == "assistant"]
        final = assistant_msgs[-1]["content"]
        assert "Verdict: DENIED" in final

        # calculate_payout should NOT have been called
        tool_calls = [
            m["content"] for m in trace
            if m["role"] == "assistant" and "calculate_payout" in m["content"]
        ]
        assert len(tool_calls) == 0

    def test_hallucination_stops_loop(self):
        """If model outputs Observation itself, loop should stop."""
        responses = [
            'Thought: Look up policy.\nAction: lookup_policy({"user_id": "P-1001"})\nObservation: {"plan_type": "Premium"}',
        ]
        client = make_mock_client(responses)
        result = run_agent("Hail claim.", "P-1001", 1200, client)
        trace = result["trace"]

        system_msgs = [m for m in trace if m["role"] == "system"]
        assert any("REJECTED" in m["content"] for m in system_msgs)

    def test_unknown_tool_returns_error_observation(self):
        """Unknown tool name should produce an error observation."""
        responses = [
            'Thought: I will use a custom tool.\nAction: unknown_tool({"param": "value"})',
            "Verdict: DENIED\nPayout: $0\nReasoning: Could not process.",
        ]
        client = make_mock_client(responses)
        result = run_agent("Some claim.", "P-1001", 1000, client)
        trace = result["trace"]

        user_msgs = [m for m in trace if m["role"] == "user"]
        error_msgs = [m for m in user_msgs if "unknown tool" in m["content"].lower()]
        assert len(error_msgs) > 0

    def test_trace_starts_with_user_message(self):
        """Trace should start with the user claim message."""
        responses = [
            "Verdict: APPROVED\nPayout: $0\nReasoning: Quick decision.",
        ]
        client = make_mock_client(responses)
        result = run_agent("Quick claim.", "P-1001", 100, client)
        trace = result["trace"]
        assert trace[0]["role"] == "user"

    def test_max_steps_respected(self):
        """Agent should not exceed max_steps iterations."""
        # Model never outputs Verdict — should stop at max_steps
        responses = [
            'Thought: Still thinking.\nAction: lookup_policy({"user_id": "P-1001"})',
        ] * 10  # more than max_steps
        client = make_mock_client(responses)
        result = run_agent("Claim.", "P-1001", 1000, client, max_steps=3)
        trace = result["trace"]

        assistant_msgs = [m for m in trace if m["role"] == "assistant"]
        assert len(assistant_msgs) <= 3

    def test_run_agent_returns_risk_level(self):
        """run_agent() should return risk_level and risk_confidence from classifier."""
        responses = [
            "Verdict: APPROVED\nPayout: $700\nReasoning: Covered.",
        ]
        client = make_mock_client(responses)
        result = run_agent("My house was destroyed by fire.", "P-1001", 5000, client)

        assert "risk_level" in result
        assert result["risk_level"] in ["high", "medium", "low"]
        assert "risk_confidence" in result
        assert 0.0 <= result["risk_confidence"] <= 1.0