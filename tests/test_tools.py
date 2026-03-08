
"""
tests/test_tools.py
Unit tests for tools/calculator.py and tools/rules.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.calculator import calculate_payout
from tools.rules import check_rules


# ── CALCULATOR ────────────────────────────────────────────────────────────────

class TestCalculatePayout:

    def test_normal_claim(self):
        result = calculate_payout(1200, 500, 15000, 50000)
        assert result["payout"] == 700.0

    def test_below_deductible(self):
        result = calculate_payout(300, 500, 15000, 50000)
        assert result["payout"] == 0
        assert "reason" in result

    def test_exactly_at_deductible(self):
        result = calculate_payout(500, 500, 15000, 50000)
        assert result["payout"] == 0

    def test_capped_by_single_claim_limit(self):
        result = calculate_payout(35000, 500, 20000, 50000)
        assert result["payout"] == 20000.0

    def test_capped_by_annual_limit(self):
        result = calculate_payout(5000, 500, 15000, 50000, already_claimed_this_year=48000)
        assert result["payout"] == 2000.0

    def test_annual_limit_exhausted(self):
        result = calculate_payout(5000, 500, 15000, 50000, already_claimed_this_year=50000)
        assert result["payout"] == 0

    def test_breakdown_present(self):
        result = calculate_payout(1200, 500, 15000, 50000)
        assert "breakdown" in result
        assert result["breakdown"]["claimed_amount"] == 1200
        assert result["breakdown"]["deductible_applied"] == 500
        assert result["breakdown"]["after_deductible"] == 700

    def test_already_claimed_defaults_to_zero(self):
        r1 = calculate_payout(1200, 500, 15000, 50000)
        r2 = calculate_payout(1200, 500, 15000, 50000, already_claimed_this_year=0)
        assert r1["payout"] == r2["payout"]

    def test_large_claim_flood(self):
        # P-1009: claimed 60000, deductible 500, max_single 35000, max_annual 50000
        result = calculate_payout(60000, 500, 35000, 50000)
        assert result["payout"] == 35000.0

    def test_rounding(self):
        result = calculate_payout(1200.5, 500.3, 15000, 50000)
        assert result["payout"] == round(result["payout"], 2)


# ── RULES ─────────────────────────────────────────────────────────────────────

class TestCheckRules:

    def test_valid_hail_claim(self):
        result = check_rules("hail", "Premium",
                             ["storm", "hail", "theft", "fire", "flood", "collision"],
                             "active", 1)
        assert result["eligible"] is True
        assert result["max_single_claim"] == 15000

    def test_lapsed_policy(self):
        result = check_rules("storm", "Premium",
                             ["storm", "hail"], "lapsed", 0)
        assert result["eligible"] is False
        assert "lapsed" in result["reason"]

    def test_cancelled_policy(self):
        result = check_rules("fire", "Standard",
                             ["storm", "fire"], "cancelled", 0)
        assert result["eligible"] is False

    def test_claim_type_not_covered(self):
        result = check_rules("flood", "Basic",
                             ["fire", "theft"], "active", 0)
        assert result["eligible"] is False
        assert "flood" in result["reason"]

    def test_annual_limit_reached_basic(self):
        result = check_rules("fire", "Basic",
                             ["fire", "theft"], "active", 2)
        assert result["eligible"] is False
        assert "limit" in result["reason"].lower()

    def test_annual_limit_reached_standard(self):
        result = check_rules("storm", "Standard",
                             ["storm", "fire"], "active", 4)
        assert result["eligible"] is False

    def test_annual_limit_not_reached_premium(self):
        result = check_rules("flood", "Premium",
                             ["storm", "hail", "theft", "fire", "flood", "collision"],
                             "active", 9)
        assert result["eligible"] is True

    def test_premium_limit_reached(self):
        result = check_rules("flood", "Premium",
                             ["storm", "hail", "theft", "fire", "flood", "collision"],
                             "active", 10)
        assert result["eligible"] is False

    def test_theft_requires_inspection(self):
        result = check_rules("theft", "Standard",
                             ["storm", "fire", "theft", "collision"],
                             "active", 0)
        assert result["eligible"] is True
        assert result["requires_inspection"] is True

    def test_storm_no_inspection(self):
        result = check_rules("storm", "Premium",
                             ["storm", "hail", "theft", "fire", "flood", "collision"],
                             "active", 0)
        assert result["eligible"] is True
        assert result["requires_inspection"] is False

    def test_unknown_claim_type(self):
        result = check_rules("earthquake", "Premium",
                             ["storm", "hail"], "active", 0)
        assert result["eligible"] is False

    def test_all_claim_types_eligible(self):
        covers = ["storm", "hail", "theft", "fire", "flood", "collision"]
        for claim_type in covers:
            result = check_rules(claim_type, "Premium", covers, "active", 0)
            assert result["eligible"] is True, f"{claim_type} should be eligible"