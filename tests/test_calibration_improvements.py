"""Tests for calibrated_shrink aggregator, devils-advocate protocol, and prompt v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregation.calibrated_shrink import CalibratedShrinkAggregator
from src.aggregation.mean import MeanAggregator
from src.agents.protocol import summarize_round
from src.agents.prompts import SYSTEM_PROMPT_CALIBRATED, SYSTEM_PROMPT_CALIBRATED_V2
from src.data.schema import Forecast


def f(p, rationale="r", evidence=None):
    return Forecast.from_p_yes(p, rationale=rationale, evidence_used=evidence or [])


class TestCalibratedShrink:
    def test_high_side_shrinks_toward_prior(self):
        agg = CalibratedShrinkAggregator(p0=0.30, w_lo=0.8, w_hi=0.5)
        out = agg.aggregate([f(0.9), f(0.9), f(0.9)])
        assert abs(out.p_yes - (0.30 + 0.5 * (0.9 - 0.30))) < 1e-9  # 0.60
        assert out.label == "YES"

    def test_low_side_shrinks_less(self):
        agg = CalibratedShrinkAggregator(p0=0.30, w_lo=0.8, w_hi=0.5)
        out = agg.aggregate([f(0.1), f(0.1), f(0.1)])
        assert abs(out.p_yes - (0.30 + 0.8 * (0.1 - 0.30))) < 1e-9  # 0.14

    def test_at_prior_is_fixed_point(self):
        agg = CalibratedShrinkAggregator(p0=0.30)
        out = agg.aggregate([f(0.30), f(0.30)])
        assert abs(out.p_yes - 0.30) < 1e-9

    def test_identity_weights_match_mean(self):
        agg = CalibratedShrinkAggregator(p0=0.30, w_lo=1.0, w_hi=1.0)
        mean = MeanAggregator()
        fs = [f(0.2), f(0.5), f(0.9)]
        assert abs(agg.aggregate(fs).p_yes - mean.aggregate(fs).p_yes) < 1e-9

    def test_empty_forecasts(self):
        out = CalibratedShrinkAggregator().aggregate([])
        assert out.p_yes == 0.5

    def test_invalid_params_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            CalibratedShrinkAggregator(p0=0.0)
        with pytest.raises(ValueError):
            CalibratedShrinkAggregator(w_lo=1.5)


class TestDevilsAdvocate:
    def test_unanimous_yes_triggers_opposing_case(self):
        fs = [f(0.8), f(0.7), f(0.9)]
        s = summarize_round(fs, share_mode="arguments", devils_advocate=True)
        assert "resolves NO" in s
        assert "lean YES" in s

    def test_unanimous_no_targets_yes(self):
        fs = [f(0.1), f(0.2), f(0.3)]
        s = summarize_round(fs, share_mode="full", devils_advocate=True)
        assert "resolves YES" in s

    def test_split_round_no_injection(self):
        fs = [f(0.8), f(0.2), f(0.9)]
        s = summarize_round(fs, share_mode="arguments", devils_advocate=True)
        assert "Unanimity" not in s

    def test_default_off_reproduces_legacy(self):
        fs = [f(0.8), f(0.7), f(0.9)]
        legacy = summarize_round(fs, share_mode="arguments")
        explicit_off = summarize_round(fs, share_mode="arguments", devils_advocate=False)
        assert legacy == explicit_off
        assert "Unanimity" not in legacy

    def test_arguments_mode_still_hides_numbers(self):
        fs = [f(0.8), f(0.7), f(0.9)]
        s = summarize_round(fs, share_mode="arguments", devils_advocate=True)
        assert "0.8" not in s and "p_yes=" not in s


class TestPromptV2:
    def test_v2_extends_v1_themes(self):
        assert "RESOLUTION CRITERIA" in SYSTEM_PROMPT_CALIBRATED_V2
        assert "HIGH-CONFIDENCE GATE" in SYSTEM_PROMPT_CALIBRATED_V2
        assert "BASE RATE" in SYSTEM_PROMPT_CALIBRATED_V2
        assert "valid JSON" in SYSTEM_PROMPT_CALIBRATED_V2

    def test_v1_unchanged(self):
        assert "RESOLUTION CRITERIA" not in SYSTEM_PROMPT_CALIBRATED
