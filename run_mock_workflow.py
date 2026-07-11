#!/usr/bin/env python3
"""Run workflow with one mock question and 20 mock evidence items.

  python run_mock_workflow.py           # real LLM (need OPENAI_API_KEY)
  python run_mock_workflow.py --dry-run # no API, stub agent
"""
import argparse
import sys
from pathlib import Path

# run from polysim root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.aggregation.mean import MeanAggregator
from src.data.schema import EvidenceItem, Forecast, QuestionExample
from src.runner.experiment import run_s0, run_s1, run_s2
from src.utils.logger import setup_logging


def make_mock_evidence(n: int = 20) -> list[EvidenceItem]:
    """Mock 20 docs: mix of positive/negative/neutral for a Yes/No question."""
    templates = [
        ("news", "Official poll shows 58% support for the measure as of last week."),
        ("news", "Opposition leader announced they will vote no."),
        ("wiki", "Similar proposals passed in 2018 and 2020 in neighboring states."),
        ("official", "Committee vote scheduled for next Tuesday; outcome uncertain."),
        ("news", "Economists surveyed: 12 in favor, 8 against."),
        ("news", "Deadline extended; more time for lobbying on both sides."),
        ("official", "Draft text published; amendments expected before final vote."),
        ("news", "Public support has dropped 5 points since last month."),
        ("news", "Key swing member said they are leaning yes."),
        ("wiki", "Historical precedent: 3 of 5 similar measures passed in past decade."),
        ("news", "No clear majority in latest survey; within margin of error."),
        ("official", "Vote postponed indefinitely pending further review."),
        ("news", "Major coalition announced support; passage likely."),
        ("news", "Legal challenge could delay implementation even if passed."),
        ("wiki", "Requires 60% threshold; current projections at 55%."),
        ("news", "Last-minute compromise reached; sponsors optimistic."),
        ("official", "Fiscal impact report: neutral to slightly positive."),
        ("news", "Turnout in early voting suggests strong yes turnout."),
        ("news", "One critical amendment failed; bill may be withdrawn."),
        ("wiki", "If passed, effective date would be January 1 next year."),
    ]
    items = []
    for i in range(n):
        src, content = templates[i % len(templates)]
        items.append(
            EvidenceItem(
                doc_id=f"doc_{i+1:02d}",
                source=src,
                title=f"Doc {i+1}",
                content=content,
                retrieval_score=0.9 - i * 0.02,
            )
        )
    return items


class StubAgent:
    """Returns fixed forecast when --dry-run (no API)."""
    def predict(self, qid, question, public_evidence, private_evidence, history_summary):
        n = len(public_evidence) + len(private_evidence)
        p = 0.55 if n > 10 else 0.5
        return Forecast.from_p_yes(p, rationale="[dry-run stub]", evidence_used=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Use stub agent, no API call")
    args = ap.parse_args()
    setup_logging()

    question = "Will the proposed measure pass the committee vote?"
    evidence = make_mock_evidence(20)
    ex = QuestionExample(qid="mock_q1", question=question, evidence=evidence)
    examples = [ex]

    if args.dry_run:
        agent = StubAgent()
        agent_factory = lambda _: StubAgent()
    else:
        from src.agents.llm_agent import LLMAgent
        agent = LLMAgent(temperature=0, max_tokens=2048)
        agent_factory = lambda _: LLMAgent(temperature=0, max_tokens=2048)
    aggregator = MeanAggregator()
    max_items = 20
    max_chars = 500
    seed = 42

    print("=== Mock question ===")
    print(question)
    print(f"\n=== Evidence: {len(evidence)} docs ===\n")

    # S0: single agent, full evidence
    print("--- S0: Single agent (full evidence) ---")
    s0_results = run_s0(examples, agent, max_items, max_chars)
    for qid, f in s0_results:
        print(f"  qid={qid}  p_yes={f.p_yes:.3f}  label={f.label}")
        print(f"  rationale: {f.rationale[:200]}...")
        print(f"  evidence_used: {f.evidence_used}")

    # S1: 3 agents, independent
    print("\n--- S1: 3 agents, independent ---")
    s1_results = run_s1(examples, agent_factory, aggregator, num_agents=3, seed=seed,
                        evidence_max_items=max_items, evidence_max_chars=max_chars)
    for qid, f in s1_results:
        print(f"  qid={qid}  p_yes={f.p_yes:.3f}  label={f.label}")

    # S2: 3 agents, 2 rounds iterative
    print("\n--- S2: 3 agents, 2 rounds iterative ---")
    s2_results = run_s2(examples, agent_factory, aggregator, num_agents=3, num_rounds=2, seed=seed,
                        evidence_max_items=max_items, evidence_max_chars=max_chars)
    for qid, f in s2_results:
        print(f"  qid={qid}  p_yes={f.p_yes:.3f}  label={f.label}")

    print("\nDone." + (" (dry-run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
