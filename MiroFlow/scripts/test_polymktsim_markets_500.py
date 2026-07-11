#!/usr/bin/env python3

"""Batch-test PolymktSim markets_500 with MiroFlow (no-tool config).

This script:
1. Reads markets_500.jsonl.
2. Builds each task question by concatenating the original question with all evidence.
3. Runs the same single-task execution path used by MiroFlow benchmark runner.
4. Computes Brier/LogLoss/Accuracy using PolymktSim's metrics implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config
from miroflow.agents import build_agent_from_config
from miroflow.benchmark.eval_utils import Task
from miroflow.benchmark.task_runner import run_single_task
from miroflow.logging.task_tracer import get_tracer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hydra_quote_value(value: str) -> str:
    """Quote override value for Hydra if it contains special characters."""
    # Hydra override parser breaks on unquoted spaces and some punctuation.
    if re.search(r"[\s:=,\[\]{}]", value):
        return "'" + value.replace("'", "\\'") + "'"
    return value


def load_market_p_yes_map(csv_path: Path) -> dict[str, float]:
    """Load mapping: qid (mkt_<id>) -> market p_yes from final_markets_500.csv."""
    mapping: dict[str, float] = {}
    if not csv_path.exists():
        return mapping

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = str(row.get("id", "")).strip()
            p_yes_raw = str(row.get("p_yes", "")).strip()
            if not market_id or not p_yes_raw:
                continue
            try:
                p_yes_val = float(p_yes_raw)
            except ValueError:
                continue
            if 0.0 <= p_yes_val <= 1.0:
                mapping[f"mkt_{market_id}"] = p_yes_val
    return mapping


def soft_log_loss_prob_target(p_yes: float, target_prob: float, eps: float = 1e-15) -> float:
    """Cross-entropy with soft target y in [0, 1]."""
    p = max(eps, min(1.0 - eps, p_yes))
    y = max(0.0, min(1.0, target_prob))
    return -((y * math.log(p)) + ((1.0 - y) * math.log(1.0 - p)))


def build_task_question(question: str, evidence_list: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    parts.append("You are given a binary prediction market question.")
    parts.append("Answer strictly with a final boxed YES or NO.")
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("All evidence:")

    for i, ev in enumerate(evidence_list, start=1):
        title = str(ev.get("title", "") or "").strip()
        source = str(ev.get("source", "") or "").strip()
        url = str(ev.get("url", "") or "").strip()
        content = str(ev.get("content", "") or "").strip()

        parts.append(f"[Evidence {i}]")
        if title:
            parts.append(f"Title: {title}")
        if source:
            parts.append(f"Source: {source}")
        if url:
            parts.append(f"URL: {url}")
        if content:
            parts.append("Content:")
            parts.append(content)
        parts.append("")

    return "\n".join(parts).strip()


def _extract_box_content(text: str) -> str:
    match = re.search(r"\\\\boxed\s*\{([^}]*)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_p_yes(model_boxed_answer: str) -> tuple[float, str]:
    """Map model boxed output to p_yes.

    Priority:
    1) Explicit YES/NO token.
    2) Numeric probability in [0,1] or percentage.
    3) Fallback to 0.5.
    """
    raw = _extract_box_content(model_boxed_answer)
    lowered = raw.lower()

    yes_hits = re.findall(r"\byes\b", lowered)
    no_hits = re.findall(r"\bno\b", lowered)
    if yes_hits and not no_hits:
        return 1.0, "YES"
    if no_hits and not yes_hits:
        return 0.0, "NO"

    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", raw)
    if pct:
        v = float(pct.group(1))
        v = max(0.0, min(100.0, v)) / 100.0
        return v, "YES" if v >= 0.5 else "NO"

    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", raw)
    for token in nums:
        try:
            v = float(token)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            return v, "YES" if v >= 0.5 else "NO"

    return 0.5, "YES"


async def run_one_task(cfg, agent, item: dict[str, Any], pass_at_k: int, max_retry: int):
    task_id = str(item.get("qid", ""))
    question = str(item.get("question", ""))
    evidence = item.get("evidence", []) or []

    task = Task(
        task_id=task_id,
        task_question=build_task_question(question, evidence),
        ground_truth="",
        file_path=None,
        metadata={},
    )

    result = await run_single_task(
        cfg=cfg,
        agent=agent,
        task=task,
        pass_at_k=pass_at_k,
        max_retry=max_retry,
        evaluator=None,
        exceed_max_turn_summary=cfg.benchmark.execution.get("exceed_max_turn_summary", False),
        prompt_manager=agent.prompt_manager if hasattr(agent, "prompt_manager") else None,
    )

    model_boxed_answer = result.model_boxed_answer or ""
    p_yes, label = parse_p_yes(model_boxed_answer)

    return {
        "qid": task_id,
        "question": question,
        "status": result.status,
        "model_boxed_answer": model_boxed_answer,
        "p_yes": p_yes,
        "label": label,
    }


async def run_batch(args) -> None:
    dotenv.load_dotenv()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "metrics_summary.json"
    market_csv_path = Path(args.market_csv_path).expanduser().resolve()
    market_p_yes_map = load_market_p_yes_map(market_csv_path)

    rows = load_jsonl(dataset_path)
    if args.start_index > 0:
        rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} items from {dataset_path}")
    print(f"Config: {args.config_path}")
    print(f"Output: {output_dir}")
    print(f"Market CSV: {market_csv_path} (loaded {len(market_p_yes_map)} ids)")

    output_dir_override = _hydra_quote_value(str(output_dir))
    cfg = load_config(args.config_path, f"output_dir={output_dir_override}")
    tracer = get_tracer()
    tracer.set_log_path(str(output_dir))

    agent = build_agent_from_config(cfg=cfg)
    pass_at_k = args.pass_at_k
    max_retry = args.max_retry

    print(f"Execution settings: pass_at_k={pass_at_k}, max_retry={max_retry}")

    with preds_path.open("w", encoding="utf-8") as fw:
        for idx, item in enumerate(rows, start=1):
            qid = item.get("qid", "")
            print(f"[{idx}/{len(rows)}] Running {qid}")
            pred = await run_one_task(cfg, agent, item, pass_at_k=pass_at_k, max_retry=max_retry)
            pred["market_p_yes"] = market_p_yes_map.get(str(qid))
            fw.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fw.flush()

    preds = load_jsonl(preds_path)
    with_market_prob = [
        p
        for p in preds
        if isinstance(p.get("market_p_yes"), (int, float))
    ]

    summary: dict[str, Any] = {
        "n_total": len(preds),
        "n_market_prob": len(with_market_prob),
    }

    if with_market_prob:
        proxy_brier = mean(
            (float(p["p_yes"]) - float(p["market_p_yes"])) ** 2 for p in with_market_prob
        )
        proxy_log_loss = mean(
            soft_log_loss_prob_target(float(p["p_yes"]), float(p["market_p_yes"]))
            for p in with_market_prob
        )
        proxy_dir_acc = mean(
            1.0
            if (float(p["p_yes"]) >= 0.5) == (float(p["market_p_yes"]) >= 0.5)
            else 0.0
            for p in with_market_prob
        )
        summary["proxy_vs_market_p_yes"] = {
            "brier": proxy_brier,
            "log_loss": proxy_log_loss,
            "directional_accuracy": proxy_dir_acc,
        }
    else:
        summary["proxy_vs_market_p_yes"] = None
        summary["note"] = "No market_p_yes found for this dataset slice."

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Metrics Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions saved to: {preds_path}")
    print(f"Summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MiroFlow on PolymktSim markets_500 with all evidence in task-question"
    )
    parser.add_argument(
        "--config-path",
        "--config",
        default="config/agent_quickstart_no_tools.yaml",
        help="MiroFlow config path (should be no-tools config)",
    )
    parser.add_argument(
        "--dataset-path",
        default="/Users/shadowfall/Library/Mobile Documents/com~apple~CloudDocs/临时代码/PolymktSim/data/processed/markets_500.jsonl",
        help="Path to PolymktSim test dataset (jsonl)",
    )
    parser.add_argument(
        "--market-csv-path",
        default="/Users/shadowfall/Library/Mobile Documents/com~apple~CloudDocs/临时代码/PolymktSim/data/outputs/final_markets_500.csv",
        help="Path to market CSV that provides p_yes target",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/polymktsim_markets_500_no_tools",
        help="Output directory for logs and metrics",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--pass-at-k", type=int, default=1, help="Pass@k attempts per task")
    parser.add_argument("--max-retry", type=int, default=1, help="Retries per attempt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()
