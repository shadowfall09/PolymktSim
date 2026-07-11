from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_call(qid: str, agent_id: int | None, round_id: int, forecast: Any, tokens: int = 0) -> None:
    logger.info("qid=%s agent_id=%s round=%s tokens=%s", qid, agent_id, round_id, tokens)
