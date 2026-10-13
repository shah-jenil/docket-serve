"""Prompt rendering for typed decisions (the exact format the model was fine-tuned on).

Adapted from jevify (https://github.com/uspraveen/Jevify, commit a1308666, file
jevify/engine/template.py), Copyright 2026 Praveen Raj U S and Jevify contributors,
licensed under the Apache License 2.0. Changes: removed the Tev1 prompt format; imports
made package-local; no silent 12,000-character state truncation (see MAX_STATE_CHARS). Training, evaluation and serving all use this module, so the served
prompt is byte-identical to the training prompt.
"""
from __future__ import annotations

import hashlib
import json
import random
import string
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .wire import ChoiceQuestion, NoulQuestion, ScoreQuestion, parse_question

ReadoutMode = Literal["index", "label"]
PromptFormat = Literal["jevify", "tev1"]

SYSTEM_PROMPT = (
    "You are a System One model: you read a state and answer one question about it "
    "with exactly one of the allowed answers. Do not explain."
)
CONTENT_FREE_STATE = "N/A"
# jevify cut states at 12,000 characters. docket-serve does not truncate: the engine enforces a
# token limit and rejects over-long requests with HTTP 422 instead of silently dropping evidence.
# Set DOCKET_MAX_STATE_CHARS to restore a character cap (e.g. 12000 to match jevify exactly).
import os as _os
MAX_STATE_CHARS = int(_os.environ.get("DOCKET_MAX_STATE_CHARS", "0")) or None


@dataclass
class Rendered:
    prefix: str
    candidates: list[str]          # continuations to score, in presentation order
    keys: list[str]                # answer key for each candidate (option key / level index / "1","0")
    primitive: str
    mode: str
    presentation: list[int] = field(default_factory=list)   # permutation applied to the original key order
    question_hash: str = ""
    fmt: str = "jevify"            # which prompt format ``prefix`` is in (decides how it is wrapped for chat)

    def by_key(self, scores: list[float]) -> dict[str, float]:
        return dict(zip(self.keys, scores))


def render_state(state: Any) -> str:
    if isinstance(state, str):
        text = state
    else:
        text = json.dumps(state, indent=2, ensure_ascii=False)
    if MAX_STATE_CHARS and len(text) > MAX_STATE_CHARS:
        text = text[:MAX_STATE_CHARS] + "\n…[truncated]"
    return text


def _describe(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def default_identifiers(k: int) -> list[str]:
    """Fallback identifiers when no tokenizer-aware list is supplied: letters, then numbers."""
    if k <= 26:
        return list(string.ascii_uppercase[:k])
    return [str(i + 1) for i in range(k)]


CUE = "Answer: "   # trailing space: candidates follow without a leading space, which keeps digits,
                   # letters, two-letter ids, "yes" and "no" single tokens in every tokenizer we checked


def question_hash(question: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(question, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _assemble(state_block: str, qlines: list[str], state_last: bool) -> str:
    """Order the prompt. State first is the natural reading order. State *last* makes the
    question and its options a prefix shared by every request that asks that question, so
    a serving engine's prefix cache can reuse their KV blocks and a request pays only for
    its state -- the property that makes latency flat in the number of options."""
    if state_last:
        return "\n".join(qlines + ["", state_block, CUE])
    return "\n".join([state_block, ""] + qlines + [CUE])


def render(state: Any, question: dict[str, Any], *, mode: ReadoutMode = "index",
           permutation_seed: int | None = None, content_free: bool = False,
           identifiers: Sequence[str] | None = None, state_last: bool = False,
           fmt: PromptFormat = "jevify") -> Rendered:
    if fmt != "jevify":
        raise ValueError(f"unknown prompt format {fmt!r}")
    q = parse_question(question)
    body = render_state(CONTENT_FREE_STATE if content_free else state)
    state_block = f"State:\n{body}"
    qh = question_hash(question)
    instr = _describe(q.instructions).strip()

    if isinstance(q, NoulQuestion):
        qlines = [f"Question: {instr or 'Is the statement true?'}"]
        if q.criteria:
            if q.criteria.true:
                qlines.append(f"Answer yes if: {_describe(q.criteria.true)}")
            if q.criteria.false:
                qlines.append(f"Answer no if: {_describe(q.criteria.false)}")
        qlines.append("Allowed answers: yes, no")
        return Rendered(_assemble(state_block, qlines, state_last), ["yes", "no"], ["1", "0"], "noul", "yesno", [0, 1], qh)

    if isinstance(q, ScoreQuestion):
        levels = [_describe(c) for c in q.criteria]
        qlines = [f"Question: {instr or 'Which level applies?'}", "Levels (ordered from lowest to highest):"]
        qlines += [f"{i}. {d}" for i, d in enumerate(levels)]
        qlines.append(f"Allowed answers: {', '.join(str(i) for i in range(len(levels)))}")
        keys = [str(i) for i in range(len(levels))]
        return Rendered(_assemble(state_block, qlines, state_last), list(keys), keys, "score", "digit", list(range(len(keys))), qh)

    assert isinstance(q, ChoiceQuestion)
    keys = list(q.criteria.keys())
    order = list(range(len(keys)))
    if permutation_seed is not None:
        random.Random(permutation_seed).shuffle(order)
    shown = [keys[i] for i in order]
    qlines = [f"Question: {instr or 'Which option applies?'}", "Options:"]
    if mode == "index":
        ids = list(identifiers[: len(shown)]) if identifiers and len(identifiers) >= len(shown) else default_identifiers(len(shown))
        for ident, k in zip(ids, shown):
            desc = _describe(q.criteria[k])
            qlines.append(f"{ident}. {k}" + (f" — {desc}" if desc and desc != k else ""))
        qlines.append(f"Allowed answers: {', '.join(ids)}")
        return Rendered(_assemble(state_block, qlines, state_last), ids, shown, "choice", "index", order, qh)
    for k in shown:
        desc = _describe(q.criteria[k])
        qlines.append(f"- {k}" + (f": {desc}" if desc and desc != k else ""))
    qlines.append("Answer with exactly one option name.")
    return Rendered(_assemble(state_block, qlines, state_last), list(shown), shown, "choice", "label", order, qh)


def to_chat(prefix: str, tokenizer, *, system: str = SYSTEM_PROMPT) -> str:
    """Wrap a rendered prefix in the model's chat template, leaving the assistant
    turn open so candidates continue it. The trailing "Answer:" cue moves into
    the assistant turn so the model is scored on *its* answer."""
    user, cue = prefix.rsplit("\n" + CUE, 1)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            text = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            messages = messages[1:]   # no system role support
            text = tokenizer.apply_chat_template(messages, **kwargs)
    return text + CUE
