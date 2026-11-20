"""Detector: the final answer states a precise number that nothing in the
run supplied.

Failure: an agent reports "Tokyo has 13,929,286 people" when no tool result,
user message or system prompt contained that figure. The number reads as
researched and is not. Live (drift `handoff_misroute` and
`skip_node[research_expert]` on the 2-agent langgraph-supervisor system,
gpt-4o-mini, 2/2 rollouts each): the research step never ran, math_expert
produced precise populations from its own weights, and the supervisor passed
them on as the answer. `empty_handoff` names the broken handoff; this
detector names the consequence: the answer contains specifics with no source.

Real-world causes: a retrieval step skipped, misrouted or returning empty;
a tool error swallowed; a sub-agent answering from memory instead of
calling its tool. Grounding broke and the answer did not say so.

Deferred once (ADR-0008) because the bare heuristic, "a number in the answer
that appears in no tool result", also fires on correct arithmetic and on
well-known constants. Both guards are implemented here (ADR-0010):

Fires once per run when ALL hold:
  - the run has at least one successful tool result (grounding was
    available: an agent with no tools answering from knowledge is not this
    failure),
  - the final answer (last AI message with text and no tool calls) states a
    *specific* number: at least 4 significant digits, not a bare year
    (1000-2100), not part of an identifier, date or version string,
  - that number does not match, at the precision the answer states it, any
    number in a human / system / tool message or the initial state
    ("13.93 million" matches 13,929,286),
  - it is not a simple computation over those numbers: the sum, difference,
    product, ratio or percentage of any two, or the sum of any three or of
    all of them,
  - it is not a well-known constant (speed of light, pi, e, Avogadro,
    Planck, absolute zero, seconds per year, powers of two...), compared
    scale-free so 299,792 km/s and 299,792,458 m/s both match.
Numbers the model wrote into its own tool-call arguments are not grounding:
they are where fabricated figures usually first appear.
"""
from __future__ import annotations

import math
import re
from itertools import combinations

from drift.failures.base import FailureRecord
from drift.failures.library.base import CoordinationDetectorContext, collect_string_values, conversation

NAME = "ungrounded_number"
SUMMARY = (
    "The final answer states a precise number that appears in no tool result, user message or "
    "prompt, and is not a computation over them: a figure presented as researched that was not."
)
MAST_MODES = ["3.3"]  # incorrect verification (unsupported claims pass through)
SOURCE = (
    "MAST (arXiv:2503.13657) 3.3 Incorrect verification; drift live runs 2026-11-19 "
    "(supervisor skip/misroute: fabricated populations passed on); ADR-0008/0010"
)

MIN_SIG_DIGITS = 4
_MAX_TRIPLE_POOL = 25
_MAX_REPORTED = 5

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUM = re.compile(
    r"(?<![\w.,/:\-])"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?![\w/:\-]|\.\d|,\d{3})"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion)\b)?",
    re.I,
)

# Mantissas (1 <= m < 10) of well-known constants; matched scale-free.
_CONSTANTS = [
    2.99792458,   # speed of light (m/s, km/s)
    1.86282,      # speed of light (mi/s)
    3.14159265,   # pi
    2.71828183,   # e
    6.02214076,   # Avogadro
    6.62607015,   # Planck
    6.6743,       # gravitational constant
    1.602176634,  # elementary charge
    9.1093837,    # electron mass
    1.67262192,   # proton mass
    1.380649,     # Boltzmann
    8.314462,     # gas constant
    2.7315,       # 273.15 (0 degC in K)
    4.5967,       # -459.67 (0 K in degF)
    1.41421356,   # sqrt 2
    1.61803399,   # golden ratio
    3.1536,       # seconds per 365-day year
    3.15576,      # seconds per Julian year
    6.371,        # Earth mean radius (km)
    1.495978707,  # astronomical unit
    9.4607,       # light year
]
_CONSTANTS += [float(f"{2 ** k / 10 ** (len(str(2 ** k)) - 1):.12g}") for k in range(10, 41)]


def _parse(text: str) -> list[tuple[float, int, str]]:
    """(value, significant digits, as written) for every number in `text`."""
    out = []
    for m in _NUM.finditer(text or ""):
        raw = m.group("num")
        digits = raw.replace(",", "")
        mant = digits.split("e")[0].split("E")[0]
        try:
            value = float(digits)
        except ValueError:
            continue
        if "." in mant:
            sig = len(mant.replace(".", "").lstrip("0"))
        else:
            sig = len(mant.lstrip("0").rstrip("0")) or 1
        scale = (m.group("scale") or "").lower()
        if scale:
            value *= _SCALE[scale]
        written = m.group(0).strip()
        out.append((value, sig, written))
    return out


def _is_year(value: float, written: str) -> bool:
    return value.is_integer() and 1000 <= value <= 2100 and written.isdigit()


def _ulp_half(value: float, sig: int) -> float:
    """Half a unit in the last place of `value` written to `sig` digits."""
    if value == 0:
        return 0.0
    exp = math.floor(math.log10(abs(value)))
    return 0.5 * 10 ** (exp - sig + 1)


def _matches(value: float, sig: int, candidate: float) -> bool:
    return abs(value - candidate) <= _ulp_half(value, sig) * (1 + 1e-9)


def _mantissa(value: float) -> float:
    v = abs(value)
    return v / 10 ** math.floor(math.log10(v)) if v else 0.0


def _is_constant(value: float, sig: int) -> bool:
    if value == 0:
        return False
    m = _mantissa(value)
    return any(_matches(m, sig, c) for c in _CONSTANTS)


def _derived(pool: list[float]) -> list[float]:
    """Simple computations over grounded numbers."""
    out: list[float] = []
    for a, b in combinations(pool, 2):
        out += [a + b, abs(a - b), a * b]
        if b:
            out += [a / b, a / b * 100]
        if a:
            out += [b / a, b / a * 100]
    if len(pool) <= _MAX_TRIPLE_POOL:
        out += [a + b + c for a, b, c in combinations(pool, 3)]
    if len(pool) > 3:
        out.append(sum(pool))
    return out


class _Pool:
    """Grounded numbers, in the order the run supplied them."""

    def __init__(self) -> None:
        self.values: list[float] = []
        self._seen: set[float] = set()

    def add_text(self, text: str) -> None:
        for value, _sig, _w in _parse(text):
            if value not in self._seen:
                self._seen.add(value)
                self.values.append(value)


def _arg_text(args) -> str:
    """Tool-call arguments as text, numbers written plainly (13929286, not 1.39e7)."""
    parts: list[str] = []

    def walk(v, depth=0):
        if depth > 8:
            return
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            parts.append(str(v))
        elif isinstance(v, float):
            parts.append(repr(v) if not v.is_integer() else str(int(v)))
        elif isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x, depth + 1)

    walk(args)
    return " ; ".join(parts)


def ungrounded_in(answer: str, grounded: list[float]) -> list[str]:
    """Numbers in `answer` (as written) that `grounded` does not explain."""
    derived: list[float] | None = None
    flagged: list[str] = []
    for value, sig, written in _parse(answer):
        if sig < MIN_SIG_DIGITS or _is_year(value, written):
            continue
        if any(_matches(value, sig, g) for g in grounded):
            continue
        if _is_constant(value, sig):
            continue
        if derived is None:
            derived = _derived(grounded)
        if any(_matches(value, sig, d) for d in derived):
            continue
        if written not in flagged:
            flagged.append(written)
    return flagged


def _shown(nums: list[str]) -> str:
    extra = len(nums) - _MAX_REPORTED
    return ", ".join(nums[:_MAX_REPORTED]) + (f" (+{extra} more)" if extra > 0 else "")


def detect(ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    msgs = conversation(ctx)
    if not any(m.role == "tool" and not m.is_error_result and not m.is_drift_injected for m in msgs):
        return []
    final = next((m for m in reversed(msgs) if m.role == "ai" and m.text.strip() and not m.tool_calls), None)
    if final is None:
        return []

    # Walk the run in order. A tool call whose arguments carry a number the
    # run never supplied is where a fabricated figure enters; its result is
    # computed from that figure, so it is not grounding (otherwise
    # "13,929,286" is explained backwards as 6,964,643 / 0.5).
    pool = _Pool()
    if ctx.initial_state:
        strs: list[str] = []
        collect_string_values({k: v for k, v in ctx.initial_state.items() if k != "messages"}, strs)
        for s in strs:
            pool.add_text(s)
    tainted_ids: set[str] = set()
    pending_single: str | None = None
    fabricated_args: list[tuple[str, str, list[str]]] = []   # (caller, tool, numbers)
    for m in msgs:
        if m.role in ("human", "system"):
            pool.add_text(m.text)
        elif m.role == "tool":
            tid = m.tool_call_id or pending_single
            if tid and tid in tainted_ids:
                continue
            if not m.is_error_result:
                pool.add_text(m.text)
        elif m.role == "ai" and m.tool_calls and not m.is_drift_injected:
            pending_single = None
            for tc in m.tool_calls:
                nums = ungrounded_in(_arg_text(tc.get("args")), pool.values)
                if not nums:
                    continue
                tid = str(tc.get("id") or "") or f"__call_{m.index}"
                tainted_ids.add(tid)
                if len(m.tool_calls) == 1 and not tc.get("id"):
                    pending_single = tid
                fabricated_args.append((m.name or m.agent or "an agent", str(tc.get("name") or "a tool"), nums))

    flagged = ungrounded_in(final.text, pool.values)
    if not flagged and not fabricated_args:
        return []

    who = final.name or final.agent or "the final answer"
    parts: list[str] = []
    if fabricated_args:
        caller, tool, nums = fabricated_args[0]
        more = f" (and {len(fabricated_args) - 1} more call{'s' if len(fabricated_args) > 2 else ''})" if len(fabricated_args) > 1 else ""
        parts.append(f"{caller} called {tool} with {_shown(nums)}{more}, which nothing in the run supplied")
    if flagged:
        parts.append(
            f"{who}'s final answer states {_shown(flagged)}, which no tool result, user message or prompt "
            f"contained{' except via those calls' if fabricated_args else ''} and no simple computation over them gives"
        )
    else:
        parts.append(f"{who}'s final answer rests on {'that call' if len(fabricated_args) == 1 else 'those calls'}")
    agents = sorted({a for a in [who] + [c for c, _t, _n in fabricated_args] if a})
    return [FailureRecord(
        timestep=final.step,
        failure_type=NAME,
        agents_involved=agents,
        evidence_action_ids=[],
        summary="; ".join(parts) + ". Presented as researched; the run never grounded it",
        snapshot_timestep=final.step,
    )]


def detect_from_text(transcript: str) -> list[FailureRecord]:
    """No text-only variant: needs structured tool results to know what was grounded."""
    return []


__all__ = ["MAST_MODES", "NAME", "SOURCE", "SUMMARY", "detect", "detect_from_text", "ungrounded_in"]
