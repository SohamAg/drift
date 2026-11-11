"""User guidelines — drift's rule language for "what must never happen here".

The detector library ships nineteen generic coordination failures. The
things a team actually cares about are specific to their graph: *the
researcher must never call `send_email`*, *`issue_refund` only after
`approve_refund`*, *the final answer must cite a ticket number*,
*`budget_remaining` must stay non-negative*. Those are one line each,
and they should run for free on every run, baseline included.

Three surfaces, from cheapest to most expressive:

1. **Judge strings** — plain-English patterns appended to the LLM judge's
   prompt (`user_guidelines=`). Costs judge calls; catches what only a
   reader can.
2. **Rules** — the deterministic language below, loaded from a YAML file
   (`drift.yml` / `.drift/guidelines.yml`) or passed as dicts. Run inside
   the coordination library after the generic detectors; no LLM. Every
   violation is a `FailureRecord` with `failure_type="guideline:<id>"`, so
   `--fail-on finding:guideline:<id>` and the `guideline` gate rule work
   unchanged.
3. **Python** — `@drift.guideline("id")` on a function that receives a
   `RunView` and returns None (pass), a string (one violation) or a list
   of strings. Referenced from YAML as `python: pkg.mod:func`.

The rule kinds are deliberately few and literal — the same philosophy as
the gate: a rule a reviewer cannot read at a glance is a rule nobody
trusts. Each rule is a mapping with an `id` and exactly one kind key:

    rules:
      - id: no-raw-sql
        forbid_tool: execute_sql            # glob; `agent:` narrows who
      - id: must-hand-off
        require_tool: transfer_to_*         # must be called by the end
      - id: approve-first
        tool_order: {before: approve_refund, after: issue_refund}
      - id: search-budget
        max_tool_calls: {tool: web_search, count: 3}   # or a bare int
      - id: bounded
        max_steps: 25
      - id: cite-ticket
        final_answer: {must_match: "#\\\\d+"}            # / must_not_match
      - id: no-ssn
        no_message_match: "\\\\b\\\\d{3}-\\\\d{2}-\\\\d{4}\\\\b"  # any AI text or tool args
      - id: budget
        state: "budget_remaining >= 0"      # safe expression over the final state
      - id: custom
        python: mypkg.checks:no_duplicate_orders

Optional keys on every rule: `severity` (high | medium | low, default
high), `why` (free text echoed in the finding), `agent` (glob over the
calling agent / sub-agent, tool rules only).

`state:` expressions see the final state's top-level fields as names,
plus `n_steps`, `tool_calls` (list of tool names in call order),
`final_answer` (last AI text) and `state` (the dict). Operators:
comparisons, `and/or/not`, arithmetic, `in`, indexing, and the builtins
`len min max abs sum any all str int float bool sorted round`. Nothing
else — no attribute access, no calls to anything you did not list.

A rule that cannot be evaluated (bad regex, name missing from the state,
import failure) is itself a finding, `guideline:<id>:error`, never a
silent pass.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from drift.failures.base import FailureRecord
from drift.failures.library.base import (
    CoordinationDetectorContext,
    Msg,
    conversation,
)

FAILURE_PREFIX = "guideline:"
SEVERITIES = ("high", "medium", "low")
RULE_KINDS = (
    "forbid_tool", "require_tool", "tool_order", "max_tool_calls", "max_steps",
    "final_answer", "no_message_match", "state", "python",
)
#: Where `load_default()` looks, in order, relative to the working directory.
DEFAULT_FILES = ("drift.yml", "drift.yaml", ".drift/guidelines.yml", ".drift/guidelines.yaml")


class GuidelineError(ValueError):
    """A guideline file or rule mapping is malformed. Message names the rule."""


@dataclass
class Rule:
    id: str
    kind: str
    params: Any
    severity: str = "high"
    why: str = ""
    agent: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, self.kind: self.params}
        if self.kind == "python" and callable(self.params):
            fn = self.params
            out[self.kind] = f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}"
        if self.severity != "high":
            out["severity"] = self.severity
        if self.why:
            out["why"] = self.why
        if self.agent:
            out["agent"] = self.agent
        return out


@dataclass
class GuidelineSpec:
    """Everything a guideline file can declare."""
    judge: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    # Declarative chaos requests — consumed by the adapter's planner
    # (`drift.chaos.tools` for `tool:` entries, field patterns for `field:`).
    chaos: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.judge or self.rules or self.chaos)

    def rule_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rules]


# ---------------------------------------------------------------------------
# Loading / parsing
# ---------------------------------------------------------------------------


def load(path: str | Path) -> GuidelineSpec:
    """Load a guideline file. `.yml`/`.yaml` → full spec; anything else is
    read as one judge guideline per non-blank line (the original format)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yml", ".yaml"):
        spec = parse_yaml(text)
    else:
        spec = GuidelineSpec(judge=[ln.strip() for ln in text.splitlines() if ln.strip()])
    spec.source = str(p)
    return spec


def load_default(cwd: str | Path | None = None) -> GuidelineSpec | None:
    """The first of `DEFAULT_FILES` that exists under `cwd`, or None."""
    base = Path(cwd or ".")
    for name in DEFAULT_FILES:
        p = base / name
        if p.is_file():
            return load(p)
    return None


def parse_yaml(text: str) -> GuidelineSpec:
    try:
        import yaml  # local import: pyyaml is a hard dependency but keep import cost off the hot path
    except ImportError as exc:  # pragma: no cover
        raise GuidelineError("guideline YAML needs pyyaml (`pip install pyyaml`)") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise GuidelineError(f"guideline YAML does not parse: {exc}") from exc
    return parse(data)


def parse(data: Any) -> GuidelineSpec:
    """Build a spec from already-parsed YAML/JSON. Accepts:
      - a mapping with any of `judge` / `guidelines` (judge strings),
        `rules`, `chaos`;
      - a bare list of strings (judge guidelines only);
      - a bare list of rule mappings.
    """
    if data is None:
        return GuidelineSpec()
    if isinstance(data, str):
        return GuidelineSpec(judge=[data] if data.strip() else [])
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return GuidelineSpec(judge=[x.strip() for x in data if x.strip()])
        return GuidelineSpec(rules=parse_rules(data))
    if not isinstance(data, dict):
        raise GuidelineError(f"guideline file must be a mapping or a list; got {type(data).__name__}")
    judge_raw = data.get("judge", data.get("guidelines", [])) or []
    if isinstance(judge_raw, str):
        judge_raw = [judge_raw]
    if not isinstance(judge_raw, list) or not all(isinstance(x, str) for x in judge_raw):
        raise GuidelineError("`judge` must be a list of strings")
    chaos_raw = data.get("chaos") or []
    if not isinstance(chaos_raw, list) or not all(isinstance(x, dict) for x in chaos_raw):
        raise GuidelineError("`chaos` must be a list of mappings")
    unknown = set(data) - {"judge", "guidelines", "rules", "chaos", "version"}
    if unknown:
        raise GuidelineError(f"unknown top-level key(s): {', '.join(sorted(unknown))}; "
                             "expected judge, rules, chaos")
    return GuidelineSpec(
        judge=[x.strip() for x in judge_raw if x.strip()],
        rules=parse_rules(data.get("rules") or []),
        chaos=[dict(c) for c in chaos_raw],
    )


def parse_rules(raw: Iterable[Any]) -> list[Rule]:
    rules: list[Rule] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        rule = item if isinstance(item, Rule) else _parse_rule(item, i)
        if rule.id in seen:
            raise GuidelineError(f"duplicate rule id {rule.id!r}")
        seen.add(rule.id)
        rules.append(rule)
    return rules


def _parse_rule(item: Any, index: int) -> Rule:
    if callable(item) and getattr(item, "__drift_guideline__", None) is not None:
        return item.__drift_guideline__
    if callable(item):
        return Rule(id=getattr(item, "__name__", f"rule{index + 1}"), kind="python", params=item)
    if not isinstance(item, dict):
        raise GuidelineError(f"rule #{index + 1} must be a mapping; got {type(item).__name__}")
    item = dict(item)
    rid = str(item.pop("id", "") or "").strip()
    kinds = [k for k in RULE_KINDS if k in item]
    if not rid:
        raise GuidelineError(f"rule #{index + 1} has no `id`")
    if len(kinds) != 1:
        raise GuidelineError(
            f"rule {rid!r} must have exactly one of {', '.join(RULE_KINDS)}; got {kinds or 'none'}"
        )
    kind = kinds[0]
    params = item.pop(kind)
    severity = str(item.pop("severity", "high") or "high").lower()
    if severity not in SEVERITIES:
        raise GuidelineError(f"rule {rid!r}: severity must be one of {', '.join(SEVERITIES)}")
    why = str(item.pop("why", item.pop("description", "")) or "")
    agent = str(item.pop("agent", "") or "")
    if item:
        raise GuidelineError(f"rule {rid!r}: unknown key(s) {', '.join(sorted(item))}")
    _validate_params(rid, kind, params)
    return Rule(id=rid, kind=kind, params=params, severity=severity, why=why, agent=agent)


def _validate_params(rid: str, kind: str, params: Any) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise GuidelineError(f"rule {rid!r}: {kind} {msg}")

    if kind in ("forbid_tool", "require_tool"):
        need(isinstance(params, str) and bool(params.strip()), "needs a tool-name glob")
    elif kind == "tool_order":
        need(isinstance(params, dict) and isinstance(params.get("before"), str)
             and isinstance(params.get("after"), str), "needs {before: <glob>, after: <glob>}")
    elif kind == "max_tool_calls":
        if isinstance(params, dict):
            need(isinstance(params.get("count"), int) and params["count"] >= 0,
                 "needs {tool: <glob>, count: <int>}")
        else:
            need(isinstance(params, int) and not isinstance(params, bool) and params >= 0,
                 "needs an int or {tool: <glob>, count: <int>}")
    elif kind == "max_steps":
        need(isinstance(params, int) and not isinstance(params, bool) and params >= 1, "needs an int ≥ 1")
    elif kind == "final_answer":
        need(isinstance(params, dict) and set(params) <= {"must_match", "must_not_match"} and params,
             "needs must_match and/or must_not_match")
        for k, v in params.items():
            need(isinstance(v, str), f"{k} must be a regex string")
            _compile(rid, v)
    elif kind == "no_message_match":
        need(isinstance(params, str) and bool(params), "needs a regex string")
        _compile(rid, params)
    elif kind == "state":
        need(isinstance(params, str) and bool(params.strip()), "needs an expression string")
        _SafeEval.compile(params)  # raises GuidelineError on unsupported syntax
    elif kind == "python":
        need(callable(params) or (isinstance(params, str) and ":" in params),
             "needs 'package.module:function' or a callable")


def _compile(rid: str, pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.I | re.S)
    except re.error as exc:
        raise GuidelineError(f"rule {rid!r}: bad regex {pattern!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# The decorator (surface 3)
# ---------------------------------------------------------------------------


def guideline(id: str, *, severity: str = "high", why: str = "") -> Callable:
    """Mark a function as a guideline rule. The function receives a `RunView`
    and returns None / "" (pass), a string (one violation) or a list of
    strings. Pass the function (or its module path) in `rules=`.

        @drift.guideline("no-duplicate-orders", why="idempotency")
        def no_duplicate_orders(run):
            ids = [c.args.get("order_id") for c in run.tool_calls if c.name == "place_order"]
            if len(ids) != len(set(ids)):
                return f"place_order called twice for {ids}"
    """
    if severity not in SEVERITIES:
        raise GuidelineError(f"severity must be one of {', '.join(SEVERITIES)}")

    def deco(fn: Callable) -> Callable:
        fn.__drift_guideline__ = Rule(id=id, kind="python", params=fn, severity=severity, why=why)
        return fn
    return deco


# ---------------------------------------------------------------------------
# What a rule sees
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    args: Any
    step: int
    agent: str        # node that made the call ("researcher (tools/model)" when a sub-agent)
    index: int        # position in the conversation
    id: str = ""

    def matches(self, glob: str) -> bool:
        return _glob(self.name, glob)


@dataclass
class RunView:
    """Read-only view of one run handed to `state:` expressions and Python
    rules. Built from the coordination context; cheap to construct."""
    steps: list
    messages: list[Msg]
    tool_calls: list[ToolCall]
    final_state: dict[str, Any]
    final_answer: str
    ctx: CoordinationDetectorContext

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def agents(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            if s.agent and s.agent not in out:
                out.append(s.agent)
        return out

    def calls(self, glob: str = "*", *, agent: str = "") -> list[ToolCall]:
        return [c for c in self.tool_calls if c.matches(glob) and (not agent or _glob(c.agent, agent))]

    def ai_texts(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "ai" and m.step >= 1 and m.text]


def build_view(ctx: CoordinationDetectorContext) -> RunView:
    msgs = conversation(ctx)
    sub_by_step = {s.step: s.subagent for s in ctx.steps if s.subagent}
    calls: list[ToolCall] = []
    for m in msgs:
        if m.role != "ai" or m.step < 1:
            continue
        for tc in m.tool_calls:
            cid = str(tc.get("id") or "")
            if cid.startswith("drift_call_"):
                continue
            name = str(tc.get("name") or "").strip()
            if not name:
                continue
            sub = sub_by_step.get(m.step, "")
            agent = f"{sub} ({m.agent})" if sub else (m.agent or m.name or "")
            calls.append(ToolCall(name=name, args=tc.get("args"), step=m.step, agent=agent,
                                  index=m.index, id=cid))
    final_state = dict(ctx.steps[-1].state_after) if ctx.steps and ctx.steps[-1].state_after else {}
    final_answer = ""
    for m in reversed(msgs):
        if m.role == "ai" and m.step >= 1 and m.text and not m.tool_calls:
            final_answer = m.text
            break
    if not final_answer:
        for m in reversed(msgs):
            if m.role == "ai" and m.step >= 1 and m.text:
                final_answer = m.text
                break
    return RunView(steps=list(ctx.steps), messages=msgs, tool_calls=calls,
                   final_state=final_state, final_answer=final_answer, ctx=ctx)


def _glob(value: str, pattern: str) -> bool:
    if not pattern or pattern == "*":
        return True
    v = (value or "").lower()
    p = pattern.lower()
    # An agent label can be "critic (tools/model)"; match either half too.
    if fnmatch.fnmatchcase(v, p):
        return True
    if " (" in v:
        head, _, tail = v.partition(" (")
        return fnmatch.fnmatchcase(head, p) or fnmatch.fnmatchcase(tail.rstrip(")"), p)
    return False


# ---------------------------------------------------------------------------
# Safe expression evaluator for `state:` rules
# ---------------------------------------------------------------------------


class _SafeEval:
    _BUILTINS: dict[str, Callable] = {
        "len": len, "min": min, "max": max, "abs": abs, "sum": sum, "any": any, "all": all,
        "str": str, "int": int, "float": float, "bool": bool, "sorted": sorted, "round": round,
    }
    _BIN = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // b,
    }
    _CMP = {
        ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
        ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b,
    }

    @classmethod
    def compile(cls, expr: str) -> ast.Expression:
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as exc:
            raise GuidelineError(f"state expression {expr!r} does not parse: {exc.msg}") from exc
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr,
                                 ast.Starred, ast.FormattedValue, ast.JoinedStr)):
                raise GuidelineError(
                    f"state expression {expr!r}: {type(node).__name__} is not allowed "
                    "(names, comparisons, and/or/not, arithmetic, indexing, len/min/max/... only)"
                )
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in cls._BUILTINS:
                    raise GuidelineError(
                        f"state expression {expr!r}: only {', '.join(cls._BUILTINS)} may be called"
                    )
        return tree

    @classmethod
    def run(cls, expr: str, names: dict[str, Any]) -> Any:
        tree = cls.compile(expr)
        return cls._eval(tree.body, names)

    @classmethod
    def _eval(cls, node: ast.AST, names: dict[str, Any]) -> Any:
        ev = lambda n: cls._eval(n, names)  # noqa: E731
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in cls._BUILTINS:
                return cls._BUILTINS[node.id]
            if node.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[node.id]
            raise NameError(f"{node.id!r} is not a field of the final state "
                            f"(have: {', '.join(sorted(k for k in names if not k.startswith('_')))[:200]})")
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result: Any = True
                for v in node.values:
                    result = ev(v)
                    if not result:
                        return result
                return result
            result = False
            for v in node.values:
                result = ev(v)
                if result:
                    return result
            return result
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not v
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            raise ValueError(f"unsupported unary op {type(node.op).__name__}")
        if isinstance(node, ast.BinOp) and type(node.op) in cls._BIN:
            return cls._BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if not cls._CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(ev(node.lower) if node.lower else None, ev(node.upper) if node.upper else None,
                         ev(node.step) if node.step else None)
        if isinstance(node, ast.Call):
            fn = cls._BUILTINS[node.func.id]  # type: ignore[attr-defined]
            return fn(*[ev(a) for a in node.args], **{k.arg: ev(k.value) for k in node.keywords if k.arg})
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [ev(e) for e in node.elts]
            return vals if isinstance(node, ast.List) else tuple(vals)
        if isinstance(node, ast.Set):
            return {ev(e) for e in node.elts}
        if isinstance(node, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values) if k is not None}
        raise ValueError(f"unsupported syntax {type(node).__name__}")


def evaluate_state_expression(expr: str, view: RunView) -> Any:
    names: dict[str, Any] = dict(view.final_state)
    names.update({
        "state": view.final_state,
        "n_steps": view.n_steps,
        "tool_calls": [c.name for c in view.tool_calls],
        "final_answer": view.final_answer,
        "agents": view.agents,
    })
    return _SafeEval.run(expr, names)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _record(rule: Rule, step: int, summary: str, agents: list[str] | None = None,
            *, error: bool = False) -> FailureRecord:
    text = f"[{rule.severity}] {summary}"
    if rule.why:
        text += f" — {rule.why}"
    return FailureRecord(
        timestep=max(0, step),
        failure_type=f"{FAILURE_PREFIX}{rule.id}" + (":error" if error else ""),
        agents_involved=sorted({a for a in (agents or []) if a}),
        evidence_action_ids=[],
        summary=text,
        snapshot_timestep=max(0, step),
    )


def _preview(args: Any, limit: int = 60) -> str:
    try:
        text = args if isinstance(args, str) else json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(args)
    text = " ".join(str(text).split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _resolve_python(spec: Any) -> Callable:
    if callable(spec):
        return spec
    mod_name, _, fn_name = str(spec).partition(":")
    mod = importlib.import_module(mod_name)
    fn: Any = mod
    for part in fn_name.split("."):
        fn = getattr(fn, part)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def evaluate_rule(rule: Rule, view: RunView) -> list[FailureRecord]:
    """Apply one rule to one run. Never raises: an unevaluable rule yields a
    `guideline:<id>:error` finding."""
    try:
        return _evaluate(rule, view)
    except Exception as exc:  # noqa: BLE001 — surface, never hide
        last = view.steps[-1].step if view.steps else 0
        return [_record(rule, last, f"rule could not be evaluated: {type(exc).__name__}: {exc}", error=True)]


def _evaluate(rule: Rule, view: RunView) -> list[FailureRecord]:
    k, p = rule.kind, rule.params
    last = view.steps[-1].step if view.steps else 0

    if k == "forbid_tool":
        hits = view.calls(p, agent=rule.agent)
        if not hits:
            return []
        first = hits[0]
        who = f" by {first.agent}" if first.agent else ""
        more = f" ({len(hits)} calls)" if len(hits) > 1 else ""
        return [_record(rule, first.step,
                        f"forbidden tool {first.name!r} was called{who} at step {first.step} with "
                        f"{_preview(first.args)}{more}", [c.agent for c in hits])]

    if k == "require_tool":
        if not view.steps:
            return []  # nothing ran; the crash is reported elsewhere
        if view.calls(p, agent=rule.agent):
            return []
        who = f" by an agent matching {rule.agent!r}" if rule.agent else ""
        made = sorted({c.name for c in view.tool_calls})
        return [_record(rule, last,
                        f"required tool {p!r} was never called{who} in {view.n_steps} steps "
                        f"(tools called: {', '.join(made) or 'none'})")]

    if k == "tool_order":
        before, after = p["before"], p["after"]
        firsts_after = view.calls(after, agent=rule.agent)
        if not firsts_after:
            return []
        a = firsts_after[0]
        prior = [c for c in view.calls(before) if c.index < a.index]
        if prior:
            return []
        later = view.calls(before)
        tail = (f"; {before!r} only came at step {later[0].step}" if later
                else f"; {before!r} was never called")
        return [_record(rule, a.step,
                        f"{a.name!r} was called at step {a.step} before any {before!r}{tail}", [a.agent])]

    if k == "max_tool_calls":
        glob, limit = (p.get("tool", "*"), p["count"]) if isinstance(p, dict) else ("*", p)
        hits = view.calls(glob, agent=rule.agent)
        if len(hits) <= limit:
            return []
        over = hits[limit]
        return [_record(rule, over.step,
                        f"{len(hits)} calls to {glob!r} exceed the limit of {limit} "
                        f"(the {limit + 1}{_ordinal(limit + 1)} came at step {over.step})",
                        [c.agent for c in hits])]

    if k == "max_steps":
        if view.n_steps <= p:
            return []
        return [_record(rule, view.steps[p].step,
                        f"the run took {view.n_steps} steps; the limit is {p}")]

    if k == "final_answer":
        if not view.ai_texts():
            return []  # a graph with no AI messages has no final answer to judge
        out: list[FailureRecord] = []
        text = view.final_answer
        if "must_match" in p and not re.search(p["must_match"], text, re.I | re.S):
            out.append(_record(rule, last,
                               f"final answer does not match /{p['must_match']}/: {_preview(text, 100)!r}"))
        if "must_not_match" in p:
            m = re.search(p["must_not_match"], text, re.I | re.S)
            if m:
                out.append(_record(rule, last,
                                   f"final answer matches forbidden /{p['must_not_match']}/ "
                                   f"({m.group(0)[:40]!r})"))
        return out

    if k == "no_message_match":
        pat = re.compile(p, re.I | re.S)
        for m in view.messages:
            if m.role != "ai" or m.step < 1:
                continue
            blob = m.text
            for tc in m.tool_calls:
                blob += " " + _preview(tc.get("args"), 400)
            hit = pat.search(blob)
            if hit:
                return [_record(rule, m.step,
                                f"AI output at step {m.step} matches forbidden /{p}/ "
                                f"({hit.group(0)[:40]!r})", [m.agent])]
        return []

    if k == "state":
        if not view.steps:
            return []
        value = evaluate_state_expression(p, view)
        if value:
            return []
        shown = {name: view.final_state.get(name) for name in _names_in(p) if name in view.final_state}
        return [_record(rule, last, f"state expression `{p}` is false at the end of the run"
                        + (f" ({_preview(shown, 120)})" if shown else ""))]

    if k == "python":
        fn = _resolve_python(p)
        result = fn(view)
        if not result:
            return []
        items = [result] if isinstance(result, str) else list(result)
        return [_record(rule, last, str(item)) for item in items if item]

    raise GuidelineError(f"unknown rule kind {k!r}")


def _names_in(expr: str) -> list[str]:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []
    return sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)})


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def evaluate(rules: Iterable[Rule | dict | Callable], ctx: CoordinationDetectorContext) -> list[FailureRecord]:
    """Apply every rule to the run in `ctx`. Rules may be `Rule`s, rule
    mappings, or decorated/plain callables."""
    parsed = parse_rules(list(rules))
    if not parsed or not ctx.steps:
        return []
    view = build_view(ctx)
    out: list[FailureRecord] = []
    for rule in parsed:
        out.extend(evaluate_rule(rule, view))
    return out


def is_guideline_finding(failure_type: str) -> bool:
    return str(failure_type or "").startswith(FAILURE_PREFIX)


__all__ = [
    "DEFAULT_FILES", "FAILURE_PREFIX", "GuidelineError", "GuidelineSpec", "Rule", "RunView",
    "ToolCall", "build_view", "evaluate", "evaluate_rule", "guideline", "is_guideline_finding",
    "load", "load_default", "parse", "parse_rules", "parse_yaml",
]
