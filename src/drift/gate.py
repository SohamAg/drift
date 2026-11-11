"""Pass/fail gate over a shaped drift result — the CI half of drift.

`drift run` produces the shaped result dict (see drift.adapters.serialize);
this module decides whether it passes. The same rules run inside
`drift run --fail-on ...` and over a saved JSON with `drift gate`, so a
pipeline can split "run on the expensive box" from "decide in CI".

Rules are deliberately few and literal. A gate that needs a DSL to
express is a gate nobody will trust at 2am:

    crash                 any perturbation crashed (recursion-limit loops count)
    recursion_limit       any perturbation looped to LangGraph's recursion limit
    diverge               any perturbation diverged (after the noise cascade)
    intermittent          any perturbation's rollouts disagreed (needs rollouts > 1)
    finding               any coordination-detector finding, baseline or perturbed
    finding:<type>        a specific detector fired, e.g. finding:injected_instruction_followed
    baseline_finding      a detector fired on the UNPERTURBED run (a bug without chaos)
    guideline             any user guideline rule (drift.yml `rules:`) was violated
    judge                 any LLM-judge finding
    max_crash_rate=<0..1> crashed / perturbations above this fraction
    max_diverge_rate=<0..1>

Thresholds are fractions of scheduled perturbations, not of the schema's
pattern count, so a run with `max_perturbations=4` is judged on those 4.

A gate never passes vacuously: when rules are given and the run scheduled
ZERO perturbations (a light intensity on a one-field schema can do that),
the report carries a `no_perturbations` violation. Nothing was tested, so
nothing passed. Seen live: `drift run --target supervisor_basic --preset
quick` on a messages-only state → 0 perturbations → it would have said
PASS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Presets mirror web/app.js ADAPTER_PRESETS — keep both in step.
PRESETS: dict[str, dict[str, Any]] = {
    # quick is "moderate", not "light": light's 8% per-step draw yields ~2
    # perturbations on a 30-step horizon and the planner's floor is 1 — too
    # thin to call a smoke test. Moderate gives ~5, capped at 4, still free.
    "quick": dict(intensity="moderate", max_perturbations=4, judge="off",
                  divergence_mode="exact", baseline_rollouts=1, perturbation_rollouts=1,
                  max_judge_calls=0),
    "balanced": dict(intensity="aggressive", max_perturbations=6, judge="openai",
                     divergence_mode="tiered", baseline_rollouts=3, perturbation_rollouts=1,
                     max_judge_calls=10),
    "thorough": dict(intensity="aggressive", max_perturbations=12, judge="openai",
                     divergence_mode="tiered", baseline_rollouts=5, perturbation_rollouts=2,
                     max_judge_calls=25),
    "exhaustive": dict(intensity="exhaustive", max_perturbations=100, judge="openai",
                       divergence_mode="tiered", baseline_rollouts=5, perturbation_rollouts=3,
                       max_judge_calls=50),
}

SIMPLE_RULES = (
    "crash", "recursion_limit", "diverge", "intermittent", "finding",
    "baseline_finding", "guideline", "judge",
)
RATE_RULES = ("max_crash_rate", "max_diverge_rate")


@dataclass
class Violation:
    rule: str
    summary: str
    # Which perturbations (event names) triggered it; empty for baseline-only rules.
    perturbations: list[str] = field(default_factory=list)


@dataclass
class GateReport:
    rules: list[str]
    violations: list[Violation]
    n_perturbations: int

    @property
    def passed(self) -> bool:
        return not self.violations

    def lines(self) -> list[str]:
        if not self.rules:
            return ["gate: no rules given — nothing to fail on (use --fail-on)"]
        if self.passed:
            return [f"gate: PASS ({len(self.rules)} rule(s), {self.n_perturbations} perturbation(s))"]
        out = [f"gate: FAIL — {len(self.violations)} rule(s) violated"]
        for v in self.violations:
            out.append(f"  [{v.rule}] {v.summary}")
            for name in v.perturbations[:8]:
                out.append(f"      - {name}")
            if len(v.perturbations) > 8:
                out.append(f"      … and {len(v.perturbations) - 8} more")
        return out


def parse_rules(specs: list[str] | None) -> list[str]:
    """Normalise `--fail-on` values: comma-separated lists, whitespace, and
    validate every rule name. Raises ValueError with the offending token."""
    rules: list[str] = []
    for spec in specs or []:
        for raw in str(spec).split(","):
            tok = raw.strip()
            if not tok:
                continue
            _validate_rule(tok)
            if tok not in rules:
                rules.append(tok)
    return rules


def _validate_rule(tok: str) -> None:
    if tok in SIMPLE_RULES:
        return
    if tok.startswith("finding:") and tok[len("finding:"):].strip():
        return
    for rate in RATE_RULES:
        if tok.startswith(rate + "="):
            try:
                v = float(tok.split("=", 1)[1])
            except ValueError:
                raise ValueError(f"{tok!r}: expected a number after '='") from None
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{tok!r}: rate must be between 0 and 1")
            return
    raise ValueError(
        f"unknown gate rule {tok!r}; valid: {', '.join(SIMPLE_RULES)}, finding:<type>, "
        f"{', '.join(r + '=<0..1>' for r in RATE_RULES)}"
    )


def _finding_types(findings: list[dict] | None) -> list[str]:
    return [str(f.get("failure_type", "")) for f in (findings or [])]


def evaluate(shaped: dict[str, Any], rules: list[str]) -> GateReport:
    """Apply `rules` to a shaped result dict (web/CLI JSON shape)."""
    perts: list[dict] = list(shaped.get("perturbations") or [])
    baseline: dict = dict(shaped.get("baseline") or {})
    n = len(perts)
    violations: list[Violation] = []
    if rules and n == 0:
        violations.append(Violation(
            "no_perturbations",
            "the run scheduled no perturbations, so nothing was tested — raise the intensity "
            "(the quick preset is 'light', which can sample zero patterns on a one-field schema) "
            "or pass --max-perturbations",
        ))
    if rules and baseline.get("crashed"):
        violations.append(Violation(
            "baseline_crashed",
            f"the unperturbed run itself crashed ({baseline.get('error_type') or 'error'}: "
            f"{str(baseline.get('error') or '')[:120]}) — no comparison is possible",
        ))

    def names(pred) -> list[str]:
        return [str(p.get("event_name", "?")) for p in perts if pred(p)]

    for rule in rules:
        if rule == "crash":
            hit = names(lambda p: p.get("crashed"))
            if hit:
                violations.append(Violation(rule, f"{len(hit)}/{n} perturbation(s) crashed", hit))
        elif rule == "recursion_limit":
            hit = names(lambda p: p.get("crashed") and p.get("crash_class") == "recursion_limit")
            if hit:
                violations.append(Violation(rule, f"{len(hit)}/{n} perturbation(s) looped to the recursion limit", hit))
        elif rule == "diverge":
            hit = names(lambda p: p.get("diverged") and not p.get("crashed"))
            if hit:
                violations.append(Violation(rule, f"{len(hit)}/{n} perturbation(s) diverged", hit))
        elif rule == "intermittent":
            hit = names(lambda p: p.get("is_intermittent"))
            if hit:
                violations.append(Violation(
                    rule, f"{len(hit)}/{n} perturbation(s) gave different outcomes across rollouts", hit,
                ))
        elif rule == "finding":
            hit = names(lambda p: _finding_types(p.get("coordination_findings")))
            base = _finding_types(baseline.get("coordination_findings"))
            if hit or base:
                total = sum(len(p.get("coordination_findings") or []) for p in perts) + len(base)
                summary = f"{total} coordination finding(s)"
                if base:
                    summary += f" ({len(base)} on the unperturbed baseline)"
                violations.append(Violation(rule, summary, hit))
        elif rule.startswith("finding:"):
            want = rule[len("finding:"):].strip()
            hit = names(lambda p: want in _finding_types(p.get("coordination_findings")))
            in_base = want in _finding_types(baseline.get("coordination_findings"))
            if hit or in_base:
                where = f"{len(hit)}/{n} perturbation(s)" + (" and the baseline" if in_base else "")
                violations.append(Violation(rule, f"{want} fired on {where}", hit))
        elif rule == "baseline_finding":
            base = _finding_types(baseline.get("coordination_findings"))
            if base:
                violations.append(Violation(
                    rule, f"detector(s) fired on the unperturbed run: {', '.join(sorted(set(base)))}",
                ))
        elif rule == "guideline":
            def _gl(findings: list[dict] | None) -> list[str]:
                return [t for t in _finding_types(findings) if t.startswith("guideline:")]
            hit = names(lambda p: _gl(p.get("coordination_findings")))
            base = _gl(baseline.get("coordination_findings"))
            if hit or base:
                ids = sorted({t.split(":", 1)[1] for p in perts for t in _gl(p.get("coordination_findings"))}
                             | {t.split(":", 1)[1] for t in base})
                where = f"{len(hit)}/{n} perturbation(s)" + (" and the baseline" if base else "")
                violations.append(Violation(rule, f"guideline rule(s) {', '.join(ids)} violated on {where}", hit))
        elif rule == "guideline":
            def _gl(findings: list[dict] | None) -> list[str]:
                return [t for t in _finding_types(findings) if t.startswith("guideline:")]
            hit = names(lambda p: _gl(p.get("coordination_findings")))
            base = _gl(baseline.get("coordination_findings"))
            if hit or base:
                ids = sorted({t.split(":", 1)[1] for p in perts for t in _gl(p.get("coordination_findings"))}
                             | {t.split(":", 1)[1] for t in base})
                where = f"{len(hit)}/{n} perturbation(s)" + (" and the baseline" if base else "")
                violations.append(Violation(rule, f"guideline rule(s) {', '.join(ids)} violated on {where}", hit))
        elif rule == "guideline":
            def _gl(findings: list[dict] | None) -> list[str]:
                return [t for t in _finding_types(findings) if t.startswith("guideline:")]
            hit = names(lambda p: _gl(p.get("coordination_findings")))
            base = _gl(baseline.get("coordination_findings"))
            if hit or base:
                ids = sorted({t.split(":", 1)[1] for p in perts for t in _gl(p.get("coordination_findings"))}
                             | {t.split(":", 1)[1] for t in base})
                where = f"{len(hit)}/{n} perturbation(s)" + (" and the baseline" if base else "")
                violations.append(Violation(rule, f"guideline rule(s) {', '.join(ids)} violated on {where}", hit))
        elif rule == "judge":
            hit = names(lambda p: p.get("judge_findings"))
            base = baseline.get("judge_findings") or []
            if hit or base:
                violations.append(Violation(rule, f"{shaped.get('n_judge_findings', len(hit) + len(base))} judge finding(s)", hit))
        elif rule.startswith("max_crash_rate="):
            limit = float(rule.split("=", 1)[1])
            hit = names(lambda p: p.get("crashed"))
            rate = len(hit) / n if n else 0.0
            if rate > limit:
                violations.append(Violation(rule, f"crash rate {rate:.0%} > {limit:.0%}", hit))
        elif rule.startswith("max_diverge_rate="):
            limit = float(rule.split("=", 1)[1])
            hit = names(lambda p: p.get("diverged") and not p.get("crashed"))
            rate = len(hit) / n if n else 0.0
            if rate > limit:
                violations.append(Violation(rule, f"divergence rate {rate:.0%} > {limit:.0%}", hit))
    return GateReport(rules=list(rules), violations=violations, n_perturbations=n)


__all__ = ["PRESETS", "GateReport", "Violation", "evaluate", "parse_rules"]
