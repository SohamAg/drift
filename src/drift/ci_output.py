"""CI renderings of a shaped drift result — JUnit XML and a Markdown summary.

`drift run` / `drift gate` decide pass/fail (drift.gate). This module turns
the same shaped result plus its GateReport into the two formats CI systems
already know how to display:

* **JUnit XML** (`--junit`): one ``<testcase>`` for the unperturbed baseline,
  one per perturbation, and one per run-level gate violation (no
  perturbations scheduled, baseline crashed, findings on the baseline).
  A perturbation is a ``<failure>`` only when a gate rule named it, so the
  test report agrees with the exit code; a crash that no rule asked about
  is still recorded, as ``<system-out>``, not as a failure. GitHub, GitLab,
  Jenkins, CircleCI and Azure all render this format.

* **Markdown summary** (`--summary`, and automatically appended to
  ``$GITHUB_STEP_SUMMARY`` when that variable is set): the gate verdict,
  the counts, and a table of every perturbation with its outcome and
  findings — what a reviewer reads on the PR's checks page.

Both are pure functions of (shaped, report): no graph, no key, so
`drift gate results.json --junit out.xml` renders an artifact produced
elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from drift.gate import GateReport

__all__ = ["to_junit", "to_markdown", "write_github_step_summary"]

def _outcome(p: dict) -> str:
    if p.get("crashed"):
        return "crash" + (f" ({p['crash_class']})" if p.get("crash_class") else "")
    if p.get("diverged"):
        return "diverged"
    return "unchanged"


def _findings_text(p: dict) -> list[str]:
    out = [f"{f.get('failure_type')}: {f.get('summary', '')}" for f in p.get("coordination_findings") or []]
    out += [f"judge {f.get('failure_type', '')}: {f.get('summary', '')}" for f in p.get("judge_findings") or []]
    return out


def _violations_by_perturbation(report: GateReport) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for v in report.violations:
        for name in v.perturbations:
            by.setdefault(name, []).append(f"[{v.rule}] {v.summary}")
    return by


def to_junit(shaped: dict[str, Any], report: GateReport, *, suite_name: str | None = None) -> str:
    """Render a JUnit XML document (one ``<testsuite>``)."""
    perts: list[dict] = list(shaped.get("perturbations") or [])
    baseline: dict = dict(shaped.get("baseline") or {})
    graph = str(shaped.get("graph_name") or "graph")
    suite = suite_name or f"drift · {graph}"
    by_pert = _violations_by_perturbation(report)
    cases: list[str] = []
    failures = 0

    def case(name: str, time_s: float | None, failure: str | None, out_lines: list[str]) -> None:
        nonlocal failures
        attrs = f'classname={quoteattr("drift." + graph)} name={quoteattr(name)}'
        if time_s is not None:
            attrs += f' time="{float(time_s):.3f}"'
        body = ""
        if failure is not None:
            failures += 1
            first = failure.splitlines()[0] if failure else "gate rule violated"
            body += f"<failure message={quoteattr(first[:300])} type=\"drift.gate\">{escape(failure)}</failure>"
        if out_lines:
            body += f"<system-out>{escape(chr(10).join(out_lines))}</system-out>"
        cases.append(f"    <testcase {attrs}>{body}</testcase>" if body else f"    <testcase {attrs}/>")

    # Baseline: fails on baseline_crashed / baseline_finding and on rules
    # that counted baseline findings (finding, finding:<t>, guideline, judge).
    base_fail = [f"[{v.rule}] {v.summary}" for v in report.violations if v.baseline]
    b_out = [f"crashed: {baseline.get('error_type')}: {baseline.get('error')}"] if baseline.get("crashed") else []
    b_out += _findings_text(baseline)
    case("baseline (unperturbed)", baseline.get("duration_s"), "\n".join(base_fail) or None, b_out)

    for p in perts:
        name = str(p.get("event_name", "?"))
        out = [f"outcome: {_outcome(p)}"]
        if p.get("event_summary"):
            out.append(f"chaos: {p['event_summary']}")
        if p.get("fault_layer") == "tool":
            out.append("layer: tool" + ("" if p.get("fault_fired", True) else " (fault did not fire)"))
        if p.get("chaos_timing") == "mid_run":
            out.append(f"timing: mid-run at step {p.get('fork_step')}")
        if (p.get("n_rollouts") or 1) > 1:
            out.append(f"rollouts: {p.get('outcome_label', '')}" + (" (intermittent)" if p.get("is_intermittent") else ""))
        if p.get("crashed"):
            out.append(f"error: {p.get('error_type')}: {p.get('error')}")
        elif p.get("diverged") and p.get("divergence_summary"):
            out.append(f"divergence: {p['divergence_summary']}")
        out += _findings_text(p)
        fails = by_pert.get(name)
        case(name, p.get("duration_s"), "\n".join(fails) if fails else None, out)

    for v in report.violations:
        if v.rule == "no_perturbations":
            case(f"gate: {v.rule}", None, f"[{v.rule}] {v.summary}", [])

    total = len(cases)
    head = (
        f'<testsuite name={quoteattr(suite)} tests="{total}" failures="{failures}" errors="0" skipped="0">'
    )
    props = [
        ("drift.graph", graph),
        ("drift.intensity", shaped.get("intensity")),
        ("drift.seed", shaped.get("seed")),
        ("drift.chaos_timing", shaped.get("chaos_timing")),
        ("drift.rules", ",".join(report.rules)),
        ("drift.gate", "pass" if report.passed else "fail"),
    ]
    prop_xml = "".join(
        f"<property name={quoteattr(k)} value={quoteattr(str(v))}/>" for k, v in props if v not in (None, "")
    )
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<testsuites>", "  " + head,
             f"    <properties>{prop_xml}</properties>", *cases, "  </testsuite>", "</testsuites>"]
    return "\n".join(lines) + "\n"


def _md_cell(text: Any, limit: int = 140) -> str:
    s = " ".join(str(text or "").split()).replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def to_markdown(shaped: dict[str, Any], report: GateReport) -> str:
    """A GitHub-flavoured Markdown summary of the run and the gate."""
    perts: list[dict] = list(shaped.get("perturbations") or [])
    graph = shaped.get("graph_name") or "graph"
    if not report.rules:
        verdict = "⚪ no gate rules"
    elif report.passed:
        verdict = "✅ PASS"
    else:
        verdict = "❌ FAIL"
    n_crash = sum(1 for p in perts if p.get("crashed"))
    n_div = sum(1 for p in perts if p.get("diverged") and not p.get("crashed"))
    n_find = sum(len(p.get("coordination_findings") or []) for p in perts) + len(
        (shaped.get("baseline") or {}).get("coordination_findings") or []
    )
    out = [
        f"## drift · `{graph}` — {verdict}",
        "",
        f"{len(perts)} perturbation(s) · {n_crash} crashed · {n_div} diverged · {n_find} coordination finding(s)"
        f" · intensity `{shaped.get('intensity')}` · timing `{shaped.get('chaos_timing', 'initial')}`"
        f" · seed `{shaped.get('seed')}`",
        "",
    ]
    if report.rules:
        out.append("**Gate rules:** " + ", ".join(f"`{r}`" for r in report.rules))
        out.append("")
    if report.violations:
        out.append("| Violated rule | What happened |")
        out.append("|---|---|")
        for v in report.violations:
            out.append(f"| `{v.rule}` | {_md_cell(v.summary, 200)} |")
        out.append("")
    base = shaped.get("baseline") or {}
    if base.get("crashed") or base.get("coordination_findings"):
        out.append("**Baseline (no chaos):** " + (
            f"crashed — {_md_cell(base.get('error_type'))}" if base.get("crashed")
            else ", ".join(f"`{f.get('failure_type')}`" for f in base["coordination_findings"])
        ))
        out.append("")
    if perts:
        flagged = {n for v in report.violations for n in v.perturbations}
        out.append("<details><summary>Every perturbation</summary>\n")
        out.append("| | Perturbation | Outcome | Findings |")
        out.append("|---|---|---|---|")
        order = sorted(perts, key=lambda p: (0 if p.get("event_name") in flagged else 1,
                                             0 if p.get("crashed") else 1 if p.get("diverged") else 2))
        for p in order:
            mark = "❌" if p.get("event_name") in flagged else ("⚠️" if p.get("crashed") or p.get("diverged") else "·")
            outcome = _outcome(p)
            if (p.get("n_rollouts") or 1) > 1:
                outcome += f" ({p.get('outcome_label', '')})"
            finds = ", ".join(f"`{f.get('failure_type')}`" for f in p.get("coordination_findings") or []) or "—"
            out.append(f"| {mark} | `{_md_cell(p.get('event_name'), 80)}` | {_md_cell(outcome)} | {finds} |")
        out.append("\n</details>")
        out.append("")
    return "\n".join(out) + "\n"


def write_github_step_summary(markdown: str, env: dict[str, str] | None = None) -> Path | None:
    """Append to ``$GITHUB_STEP_SUMMARY`` when running inside GitHub Actions.
    Returns the path written, or None when the variable is unset."""
    target = (env if env is not None else os.environ).get("GITHUB_STEP_SUMMARY")
    if not target:
        return None
    path = Path(target)
    with path.open("a", encoding="utf-8") as f:
        f.write(markdown)
    return path
