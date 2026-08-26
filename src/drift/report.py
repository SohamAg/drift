"""Evidence-pack report generator — `drift report`.

Turns a saved drift results JSON (the output of the example runners /
`drift_test` summaries) into a single self-contained HTML report: what
system was tested, what chaos was applied, what diverged, what the judge
found, and — optionally — fork-edit-replay proof that a hypothesized fix
closes a finding.

Why this exists as a first-class feature and not a notebook: the report
is the artifact a reviewer, auditor, or insurer actually consumes.
AIUC-1-style certification requires evidence of adversarial simulation
on a recurring cadence; this module renders drift's raw run data into
that evidence shape. The HTML is fully offline (no external fonts,
scripts, or network fetches) and print-friendly, so it can be archived
or exported to PDF as-is.

Input shapes supported (auto-detected):
  1. Multi-question runner output: {"config": {...}, "per_question": [...]}
     (examples/adapters/run_drift_on_langgraph_supervisor_extended.py)
  2. Single-row runner output: {"question": ..., "perturbations": [...]}
     (examples/adapters/run_drift_on_deepagents.py)

Optional fork proof JSON (from examples/adapters/fork_edit_demo.py
--save-json) adds a "Counterfactual proof" section.

Usage:
    drift report results/langgraph_supervisor_extended/extended_XXXX.json \
        --fork results/fork_demo/fork_XXXX.json --out report.html
"""
from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["load_run", "render_report", "write_report"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _norm_perturbation(p: dict) -> dict:
    """Normalize the two runner perturbation shapes into one."""
    tags = p.get("tags") or []
    if "CRASH" in tags or p.get("crashed"):
        outcome = "CRASH"
    elif "DIVERGE" in tags or p.get("diverged"):
        outcome = "DIVERGE"
    else:
        outcome = "UNCHANGED"
    return {
        "event": p.get("event", ""),
        "field": p.get("field", ""),
        "pattern": p.get("pattern", ""),
        "outcome": outcome,
        "error": p.get("error", ""),
        "divergence": p.get("divergence", ""),
        "judge_findings": p.get("judge_findings") or [],
        "coord_findings": (
            p.get("coordination_findings") or p.get("coord_findings") or []
        ),
    }


def _norm_question(q: dict) -> dict:
    perturbations = [_norm_perturbation(p) for p in (q.get("perturbations") or [])]
    return {
        "question": q.get("question", ""),
        "category": q.get("category", ""),
        "baseline_crashed": bool(q.get("baseline_crashed")),
        "baseline_trace_steps": q.get("baseline_trace_steps", 0),
        "agents": q.get("baseline_unique_agents") or [],
        "baseline_judge": q.get("baseline_judge") or [],
        "baseline_coord": q.get("baseline_coord") or [],
        "perturbations": perturbations,
    }


def load_run(path: str | Path) -> dict:
    """Load + normalize a results JSON; attach source integrity metadata."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8"))

    if "per_question" in data:
        questions = [_norm_question(q) for q in data["per_question"]]
        config = data.get("config") or {}
        elapsed = data.get("elapsed_seconds")
    else:
        questions = [_norm_question(data)]
        config = {
            k: data[k]
            for k in ("model", "intensity", "max_perturbations")
            if k in data
        }
        elapsed = data.get("_elapsed_s")

    all_perts = [p for q in questions for p in q["perturbations"]]
    judge_findings = [
        {"where": f"baseline · {q['category'] or 'run'}", **f}
        for q in questions
        for f in q["baseline_judge"]
    ] + [
        {"where": f"perturbed · {p['event']}", **f}
        for q in questions
        for p in q["perturbations"]
        for f in p["judge_findings"]
    ]
    coord_findings = [
        {"where": f"baseline · {q['category'] or 'run'}", **f}
        for q in questions
        for f in q["baseline_coord"]
    ] + [
        {"where": f"perturbed · {p['event']}", **f}
        for q in questions
        for p in q["perturbations"]
        for f in p["coord_findings"]
    ]

    agents: set[str] = set()
    for q in questions:
        agents.update(a for a in q["agents"] if a)

    return {
        "config": config,
        "elapsed_seconds": elapsed,
        "questions": questions,
        "summary": {
            "n_scenarios": len(questions),
            "n_runs": len(questions) + len(all_perts),
            "n_perturbations": len(all_perts),
            "n_crashed": sum(1 for p in all_perts if p["outcome"] == "CRASH"),
            "n_diverged": sum(1 for p in all_perts if p["outcome"] == "DIVERGE"),
            "n_unchanged": sum(1 for p in all_perts if p["outcome"] == "UNCHANGED"),
            "n_judge_findings": len(judge_findings),
            "n_coord_findings": len(coord_findings),
            "agents": sorted(agents),
        },
        "judge_findings": judge_findings,
        "coord_findings": coord_findings,
        "source": {
            "path": str(path),
            "name": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
    }


def load_fork_proof(path: str | Path) -> dict:
    """Load a fork-edit-replay proof JSON (fork_edit_demo --save-json)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source_name"] = path.name
    return data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_e = html.escape

_OUTCOME_CHIP = {
    "CRASH": "chip-critical",
    "DIVERGE": "chip-warning",
    "UNCHANGED": "chip-quiet",
}

_CSS = """
:root {
  --bg: #f6f8fc; --panel: #ffffff; --panel2: #f3f5fb;
  --fg: #0f172a; --fg-strong: #020617; --muted: #475569; --subtle: #94a3b8;
  --border: #e2e8f0; --border-strong: #cbd5e1;
  --accent: #4f46e5; --accent-soft: rgba(99, 102, 241, .10);
  --critical: #dc2626; --critical-soft: rgba(220, 38, 38, .09);
  --warning: #b45309; --warning-soft: rgba(180, 83, 9, .10);
  --success: #047857; --success-soft: rgba(4, 120, 87, .10);
  --info: #1d4ed8;
  --mono: ui-monospace, "Cascadia Code", "SF Mono", Consolas, Menlo, monospace;
  --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #07090f; --panel: #0f1320; --panel2: #131829;
    --fg: #e5e9f3; --fg-strong: #f5f7fc; --muted: #97a0b6; --subtle: #65708a;
    --border: #1d2235; --border-strong: #2a3147;
    --accent: #818cf8; --accent-soft: rgba(129, 140, 248, .14);
    --critical: #f87171; --critical-soft: rgba(248, 113, 113, .12);
    --warning: #fbbf24; --warning-soft: rgba(251, 191, 36, .12);
    --success: #34d399; --success-soft: rgba(52, 211, 153, .12);
    --info: #60a5fa;
  }
}
* { box-sizing: border-box; margin: 0; }
body {
  background: var(--bg); color: var(--fg);
  font: 15px/1.6 var(--sans);
  padding: 48px 24px 96px;
}
.page { max-width: 880px; margin: 0 auto; }
a { color: var(--accent); }

.masthead { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.wordmark {
  font: 700 22px/1 var(--mono); letter-spacing: -.02em; color: var(--fg-strong);
}
.wordmark::after { content: "▮"; color: var(--accent); margin-left: 2px; }
.doc-kind {
  font: 600 11px/1 var(--sans); letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); border: 1px solid var(--border-strong);
  padding: 5px 9px; border-radius: 3px;
}
h1 {
  font: 600 26px/1.25 var(--sans); letter-spacing: -.01em;
  color: var(--fg-strong); margin: 26px 0 6px; text-wrap: balance;
}
.subtitle { color: var(--muted); max-width: 64ch; }

.meta-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--border); border: 1px solid var(--border);
  border-radius: 6px; overflow: hidden; margin-top: 26px;
}
.meta-grid > div { background: var(--panel); padding: 10px 14px; }
.meta-grid dt {
  font: 600 10px/1 var(--sans); letter-spacing: .12em; text-transform: uppercase;
  color: var(--subtle); margin-bottom: 5px;
}
.meta-grid dd { font: 500 13px/1.4 var(--mono); color: var(--fg); overflow-wrap: anywhere; }

section { margin-top: 44px; }
.eyebrow {
  font: 600 11px/1 var(--sans); letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}
.eyebrow::after { content: ""; flex: 1; height: 1px; background: var(--border); }

.stat-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
}
.stat {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 14px 16px;
}
.stat .n {
  font: 600 28px/1.1 var(--mono); font-variant-numeric: tabular-nums;
  color: var(--fg-strong);
}
.stat .n.crit { color: var(--critical); }
.stat .n.warn { color: var(--warning); }
.stat .n.ok   { color: var(--success); }
.stat .l { font-size: 12px; color: var(--muted); margin-top: 4px; }

.finding {
  background: var(--panel); border: 1px solid var(--border);
  border-left: 3px solid var(--critical); border-radius: 6px;
  padding: 14px 18px; margin-bottom: 10px;
}
.finding .tag {
  font: 600 11px/1 var(--mono); color: var(--critical);
  background: var(--critical-soft); padding: 4px 8px; border-radius: 3px;
}
.finding .where { font: 500 11px/1 var(--mono); color: var(--subtle); margin-left: 8px; }
.finding p { margin-top: 9px; max-width: 70ch; }

.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; background: var(--panel); }
th {
  font: 600 10px/1 var(--sans); letter-spacing: .12em; text-transform: uppercase;
  color: var(--subtle); text-align: left; padding: 10px 14px;
  border-bottom: 1px solid var(--border-strong); background: var(--panel2);
  white-space: nowrap;
}
td {
  padding: 9px 14px; border-bottom: 1px solid var(--border);
  font-size: 13px; vertical-align: top;
}
tr:last-child td { border-bottom: none; }
td.mono { font-family: var(--mono); font-size: 12.5px; overflow-wrap: anywhere; }
td .div-excerpt {
  display: block; color: var(--muted); font: 400 11.5px/1.5 var(--mono);
  margin-top: 4px; max-width: 62ch; overflow-wrap: anywhere;
}
.chip {
  display: inline-block; font: 600 10.5px/1 var(--mono); letter-spacing: .04em;
  padding: 4px 8px; border-radius: 3px; white-space: nowrap;
}
.chip-critical { color: var(--critical); background: var(--critical-soft); }
.chip-warning  { color: var(--warning);  background: var(--warning-soft); }
.chip-success  { color: var(--success);  background: var(--success-soft); }
.chip-quiet    { color: var(--muted);    background: var(--panel2); }

.scenario { margin-bottom: 26px; }
.scenario-head {
  display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 10px;
}
.scenario-head .q {
  font-weight: 600; color: var(--fg-strong); max-width: 66ch; text-wrap: balance;
}
.scenario-head .cat { font: 500 11px/1 var(--mono); color: var(--subtle); }
.agent-strip { margin: 4px 0 10px; }
.agent-strip .chip { margin-right: 6px; margin-bottom: 4px; }

.fork-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 640px) { .fork-grid { grid-template-columns: 1fr; } }
.fork-panel {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 16px 18px;
}
.fork-panel h3 {
  font: 600 11px/1 var(--sans); letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 10px;
}
.fork-panel .verdict { font: 600 20px/1.2 var(--mono); }
.fork-panel .verdict.closed { color: var(--success); }
.fork-panel .verdict.open { color: var(--critical); }
.fork-panel p { font-size: 13px; color: var(--muted); margin-top: 8px; }
.edit-block {
  font: 12.5px/1.6 var(--mono); background: var(--panel2);
  border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px;
  margin: 12px 0; overflow-x: auto;
}

.method p { color: var(--muted); font-size: 13.5px; max-width: 72ch; margin-bottom: 10px; }
.method strong { color: var(--fg); }

footer {
  margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--border);
  font: 12px/1.7 var(--mono); color: var(--subtle);
}
footer .sha { overflow-wrap: anywhere; }

@media print {
  body { padding: 0; background: #fff; color: #0f172a; }
  .stat, .finding, .fork-panel, table { break-inside: avoid; }
}
"""


def _render_findings(findings: list[dict], kind: str) -> str:
    if not findings:
        return (
            f'<div class="finding" style="border-left-color: var(--success)">'
            f'<span class="tag" style="color:var(--success);background:var(--success-soft)">'
            f"none</span><p>No {kind} findings surfaced on this run.</p></div>"
        )
    rows = []
    for f in findings:
        ftype = _e(str(f.get("failure_type") or f.get("type") or "finding"))
        rows.append(
            '<div class="finding">'
            f'<span class="tag">{ftype}</span>'
            f'<span class="where">{_e(str(f.get("where", "")))}</span>'
            f"<p>{_e(str(f.get('summary', '')))}</p>"
            "</div>"
        )
    return "\n".join(rows)


def _render_scenarios(questions: list[dict]) -> str:
    out = []
    for i, q in enumerate(questions, 1):
        agent_chips = "".join(
            f'<span class="chip chip-quiet">{_e(a)}</span>' for a in q["agents"]
        )
        pert_rows = []
        for p in q["perturbations"]:
            chip = _OUTCOME_CHIP.get(p["outcome"], "chip-quiet")
            divergence = ""
            if p["divergence"]:
                divergence = (
                    f'<span class="div-excerpt">{_e(p["divergence"][:220])}</span>'
                )
            n_findings = len(p["judge_findings"]) + len(p["coord_findings"])
            findings_cell = (
                f'<span class="chip chip-critical">{n_findings}</span>'
                if n_findings
                else '<span class="chip chip-quiet">0</span>'
            )
            pert_rows.append(
                "<tr>"
                f'<td class="mono">{_e(p["event"])}{divergence}</td>'
                f'<td><span class="chip {chip}">{p["outcome"]}</span></td>'
                f"<td>{findings_cell}</td>"
                "</tr>"
            )
        table = (
            '<div class="table-wrap"><table>'
            "<thead><tr><th>Chaos event (auto-derived from state schema)</th>"
            "<th>Outcome</th><th>Findings</th></tr></thead>"
            f'<tbody>{"".join(pert_rows) or "<tr><td colspan=3>baseline only</td></tr>"}</tbody>'
            "</table></div>"
        )
        cat = f'<span class="cat">{_e(q["category"])}</span>' if q["category"] else ""
        out.append(
            f'<div class="scenario">'
            f'<div class="scenario-head"><span class="q">{i}. {_e(q["question"][:220])}</span>{cat}</div>'
            f'<div class="agent-strip">{agent_chips}</div>'
            f"{table}</div>"
        )
    return "\n".join(out)


def _render_fork_section(fork: dict) -> str:
    def _panel(title: str, closed: bool, desc: str) -> str:
        cls = "closed" if closed else "open"
        verdict = "CLOSED" if closed else "STILL FIRES"
        return (
            f'<div class="fork-panel"><h3>{_e(title)}</h3>'
            f'<div class="verdict {cls}">{verdict}</div><p>{_e(desc)}</p></div>'
        )

    edits = json.dumps(fork.get("edits", {}), indent=2)
    return f"""
<section>
  <div class="eyebrow">Counterfactual proof — fork-edit-replay</div>
  <p style="max-width:72ch; margin-bottom:14px">
    Finding <span class="chip chip-critical">{_e(str(fork.get("finding_type", "")))}</span>
    fired on the baseline run. Hypothesis: grounding the referenced entity in state
    closes it. Drift forked the run at step {_e(str(fork.get("fork_step", "?")))},
    applied the edit below, and replayed both the forked branch and a fresh run
    with the same edit at the initial state.
  </p>
  <div class="edit-block">{_e(edits)}</div>
  <div class="fork-grid">
    {_panel("Fork at step " + str(fork.get("fork_step", "?")), bool(fork.get("fork_closed")),
            "Edit applied mid-run at the fork point; downstream re-executed.")}
    {_panel("Same edit at initial state", bool(fork.get("top_closed")),
            "Edit applied at design level (initial state); full run re-executed.")}
  </div>
  <p style="margin-top:12px; color:var(--muted); font-size:13px; max-width:72ch">
    Both branches closing the finding is evidence the fix is legitimate at both the
    local (mid-run) and design (initial-state) level — not a path-dependent accident.
  </p>
</section>"""


def render_report(
    run: dict,
    *,
    fork: dict | None = None,
    title: str | None = None,
    target_name: str | None = None,
) -> str:
    cfg = run["config"]
    s = run["summary"]
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    title = title or "Drift reliability evidence"
    target = target_name or run["source"]["name"].rsplit(".", 1)[0]

    meta_items = [
        ("Target system", target),
        ("Model under test", str(cfg.get("model", "—"))),
        ("Chaos intensity", str(cfg.get("intensity", "—"))),
        ("Judge", "enabled" if cfg.get("use_judge", True) else "disabled"),
        ("Wall clock", f"{run['elapsed_seconds']}s" if run.get("elapsed_seconds") else "—"),
        ("Generated", generated),
    ]
    meta_html = "".join(
        f"<div><dt>{_e(k)}</dt><dd>{_e(v)}</dd></div>" for k, v in meta_items
    )

    agents_html = "".join(
        f'<span class="chip chip-quiet">{_e(a)}</span> ' for a in s["agents"]
    )

    n_findings = s["n_judge_findings"] + s["n_coord_findings"]
    stats = [
        (s["n_runs"], "total runs", ""),
        (s["n_perturbations"], "chaos perturbations", ""),
        (s["n_diverged"], "diverged", "warn" if s["n_diverged"] else ""),
        (s["n_crashed"], "crashed", "crit" if s["n_crashed"] else "ok"),
        (n_findings, "coordination findings", "crit" if n_findings else "ok"),
    ]
    stats_html = "".join(
        f'<div class="stat"><div class="n {cls}">{n}</div><div class="l">{lbl}</div></div>'
        for n, lbl, cls in stats
    )

    fork_html = _render_fork_section(fork) if fork else ""

    findings_all = run["judge_findings"] + run["coord_findings"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">

<div class="masthead">
  <span class="wordmark">drift</span>
  <span class="doc-kind">Reliability evidence</span>
</div>
<h1>{_e(title)}</h1>
<p class="subtitle">
  Adversarial simulation report for a multi-agent system. Chaos events are
  derived automatically from the system's own state schema; failures are
  detected by a coordination-failure taxonomy judge; counterfactual claims
  are verified by fork-edit-replay.
</p>

<dl class="meta-grid">{meta_html}</dl>

<section>
  <div class="eyebrow">Summary</div>
  <div class="stat-row">{stats_html}</div>
  <p style="margin-top:14px; color:var(--muted); font-size:13px">
    Agents observed in traces:&nbsp; {agents_html or "—"}
  </p>
</section>

<section>
  <div class="eyebrow">Findings</div>
  {_render_findings(findings_all, "coordination")}
</section>

<section>
  <div class="eyebrow">Scenarios &amp; chaos matrix</div>
  {_render_scenarios(run["questions"])}
</section>
{fork_html}
<section class="method">
  <div class="eyebrow">Methodology</div>
  <p><strong>Schema-walked chaos.</strong> Drift introspects the target graph's
  state schema at run start and derives type-appropriate perturbations
  (list clears, entry duplication, null injection, boundary values) — no
  hand-written fault list. Each perturbation is applied mid-run, between
  agent steps, against the shared state the agents coordinate through.</p>
  <p><strong>Tiered divergence detection.</strong> Perturbed runs are compared
  to baseline in escalating tiers: structural diff, exact match, semantic
  noise band, and finally an LLM judge — so expensive judgment is spent only
  where cheap comparison is inconclusive.</p>
  <p><strong>Coordination-failure taxonomy.</strong> The judge reports against
  named failure families (coordination contradiction, grounding failure,
  state drift, emergent decay, gate bypass) rather than free-form critique,
  and must cite trace evidence for every finding.</p>
  <p><strong>Fork-edit-replay.</strong> Any run can be forked at any step with
  a sparse state edit and replayed, so a hypothesized cause can be verified
  counterfactually rather than argued from the trace.</p>
</section>

<footer>
  <div>Generated by drift · {_e(generated)}</div>
  <div class="sha">source: {_e(run["source"]["name"])} · sha256 {_e(run["source"]["sha256"])}</div>
</footer>

</div>
</body>
</html>"""


def write_report(
    results_path: str | Path,
    *,
    fork_path: str | Path | None = None,
    out_path: str | Path | None = None,
    title: str | None = None,
    target_name: str | None = None,
) -> Path:
    run = load_run(results_path)
    fork = load_fork_proof(fork_path) if fork_path else None
    html_text = render_report(run, fork=fork, title=title, target_name=target_name)
    if out_path is None:
        out_path = Path(results_path).with_suffix(".report.html")
    out_path = Path(out_path)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path
