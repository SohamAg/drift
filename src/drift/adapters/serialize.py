"""JSON shaping for adapter results — shared by the web server and the CLI.

One function, `shape_adapter_result`, produces the dict that the web UI
renders, `drift run --json` writes, `drift gate` reads, the HTML report
consumes and regression tracking snapshots. It lives here rather than in
server.py so the CLI works without the `web` extra installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def shape_adapter_result(
    result: Any,
    *,
    graph_meta: dict[str, Any],
    seed: int,
    judge: str,
    judge_model: str | None,
    user_guidelines: list[str] | None,
    track_history: bool = False,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    """Turn an AdapterResult into the JSON-safe dict every drift surface
    speaks: the web UI (Test / Import / Results tabs), `drift run --json`,
    `drift gate`, the HTML report and regression tracking all read this
    shape. Keep it dependency-free — no FastAPI here — so the CLI can emit
    it without the web extra installed.

    `graph_meta` is `{"name", "description", "agents"}`. `track_history`
    appends a snapshot to `history_dir` (drift.regression) and includes the
    regression report.
    """
    def _divergence_dict(d: Any) -> dict[str, Any]:
        return {
            "name": d.name,
            "tier": d.tier,
            "baseline_value": d.baseline_value,
            "perturbed_value": d.perturbed_value,
            "summary": d.summary,
            "similarity_score": d.similarity_score,
            "within_noise_band": d.within_noise_band,
            "judge_equivalent": d.judge_equivalent,
            "judge_reasoning": d.judge_reasoning,
        }

    def _baseline_dict(b: Any) -> dict[str, Any]:
        return {
            "initial_state": b.initial_state,
            "final_state": b.final_state,
            "crashed": b.crashed,
            "error": b.error,
            "error_type": b.error_type,
            "crash_class": getattr(b, "crash_class", ""),
            "duration_s": round(b.duration_s, 4),
            "trace": b.trace,
            "judge_findings": b.judge_findings,
            "coordination_findings": b.coordination_findings,
            "thread_id": getattr(b, "thread_id", ""),
        }

    def _pert_dict(p: Any, *, with_rollouts: bool = True) -> dict[str, Any]:
        n_rollouts = getattr(p, "n_rollouts", 1)
        d = {
            "event_name": p.event_name,
            "event_summary": p.event_summary,
            "perturbed_field": p.perturbed_field,
            "pattern_type": p.pattern_type,
            "fault_class": getattr(p, "fault_class", ""),
            "perturbed_initial_state": p.perturbed_initial_state,
            "final_state": p.final_state,
            "crashed": p.crashed,
            "error": p.error,
            "error_type": p.error_type,
            "crash_class": getattr(p, "crash_class", ""),
            "diverged": p.diverged,
            "divergence_summary": p.divergence_summary,
            "duration_s": round(p.duration_s, 4),
            "trace": p.trace,
            "judge_findings": p.judge_findings,
            "coordination_findings": p.coordination_findings,
            "divergence_details": [_divergence_dict(d) for d in p.divergence_details],
            "filtered_divergences": [_divergence_dict(d) for d in p.filtered_divergences],
            "thread_id": getattr(p, "thread_id", ""),
            # Repeated rollouts (perturbation_rollouts > 1). The row itself is
            # the most severe rollout; `rollouts` holds the others in full.
            "rollout_index": getattr(p, "rollout_index", 0),
            "n_rollouts": n_rollouts,
        }
        if with_rollouts and n_rollouts > 1:
            d["outcome_counts"] = p.outcome_counts
            d["is_intermittent"] = p.is_intermittent
            d["finding_rollout_counts"] = p.finding_rollout_counts
            d["outcome_label"] = p.outcome_label()
            d["rollouts"] = [
                dict(_pert_dict(r, with_rollouts=False), n_rollouts=n_rollouts)
                for r in p.rollouts
            ]
        return d

    noise_band_dump = {
        name: {
            "name": band.name,
            "sample_count": band.sample_count,
            "distinct_values": band.distinct_values,
            "value_frequencies": band.value_frequencies,
            "text_min_similarity": band.text_min_similarity,
            "text_mean_similarity": band.text_mean_similarity,
            "numeric_min": band.numeric_min,
            "numeric_max": band.numeric_max,
        }
        for name, band in result.noise_band.items()
    }

    regression_report = None
    if track_history:
        from drift import regression

        regression_report = regression.track(
            result, graph_meta["name"],
            history_dir=history_dir if history_dir is not None else regression.DEFAULT_HISTORY_DIR,
        ).to_json()

    return {
        "graph_name": graph_meta["name"],
        "graph_description": graph_meta["description"],
        "graph_agents": graph_meta.get("agents", []),
        "intensity": result.intensity,
        "seed": seed,
        "judge": judge,
        "judge_model": judge_model,
        "n_user_guidelines": len(user_guidelines or []),
        "divergence_mode": result.divergence_mode,
        "baseline_rollouts": result.baseline_rollouts,
        "perturbation_rollouts": getattr(result, "perturbation_rollouts", 1),
        "n_intermittent": getattr(result, "n_intermittent", 0),
        "n_graph_runs": getattr(result, "n_graph_runs", None),
        "judge_calls_used": result.judge_calls_used,
        "judge_calls_budget": result.judge_calls_budget,
        "noise_band": noise_band_dump,
        "patterns_total": result.patterns_total,
        "n_crashed": result.n_crashed,
        "n_diverged": result.n_diverged,
        "n_unchanged": result.n_unchanged,
        "n_judge_findings": result.n_judge_findings,
        "n_coordination_findings": result.n_coordination_findings,
        "n_filtered_divergences": result.n_filtered_divergences,
        "summary_lines": result.summary_lines(),
        "baseline": _baseline_dict(result.baseline),
        "perturbations": [_pert_dict(p) for p in result.perturbations],
        "regression": regression_report,
    }


__all__ = ["shape_adapter_result"]
