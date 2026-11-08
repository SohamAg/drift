"""Generate docs/detectors/README.md from the live detector library."""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from drift.failures.library import ALL_DETECTORS  # noqa: E402

NEEDS_CONFIG = {"premature_termination", "policy_inconsistency_across_steps"}
MESSAGE_AWARE = {
    "tool_call_loop", "silent_tool_failure", "injected_instruction_followed",
    "injected_tool_call_followed", "retry_amplification", "novel_tool_call",
    "tainted_side_effect_call",
}
NEEDS_BASELINE = {"retry_amplification", "novel_tool_call"}

lines = [
    "# Coordination-failure detector library",
    "",
    "One page per detector: what fires, what is deliberately not flagged, the",
    "sources, a worked example, and the architecture-level fix. Every detector",
    "runs on every `drift_test` call — deterministic, no LLM, no configuration",
    "unless marked. This index is generated from the library itself",
    "(`python scripts/gen_detector_index.py`); a test fails if it goes stale.",
    "",
    f"**{len(ALL_DETECTORS)} detectors.** Reads: *state* = node-level state writes;",
    "*messages* = the chat messages nodes write (tool calls, tool results, final",
    "answers); *baseline* = also needs the unperturbed run's trace (the adapter",
    "supplies it for every perturbation).",
    "",
    "| Detector | Reads | MAST | One line |",
    "|---|---|---|---|",
]
for mod in ALL_DETECTORS:
    reads = "messages" if mod.NAME in MESSAGE_AWARE else "state"
    if mod.NAME in NEEDS_BASELINE:
        reads += " + baseline"
    if mod.NAME in NEEDS_CONFIG:
        reads += " · needs config"
    mast = ", ".join(mod.MAST_MODES) if mod.MAST_MODES else "—"
    summary = " ".join(mod.SUMMARY.split())
    lines.append(f"| [`{mod.NAME}`]({mod.NAME}.md) | {reads} | {mast} | {summary} |")

lines += [
    "",
    "## Security-relevant subset",
    "",
    "Four detectors exist to answer *does this graph let data drive actions?*",
    "and are rendered with the critical emphasis in the UI:",
    "",
    "- `injected_instruction_followed` — a planted instruction's canary phrase",
    "  reappears in an AI message (prompt layer, canary-verified).",
    "- `injected_tool_call_followed` — the canary reappears in a tool call's",
    "  arguments (action layer, canary-verified — AgentDojo's success criterion).",
    "- `novel_tool_call` — a tool the baseline never called, with side-effect",
    "  tools (send / write / delete / pay / execute) flagged first.",
    "- `tainted_side_effect_call` — a side-effect tool called with a value lifted",
    "  verbatim from an earlier tool result that the user never wrote; no canary.",
    "",
    "The first two are the second half of the `inject_tool_result_instruction`",
    "and `inject_tool_result_tool_hijack` chaos patterns. The last two work on",
    "any run, baseline included.",
    "",
]
out = Path("docs/detectors/README.md")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out} with {len(ALL_DETECTORS)} detectors")
