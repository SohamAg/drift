"""Model-layer and control-flow chaos — faults written onto a baseline checkpoint.

State chaos changes what the graph is given; tool chaos changes what a tool
returns. Two failure sources sit between them and neither can be reached by
editing the input:

* **The model emits a bad tool call** (`model` layer). The agent node's own
  output is wrong: arguments that fail the tool's schema, a tool name that
  does not exist, arguments that fit the schema but carry the wrong value
  (the hardest to notice: the tool succeeds), the same call twice, or no
  call at all where the baseline made one. drift runs the baseline on a checkpointer, finds the checkpoint
  where the AI message carrying a call to the target tool is pending (the
  ToolNode is about to run), rewrites that message in place (same id, so
  `add_messages` replaces it), writes it back *as the node that produced it*
  and resumes. The ToolNode then executes the corrupted call for real.

* **A node silently does not run** (`flow` layer, `skip_node[N]`). At the
  checkpoint whose next task is N, drift writes an empty update *as N*:
  LangGraph records N as having run with no output and continues along N's
  outgoing edges. Downstream nodes see whatever state N would have changed,
  unchanged. ToolNodes are never skipped: their caller's tool call would be
  left with no result, a message sequence every chat API rejects, so the
  "failure" would be the harness's, not the graph's (probed: LangGraph's
  react agent raises on exactly that).

Both need the baseline's checkpoints, so they run like mid-run timing and
need a compiled LangGraph graph (drift attaches an in-memory saver when the
graph has none). When the baseline never reaches the point a fault targets
— the tool was never called, the node never ran — the perturbation is
reported with `fault_fired=False` and a note, not silently dropped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from drift.chaos.fuzzer import ChaosSpec
from drift.chaos.patterns import AutoChaosEvent
from drift.world import World

FAULT_LAYER_MODEL = "model"
FAULT_LAYER_FLOW = "flow"

MODEL_FAULTS: dict[str, dict[str, str]] = {
    "model_bad_tool_args": {
        "fault_class": "crash",
        "models": "the model called the tool with arguments that do not match its schema (all arguments dropped)",
    },
    "model_wrong_tool_args": {
        "fault_class": "value",
        "models": "the model called the right tool with arguments that fit the schema but carry the wrong value",
    },
    "model_unknown_tool": {
        "fault_class": "crash",
        "models": "the model called a tool that does not exist (hallucinated name)",
    },
    "model_duplicate_tool_call": {
        "fault_class": "duplication",
        "models": "the model emitted the same tool call twice in one turn",
    },
    "model_dropped_tool_call": {
        "fault_class": "loss",
        "models": "the model answered with nothing instead of calling the tool",
    },
}
FLOW_FAULTS: dict[str, dict[str, str]] = {
    "skip_node": {
        "fault_class": "loss",
        "models": "the node silently did not run (no output, execution continued along its edges)",
    },
}


@dataclass
class CheckpointEdit:
    """Where and how a checkpoint fault lands. `values` is the update_state
    payload (None = record the node as having run with no output)."""
    rank: int
    snap: Any
    values: dict | None
    as_node: str
    detail: str


class CheckpointChaosEvent(AutoChaosEvent):
    """One model-layer or control-flow fault. `apply(world)` is a recorded
    no-op; the fault lands through `locate(...)` on the baseline's
    checkpoints, which drift_test turns into an update_state + resume."""

    pattern = "checkpoint_fault"
    fault_class = "crash"

    def __init__(self, target: str, *, fault: str, seed: int | None = None) -> None:
        if fault in MODEL_FAULTS:
            self.layer = FAULT_LAYER_MODEL
            info = MODEL_FAULTS[fault]
        elif fault in FLOW_FAULTS:
            self.layer = FAULT_LAYER_FLOW
            info = FLOW_FAULTS[fault]
        else:
            raise ValueError(f"unknown checkpoint fault {fault!r}; one of "
                             f"{', '.join([*MODEL_FAULTS, *FLOW_FAULTS])}")
        self.pattern = fault
        self.fault_class = info["fault_class"]
        super().__init__(target, seed=seed)
        self.target = target
        self.fault = fault
        self.models = info["models"]

    def _mutate(self, world: World) -> str | None:
        what = f"tool {self.target!r}" if self.layer == FAULT_LAYER_MODEL else f"node {self.target!r}"
        return f"{self.layer}-layer fault on {what}: {self.models} (applied to a baseline checkpoint)"

    # -- locating the checkpoint ------------------------------------------------

    def locate(self, root_snaps: dict[int, Any], root_nodes: list[str]) -> CheckpointEdit | str:
        """The edit to write, or a string saying why the baseline offers no
        place for this fault. `root_snaps` = {root step: StateSnapshot} of
        the baseline thread; `root_nodes[i]` = the node that ran root step i+1."""
        if self.layer == FAULT_LAYER_FLOW:
            return self._locate_skip(root_snaps)
        return self._locate_model(root_snaps, root_nodes)

    def _locate_skip(self, root_snaps: dict[int, Any]) -> CheckpointEdit | str:
        for rank in sorted(root_snaps):
            snap = root_snaps[rank]
            nxt = tuple(getattr(snap, "next", ()) or ())
            if nxt == (self.target,):
                return CheckpointEdit(rank=rank, snap=snap, values=None, as_node=self.target,
                                      detail=f"skipped {self.target!r} where the baseline ran it next "
                                             f"(after root step {rank})")
            if self.target in nxt:
                return (f"{self.target!r} runs in parallel with {', '.join(n for n in nxt if n != self.target)} "
                        "there; skipping one branch of a parallel step is not supported")
        return f"the baseline never had {self.target!r} as its next node, so there was nothing to skip"

    def _locate_model(self, root_snaps: dict[int, Any], root_nodes: list[str]) -> CheckpointEdit | str:
        from drift.chaos import messages as _m

        for rank in sorted(root_snaps):
            snap = root_snaps[rank]
            values = dict(getattr(snap, "values", None) or {})
            for key, msgs in values.items():
                if not (_m.is_messages_list(msgs) and msgs):
                    continue
                last = msgs[-1]
                calls = _m.message_tool_calls(last)
                if _m.message_role(last) != "ai" or not any(c.get("name") == self.target for c in calls):
                    continue
                as_node = root_nodes[rank - 1] if 0 < rank <= len(root_nodes) else ""
                corrupted = self._corrupt(last, calls, decoys=_decoy_words(msgs))
                if corrupted is None:
                    return (f"the call to {self.target!r} had no argument value drift could change "
                            "while keeping it schema-valid")
                new_msg, detail = corrupted
                return CheckpointEdit(rank=rank, snap=snap, values={key: [new_msg]}, as_node=as_node,
                                      detail=f"{detail} (the call the baseline's {as_node or 'model'} node "
                                             f"made at root step {rank})")
        return (f"the baseline never had a pending call to {self.target!r}, so there was no model output "
                "to corrupt")

    def _corrupt(self, msg: Any, calls: list[dict], decoys: tuple[str, ...] = ()) -> tuple[Any, str] | None:
        idx = next(i for i, c in enumerate(calls) if c.get("name") == self.target)
        call = dict(calls[idx])
        new_calls = [dict(c) for c in calls]
        f = self.fault
        if f == "model_bad_tool_args":
            detail = f"dropped all arguments of the call to {self.target!r} (was {_preview(call.get('args'))})"
            new_calls[idx] = {**call, "args": {}}
        elif f == "model_wrong_tool_args":
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            changed = wrong_value_args(args, decoys)
            if changed is None:
                return None
            new_args, key, old, new = changed
            detail = (f"changed argument {key!r} of the call to {self.target!r} from {_preview(old, 40)} "
                      f"to {_preview(new, 40)} (same type, still schema-valid)")
            new_calls[idx] = {**call, "args": new_args}
        elif f == "model_unknown_tool":
            bogus = f"{self.target}_v2"
            detail = f"renamed the call to {self.target!r} to the non-existent tool {bogus!r}"
            new_calls[idx] = {**call, "name": bogus}
        elif f == "model_duplicate_tool_call":
            dup_id = f"{call.get('id') or 'call'}_drift_dup"
            detail = f"duplicated the call to {self.target!r} with args {_preview(call.get('args'))}"
            new_calls.insert(idx + 1, {**call, "id": dup_id})
        else:  # model_dropped_tool_call
            detail = f"removed the call to {self.target!r} and left the message empty"
            new_calls = []
        return _with_tool_calls(msg, new_calls, drop_text=(f == "model_dropped_tool_call")), detail


def _wrong_value(v: Any, decoys: tuple[str, ...] = ()) -> Any:
    """A value of the same type as `v` that is plausibly, not absurdly, wrong.
    `decoys` are other words from the conversation: a one-word string is
    swapped for one (the model confused two entities). Returns `v` itself
    when there is no such change."""
    import re

    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1  # off by one
    if isinstance(v, float):
        return v * 10 if v else 1.0  # a unit / magnitude slip
    if isinstance(v, str):
        if re.search(r"\d", v):  # an id, a date, an amount: change the last number
            m = list(re.finditer(r"\d+", v))[-1]
            return v[:m.start()] + str(int(m.group()) + 1).zfill(len(m.group())) + v[m.end():]
        words = v.split()
        if len(words) > 1:  # a query or a name: lose its last word
            return " ".join(words[:-1])
        other = next((d for d in decoys if d.lower() != v.lower()), None)
        return other if (v and other) else v
    if isinstance(v, list) and v:
        return v[:-1] if len(v) > 1 else [_wrong_value(v[0], decoys)]
    if isinstance(v, dict):
        changed = wrong_value_args(v, decoys)
        return changed[0] if changed else v
    return v


def wrong_value_args(args: dict, decoys: tuple[str, ...] = ()) -> tuple[dict, str, Any, Any] | None:
    """Change ONE argument (the first that can be changed) to a wrong value of
    the same type: off-by-one numbers, a bumped id, a truncated query, a
    flipped flag. One wrong field, like a real model mistake, rather than
    everything at once. None when no argument can be changed."""
    for key, old in args.items():
        new = _wrong_value(old, decoys)
        if new != old:
            return {**args, key: new}, key, old, new
    return None


def _decoy_words(msgs: list) -> tuple[str, ...]:
    """Capitalised words from the conversation's human messages, newest first:
    the entities a model could plausibly mix up. A sentence's first word is
    skipped (capitalised by grammar, not because it names something)."""
    import re

    from drift.chaos import messages as _m

    seen: list[str] = []
    for m in reversed(msgs):
        if _m.message_role(m) != "human":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        for sentence in re.split(r"[.!?\n]+", str(content)):
            words = re.findall(r"\b[A-Za-z][a-zA-Z]{2,}\b", sentence)
            for w in words[1:]:
                if w[0].isupper() and w not in seen:
                    seen.append(w)
    return tuple(seen)


def _with_tool_calls(msg: Any, calls: list[dict], *, drop_text: bool = False) -> Any:
    """A copy of AI message `msg` (same id) carrying `calls`. Provider-raw
    `additional_kwargs["tool_calls"]` is dropped so no serializer resurrects
    the original call."""
    if isinstance(msg, dict):
        out = dict(msg)
        out["tool_calls"] = calls
        if drop_text:
            out["content"] = ""
        out.pop("additional_kwargs", None)
        return out
    extra = {k: v for k, v in (getattr(msg, "additional_kwargs", None) or {}).items()
             if k not in ("tool_calls", "function_call")}
    update: dict[str, Any] = {"tool_calls": calls, "additional_kwargs": extra}
    if hasattr(msg, "invalid_tool_calls"):
        update["invalid_tool_calls"] = []
    if drop_text:
        update["content"] = ""
    return msg.model_copy(update=update)


def _preview(value: Any, limit: int = 60) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    text = " ".join(str(text).split())
    return text[:limit] + ("…" if len(text) > limit else "")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def skippable_nodes(graph: Any) -> list[str]:
    """Root nodes of a compiled graph that skip_node may target: every node
    except __start__/__end__ and ToolNodes (see the module docstring)."""
    from drift.chaos.tools import find_tool_nodes

    pregel = getattr(graph, "bound", None) if not hasattr(graph, "nodes") else graph
    nodes = getattr(pregel, "nodes", None)
    if not isinstance(nodes, dict):
        return []
    tool_nodes = {path for path, _ in find_tool_nodes(graph) if "/" not in path}
    return [str(n) for n in nodes if n not in ("__start__", "__end__") and n not in tool_nodes]


def discover_model_patterns(graph: Any, *, seed: int = 0, faults: list[str] | None = None) -> list[ChaosSpec]:
    """One spec per (tool reachable through a ToolNode × model fault),
    named like `model_bad_tool_args[web_search]`."""
    from drift.chaos.tools import tool_names

    chosen = list(faults or MODEL_FAULTS)
    specs: list[ChaosSpec] = []
    for name in tool_names(graph):
        for fault in chosen:
            if fault not in MODEL_FAULTS:
                raise ValueError(f"unknown model fault {fault!r}; one of {', '.join(MODEL_FAULTS)}")
            specs.append(ChaosSpec(
                pattern_name=f"{fault}[{name}]", field=name, pattern_type=fault,
                factory=lambda n=name, f=fault, s=seed: CheckpointChaosEvent(n, fault=f, seed=s),
                fault_class=MODEL_FAULTS[fault]["fault_class"],
            ))
    return specs


def discover_flow_patterns(graph: Any, *, seed: int = 0) -> list[ChaosSpec]:
    """One `skip_node[N]` spec per skippable root node."""
    return [
        ChaosSpec(
            pattern_name=f"skip_node[{n}]", field=n, pattern_type="skip_node",
            factory=lambda n=n, s=seed: CheckpointChaosEvent(n, fault="skip_node", seed=s),
            fault_class=FLOW_FAULTS["skip_node"]["fault_class"],
        )
        for n in skippable_nodes(graph)
    ]


def is_checkpoint_event(event: Any) -> bool:
    return isinstance(event, CheckpointChaosEvent)


__all__ = [
    "FAULT_LAYER_FLOW", "FAULT_LAYER_MODEL", "FLOW_FAULTS", "MODEL_FAULTS", "CheckpointChaosEvent",
    "CheckpointEdit", "discover_flow_patterns", "discover_model_patterns", "is_checkpoint_event",
    "skippable_nodes", "wrong_value_args",
]
