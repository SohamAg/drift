"""Curate saved runs (results/, git-ignored) into tests/fixtures/ungrounded_corpus.json.

    python scripts/build_ungrounded_corpus.py

Positives: every run ungrounded_number fires on. Each was read by hand and
is a real fabrication (ADR-0011). Negatives: runs that have tool results
and whose final answers contain specific numbers the guards clear (speed
of light, computed values, grounded figures). Re-run after adding saved
runs, re-read any new positive, then commit the fixture.
"""
import glob, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from drift.failures.library import ungrounded_number as U
from drift.failures.library.base import from_adapter_trace, conversation

items = []


from drift.chaos import messages as _m


def _placeholder_state(state):
    # conversation() reads only how many messages state_after holds (to tell
    # a node's new messages from a subgraph re-emit), so keep the count and
    # the roles, not the text: the full history at every step is quadratic.
    if not isinstance(state, dict):
        return state
    return {k: ([{"type": _m.message_role(x) or "ai", "content": ""} for x in v]
                if _m.is_messages_list(v) else None) for k, v in state.items() if _m.is_messages_list(v)}


def slim(trace):
    out = []
    for s in trace:
        r = {k: s.get(k) for k in ("step", "node", "update") if k in s}
        if "state_after" in s:
            r["state_after"] = _placeholder_state(s["state_after"])
        out.append(r)
    return out


def interesting_negative(msgs):
    fin = next((m for m in reversed(msgs) if m.role == "ai" and m.text.strip() and not m.tool_calls), None)
    return fin is not None and any(sig >= 4 for _v, sig, _w in U._parse(fin.text))


for f in sorted(glob.glob("results/**/*.json", recursive=True)):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if "tradingagents" in f:  # 650 KB of third-party market and news text
        continue
    if not isinstance(d, dict) or "perturbations" not in d:
        continue
    b = d.get("baseline") or {}
    runs = [("baseline", b.get("trace"), b.get("initial_state") or b.get("perturbed_initial_state"))]
    for p in d["perturbations"]:
        if p.get("fault_fired") is False and p.get("fault_layer") in ("model", "flow"):
            continue
        runs.append((p.get("event_name") or p.get("name") or "perturbation", p.get("trace"), p.get("perturbed_initial_state")))
    for label, trace, init in runs:
        if not trace:
            continue
        init = init if isinstance(init, dict) else None
        ctx = from_adapter_trace(trace, initial_state=init)
        msgs = conversation(ctx)
        if not any(m.role == "tool" for m in msgs):
            continue
        fired = bool(U.detect(ctx))
        if fired or interesting_negative(msgs):
            items.append({
                "source": f.replace(os.sep, "/"),
                "run": label,
                "expect_fire": fired,
                "initial_state": init,
                "trace": slim(trace),
            })

blob = json.dumps({"_about": __doc__.strip(), "runs": items}, ensure_ascii=False, separators=(",", ":"), default=str)
# Refuse to write anything that looks like a key, a local path or an email.
for pat in (r"sk-[A-Za-z0-9_-]{10,}", r"[A-Za-z]:\\\\Users", r"/home/\w+", r"/Users/\w+", r"[\w.+-]+@[\w-]+\.\w+"):
    assert not re.search(pat, blob, re.I), pat
out = "tests/fixtures/ungrounded_corpus.json"
os.makedirs("tests/fixtures", exist_ok=True)
open(out, "w", encoding="utf-8", newline="\n").write(blob)
print(len(items), "runs,", sum(i["expect_fire"] for i in items), "positive,", len(blob) // 1024, "KB")
for i in items:
    print(i["expect_fire"], i["source"], i["run"])
