# ADR-0002: Tool faults are injected through ToolNode's own interceptor, not by patching tools

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-11-11 15:19 IST — `7f2757a` |
| Recorded | 2026-11-19 (after the fact) |
| Owner | Soham Agarwal |

## Context
Before this, every chaos pattern changed the graph's *initial state* and
reran it. A deployed agent fails differently. The search API times out on
the second call, a page comes back truncated, a fetched document carries
an instruction. Testing that means changing a tool's result *while the
graph runs*.

## Decision
drift walks the compiled graph, subgraphs included, and finds every
LangGraph `ToolNode`. For the length of one run it installs a wrapper
through the node's own `wrap_tool_call` / `awrap_tool_call` hook, then
restores whatever was there before. A fault targets one tool and one
occurrence of it (`on_call`). Only tools reachable through a ToolNode are
offered as targets.

## Alternatives considered
- **Monkeypatch the user's tool functions.** Rejected. It changes objects
  the user owns, leaks between concurrent runs (the web server runs many),
  and misses tools that are rebuilt per run.
- **Proxy tools at the network layer.** Rejected. Many tools never touch
  the network, and it needs infrastructure a `pip install` cannot provide.
- **Offer every tool name found anywhere.** Rejected. A fault on a tool
  called from a hand-written node could never fire, and it would be
  reported as a fault the graph survived.

## Consequences
- No user object is modified. Nested agents (a supervisor's
  `supervisor/tools`) are reachable, which ADR-0006 depends on.
- Tools called outside a ToolNode cannot be faulted. This is documented
  in `docs/tool_chaos.md`, and such tools are not offered.
- If a scheduled fault never fires, the run says so (`fault_fired=False`).
  ADR-0007 defines how such runs are counted.
