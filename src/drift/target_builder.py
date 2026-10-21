"""Generate a runnable LangGraph multi-agent system from a topology choice
and a list of agents — the "Build" tab's backend.

This is scaffolding, not a finished system. Pick a topology, name a few
agents, give each a short role description, and get back real Python
source: a `build()` / `initial_state()` module in the same shape every
other target in this project already uses (see examples/targets/), so
the generated file is immediately runnable through the Import-graph tab
or the Python API — no new execution path, no new security surface
beyond what Import-graph already documents (arbitrary local code
execution by design, Jupyter-kernel trust model).

Every agent node is a real LLM call (ChatOpenAI). Routing between agents
is FIXED, not LLM-conditional, for every topology except `loop` (which
needs a real stop condition) — this keeps generated graphs simple and
reliably terminating. The generated file's docstring says so explicitly
and points at what to hand-edit if real conditional routing is wanted;
"tweak specific aspects" means editing the generated .py file, not a
second generation pass.

User-supplied text (agent names, role descriptions) is embedded into the
generated source via `repr()`, never raw interpolation — the generated
file will later be imported and executed, so this is about producing
syntactically correct Python from arbitrary input, not sandboxing (the
existing arbitrary-code-execution trust model already covers that).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TOPOLOGY_MIN_AGENTS: dict[str, int] = {
    "supervisor": 2,
    "swarm": 2,
    "pipeline": 2,
    "loop": 2,
    # dispatcher + >=2 workers (for genuine parallelism) + merge
    "fanout": 4,
}
TOPOLOGY_MAX_AGENTS: dict[str, int] = {
    "loop": 2,  # exactly 2: generator, critic
}
TOPOLOGY_DESCRIPTIONS: dict[str, str] = {
    "supervisor": "a coordinator delegates to specialists in sequence, then synthesizes",
    "swarm": "agents hand off directly to each other, no central router",
    "pipeline": "fixed sequential stages, each building on the last",
    "loop": "two agents (generator, critic) cycle until the critic approves",
    "fanout": "one dispatcher, several workers in parallel, then a merge",
}
#: Ordering convention per topology, for callers deciding what to label
#: each agent-input slot as (the UI's Build tab uses these verbatim).
TOPOLOGY_AGENT_ORDER_HINTS: dict[str, str] = {
    "supervisor": "First agent is the coordinator; the rest are specialists it delegates to in order.",
    "swarm": "Agents hand off in the order listed, peer to peer.",
    "pipeline": "Agents run as sequential stages in the order listed.",
    "loop": "Exactly 2 agents: first is the generator, second is the critic.",
    "fanout": "First agent is the dispatcher, last is the merge step; everyone in between runs in parallel.",
}

_VALID_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_VALID_TARGET_NAME = re.compile(r"^[a-z][a-z0-9_]{1,50}$")

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class AgentSpec:
    name: str   # original, as the user typed it
    slug: str   # validated Python-identifier-safe node name
    role: str   # free-text role/system-prompt description


class TargetBuilderError(ValueError):
    """Raised for invalid topology/agent input — always a 400 at the API layer."""


def slugify_agent_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", "_").replace("-", "_"))
    if not _VALID_SLUG.match(slug):
        raise TargetBuilderError(
            f"agent name {name!r} must produce a valid identifier "
            f"(letters, digits, underscore, starting with a letter)"
        )
    return slug


def validate_target_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "", name.strip().lower().replace(" ", "_").replace("-", "_"))
    if not _VALID_TARGET_NAME.match(slug):
        raise TargetBuilderError(
            f"target name {name!r} must produce a valid identifier, 2-51 chars, "
            f"letters/digits/underscore, starting with a letter"
        )
    return slug


def _build_agent_specs(topology: str, agents: list[dict]) -> list[AgentSpec]:
    if topology not in TOPOLOGY_MIN_AGENTS:
        raise TargetBuilderError(
            f"unknown topology {topology!r}; choose one of {sorted(TOPOLOGY_MIN_AGENTS)}"
        )
    min_n = TOPOLOGY_MIN_AGENTS[topology]
    max_n = TOPOLOGY_MAX_AGENTS.get(topology)
    if len(agents) < min_n:
        raise TargetBuilderError(f"{topology!r} needs at least {min_n} agent(s), got {len(agents)}")
    if max_n is not None and len(agents) != max_n:
        raise TargetBuilderError(f"{topology!r} needs exactly {max_n} agent(s), got {len(agents)}")

    specs: list[AgentSpec] = []
    seen_slugs: set[str] = set()
    for a in agents:
        name = str(a.get("name", "")).strip()
        role = str(a.get("role", "")).strip()
        if not name:
            raise TargetBuilderError("every agent needs a name")
        if not role:
            raise TargetBuilderError(f"agent {name!r} needs a role description")
        slug = slugify_agent_name(name)
        if slug in seen_slugs:
            raise TargetBuilderError(f"duplicate agent name (after slugifying): {slug!r}")
        seen_slugs.add(slug)
        specs.append(AgentSpec(name=name, slug=slug, role=role))

    return specs


# ---------------------------------------------------------------------------
# Static source — no per-generation substitution, so no f-string / .format()
# escaping to get wrong. Agent node bodies read MODEL from a closure arg.
# ---------------------------------------------------------------------------

_NODE_FACTORY_SOURCE = '''
def _agent_node(name: str, role: str, model: str):
    """One LLM-backed agent node. Reads state["notes"] for what prior
    agents wrote, writes its own response under state["notes"][name]."""
    def node(state: dict) -> dict:
        llm = ChatOpenAI(model=model, temperature=0)
        notes = state.get("notes", {})
        context = "\\n\\n".join(f"[{k}]: {v}" for k, v in notes.items()) or "(nothing yet)"
        system = SystemMessage(content=role)
        human = HumanMessage(content=(
            f"Task: {state.get('task', '')}\\n\\n"
            f"What's happened so far:\\n{context}\\n\\n"
            f"Respond as {name}, addressing the task given what's happened so far."
        ))
        response = llm.invoke([system, human])
        new_notes = dict(notes)
        new_notes[name] = response.content
        return {"notes": new_notes}
    return node


def _loop_generator_node(name: str, role: str, model: str):
    """Like _agent_node, but also advances the loop's iteration counter —
    the conditional edge below uses this as a hard stop so a critic that
    never approves can't loop forever."""
    base = _agent_node(name, role, model)
    def node(state: dict) -> dict:
        result = base(state)
        result["_iterations"] = state.get("_iterations", 0) + 1
        return result
    return node
'''

_LOOP_ROUTER_SOURCE = '''
def _route_after_critic(state: dict) -> str:
    verdict = state.get("notes", {}).get(%(critic_slug)r, "")
    if "APPROVED" in verdict.upper() or state.get("_iterations", 0) >= %(max_iterations)d:
        return "end"
    return "retry"
'''


def _safe_docstring_text(s: str) -> str:
    """Make arbitrary user text safe to embed inside a plain (non-raw)
    triple-double-quoted docstring: neutralize backslashes (would start an
    escape sequence) and triple-double-quote runs (would terminate the
    docstring early, breaking the generated file's syntax)."""
    return s.replace("\\", "/").replace('"""', "'''")


def _render_header(topology: str, specs: list[AgentSpec], model: str) -> str:
    agent_list = "\n".join(
        f"  - {s.slug}: {_safe_docstring_text(s.role)}" for s in specs
    )
    return (
        f'"""Generated multi-agent system — {topology} topology.\n'
        f"\n"
        f"Built via drift's Build tab. This is a STARTING POINT, not a\n"
        f"finished system — edit the role descriptions below, or the graph\n"
        f"wiring further down, to shape real behavior. Routing between\n"
        f"agents is FIXED (sequential/parallel by position), not\n"
        f"LLM-conditional, except for the loop topology's generator/critic\n"
        f"check — add langgraph conditional edges yourself if you want an\n"
        f"agent to decide where to go next.\n"
        f"\n"
        f"Every agent calls a real LLM ({model} by default); running this\n"
        f"through drift costs real tokens and needs OPENAI_API_KEY set.\n"
        f"\n"
        f"Topology: {topology} — {TOPOLOGY_DESCRIPTIONS[topology]}\n"
        f"Agents:\n{agent_list}\n"
        f'"""\n'
        # Deliberately NOT "from __future__ import annotations" in the
        # generated file — the State TypedDict's Annotated[...] reducer
        # needs eager evaluation. With deferred (string) annotations,
        # LangGraph's get_type_hints() resolves them via
        # sys.modules[cls.__module__], which breaks for any import path
        # that doesn't register the module there (several legitimate ways
        # to load a .py file don't); eager evaluation sidesteps that
        # fragility entirely. Python 3.10+ (this project's floor) supports
        # `X | None` natively, so nothing generated here needs it anyway.
        f"import operator\n"
        f"from typing import Annotated, TypedDict\n"
        f"\n"
        f"from langchain_core.messages import HumanMessage, SystemMessage\n"
        f"from langchain_openai import ChatOpenAI\n"
        f"from langgraph.graph import END, START, StateGraph\n"
        f"\n"
        f"DEFAULT_MODEL = {model!r}\n"
        f"DEFAULT_QUERY = \"describe your task here\"\n"
        f"\n"
        f"\n"
        f"class State(TypedDict):\n"
        f'    """notes uses a merge reducer (not plain replace) because the\n'
        f"    fanout topology has multiple agents writing to it in the SAME\n"
        f"    super-step — without a reducer, LangGraph can't tell which\n"
        f'    concurrent write should win and raises InvalidUpdateError."""\n'
        f"    task: str\n"
        f"    notes: Annotated[dict, operator.or_]\n"
        f"    _iterations: int\n"
    )


def _node_lines(specs: list[AgentSpec], topology: str) -> list[str]:
    lines = []
    for i, s in enumerate(specs):
        use_factory = "_loop_generator_node" if (topology == "loop" and i == 0) else "_agent_node"
        lines.append(
            f'    g.add_node({s.slug!r}, {use_factory}({s.slug!r}, {s.role!r}, model))'
        )
    return lines


def _wire_chain(slugs: list[str]) -> list[str]:
    lines = [f'    g.add_edge(START, {slugs[0]!r})']
    for a, b in zip(slugs, slugs[1:]):
        lines.append(f'    g.add_edge({a!r}, {b!r})')
    lines.append(f'    g.add_edge({slugs[-1]!r}, END)')
    return lines


def _wire_supervisor(specs: list[AgentSpec]) -> tuple[list[str], list[str]]:
    """Coordinator -> specialists in sequence -> a synthesis pass (reusing
    the coordinator's own role) -> END. Returns (extra_node_lines, edge_lines)."""
    supervisor, specialists = specs[0], specs[1:]
    chain = [supervisor.slug] + [s.slug for s in specialists] + ["synthesis"]
    extra_node = [
        f'    g.add_node("synthesis", _agent_node("synthesis", {supervisor.role!r}, model))'
    ]
    return extra_node, _wire_chain(chain)


def _wire_loop(specs: list[AgentSpec], max_iterations: int) -> list[str]:
    generator, critic = specs[0], specs[1]
    lines = [
        f'    g.add_edge(START, {generator.slug!r})',
        f'    g.add_edge({generator.slug!r}, {critic.slug!r})',
        f'    g.add_conditional_edges(',
        f'        {critic.slug!r}, _route_after_critic, {{"retry": {generator.slug!r}, "end": END}},',
        f'    )',
    ]
    return lines


def _wire_fanout(specs: list[AgentSpec]) -> list[str]:
    # Convention: first spec is the dispatcher, last is the merge node,
    # everything between fans out in parallel.
    dispatcher = specs[0]
    *workers, merge = specs[1:]
    lines = [f'    g.add_edge(START, {dispatcher.slug!r})']
    for w in workers:
        lines.append(f'    g.add_edge({dispatcher.slug!r}, {w.slug!r})')
        lines.append(f'    g.add_edge({w.slug!r}, {merge.slug!r})')
    lines.append(f'    g.add_edge({merge.slug!r}, END)')
    return lines


def generate_target_source(
    topology: str, agents: list[dict], model: str = DEFAULT_MODEL,
) -> str:
    """Render a complete, syntactically valid target module as a string.
    Raises TargetBuilderError on invalid topology/agent input — never
    partially generates."""
    specs = _build_agent_specs(topology, agents)

    parts: list[str] = [_render_header(topology, specs, model)]
    parts.append(_NODE_FACTORY_SOURCE)

    if topology == "loop":
        critic = specs[1]
        parts.append(_LOOP_ROUTER_SOURCE % {
            "critic_slug": critic.slug, "max_iterations": 3,
        })

    build_lines = ["", "", "def build(model: str | None = None) -> object:",
                   "    model = model or DEFAULT_MODEL",
                   "    g = StateGraph(State)"]
    build_lines += _node_lines(specs, topology)

    if topology == "supervisor":
        extra_node, edge_lines = _wire_supervisor(specs)
        build_lines += extra_node
        build_lines += edge_lines
    elif topology in ("swarm", "pipeline"):
        build_lines += _wire_chain([s.slug for s in specs])
    elif topology == "loop":
        build_lines += _wire_loop(specs, max_iterations=3)
    elif topology == "fanout":
        build_lines += _wire_fanout(specs)

    build_lines.append("    return g.compile()")
    parts.append("\n".join(build_lines) + "\n")

    parts.append(
        "\n\n"
        "def initial_state(query: str = DEFAULT_QUERY) -> dict:\n"
        "    return {\"task\": query, \"notes\": {}, \"_iterations\": 0}\n"
    )

    return "".join(parts)
