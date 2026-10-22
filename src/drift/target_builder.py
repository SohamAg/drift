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

import json
import re
from dataclasses import dataclass, field

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

_VERDICT_NODE_FACTORY_SOURCE = '''

def _verdict_node(name: str, role: str, model: str, schema: dict):
    """Like _agent_node, but requests STRUCTURED JSON matching `schema`
    instead of free text, and writes it to state["verdict"] (not just
    state["notes"][name]) — meant to be the graph's final node,
    synthesizing everything the team found into one machine-readable
    verdict. On a malformed response, degrades to an "error" verdict
    rather than raising — a bad LLM response shouldn't crash the graph,
    it should be visible in the output."""
    def node(state: dict) -> dict:
        llm = ChatOpenAI(model=model, temperature=0)
        notes = state.get("notes", {})
        context = "\\n\\n".join(f"[{k}]: {v}" for k, v in notes.items()) or "(nothing yet)"
        schema_lines = "\\n".join(f'  "{k}": <{v}>' for k, v in schema.items())
        system = SystemMessage(content=(
            f"{role}\\n\\n"
            f"Respond with STRICT JSON, no markdown fencing, exactly this shape:\\n"
            f"{{\\n{schema_lines}\\n}}"
        ))
        human = HumanMessage(content=(
            f"Task: {state.get('task', '')}\\n\\n"
            f"What the team found:\\n{context}"
        ))
        response = llm.invoke([system, human], response_format={"type": "json_object"})
        try:
            verdict = json.loads(response.content)
            if not isinstance(verdict, dict):
                raise ValueError("verdict JSON was not an object")
        except (json.JSONDecodeError, ValueError) as e:
            verdict = {"error": f"model did not return a usable verdict: {e}", "raw": str(response.content)}
        new_notes = dict(notes)
        new_notes[name] = str(verdict)
        return {"notes": new_notes, "verdict": verdict}
    return node
'''


def _safe_docstring_text(s: str) -> str:
    """Make arbitrary user text safe to embed inside a plain (non-raw)
    triple-double-quoted docstring: neutralize backslashes (would start an
    escape sequence) and triple-double-quote runs (would terminate the
    docstring early, breaking the generated file's syntax)."""
    return s.replace("\\", "/").replace('"""', "'''")


def _render_header(
    topology: str,
    specs: list[AgentSpec],
    model: str,
    verdict_schema: dict[str, str] | None = None,
) -> str:
    agent_list = "\n".join(
        f"  - {s.slug}: {_safe_docstring_text(s.role)}" for s in specs
    )
    verdict_note = (
        "\n\nThe final node produces a STRUCTURED verdict (see VERDICT_SCHEMA\n"
        "below) instead of free text, written to state[\"verdict\"] — built\n"
        "for calibration against real historical outcomes\n"
        "(see drift.calibration), not just human reading."
        if verdict_schema is not None else ""
    )
    lines = [
        f'"""Generated multi-agent system — {topology} topology.\n',
        "\n",
        "Built via drift's Build tab. This is a STARTING POINT, not a\n",
        "finished system — edit the role descriptions below, or the graph\n",
        "wiring further down, to shape real behavior. Routing between\n",
        "agents is FIXED (sequential/parallel by position), not\n",
        "LLM-conditional, except for the loop topology's generator/critic\n",
        "check — add langgraph conditional edges yourself if you want an\n",
        "agent to decide where to go next.\n",
        "\n",
        f"Every agent calls a real LLM ({model} by default); running this\n",
        "through drift costs real tokens and needs OPENAI_API_KEY set.\n",
        "\n",
        f"Topology: {topology} — {TOPOLOGY_DESCRIPTIONS[topology]}\n",
        f"Agents:\n{agent_list}\n",
        f"{verdict_note}\n" if verdict_note else "",
        '"""\n',
        # Deliberately NOT "from __future__ import annotations" in the
        # generated file — the State TypedDict's Annotated[...] reducer
        # needs eager evaluation. With deferred (string) annotations,
        # LangGraph's get_type_hints() resolves them via
        # sys.modules[cls.__module__], which breaks for any import path
        # that doesn't register the module there (several legitimate ways
        # to load a .py file don't); eager evaluation sidesteps that
        # fragility entirely. Python 3.10+ (this project's floor) supports
        # `X | None` natively, so nothing generated here needs it anyway.
        "import json\n" if verdict_schema is not None else "",
        "import operator\n",
        "from typing import Annotated, TypedDict\n",
        "\n",
        "from langchain_core.messages import HumanMessage, SystemMessage\n",
        "from langchain_openai import ChatOpenAI\n",
        "from langgraph.graph import END, START, StateGraph\n",
        "\n",
        f"DEFAULT_MODEL = {model!r}\n",
        'DEFAULT_QUERY = "describe your task here"\n',
    ]
    if verdict_schema is not None:
        schema_body = "".join(f"    {k!r}: {v!r},\n" for k, v in verdict_schema.items())
        lines.append(f"VERDICT_SCHEMA = {{\n{schema_body}}}\n")
    lines += [
        "\n",
        "\n",
        "class State(TypedDict):\n",
        '    """notes uses a merge reducer (not plain replace) because the\n',
        "    fanout topology has multiple agents writing to it in the SAME\n",
        "    super-step — without a reducer, LangGraph can't tell which\n",
        '    concurrent write should win and raises InvalidUpdateError."""\n',
        "    task: str\n",
        "    notes: Annotated[dict, operator.or_]\n",
        "    _iterations: int\n",
    ]
    if verdict_schema is not None:
        lines.append("    verdict: dict\n")
    return "".join(lines)


def _node_lines(
    specs: list[AgentSpec], topology: str, verdict_schema: dict[str, str] | None = None,
) -> list[str]:
    # "supervisor" puts its verdict on the separately-added "synthesis"
    # node (see _wire_supervisor), not on any of `specs`. "loop" never
    # gets a verdict node at all — its critic's free-text output already
    # drives _route_after_critic's APPROVED/REVISE check; swapping it for
    # structured JSON would break that routing (see generate_target_source's
    # validation, which rejects loop + verdict_schema before getting here).
    verdict_index = (
        len(specs) - 1 if (verdict_schema is not None and topology not in ("supervisor", "loop"))
        else None
    )
    lines = []
    for i, s in enumerate(specs):
        if topology == "loop" and i == 0:
            use_factory = "_loop_generator_node"
        elif i == verdict_index:
            use_factory = "_verdict_node"
        else:
            use_factory = "_agent_node"
        extra_arg = ", VERDICT_SCHEMA" if use_factory == "_verdict_node" else ""
        lines.append(
            f'    g.add_node({s.slug!r}, {use_factory}({s.slug!r}, {s.role!r}, model{extra_arg}))'
        )
    return lines


def _wire_chain(slugs: list[str]) -> list[str]:
    lines = [f'    g.add_edge(START, {slugs[0]!r})']
    for a, b in zip(slugs, slugs[1:]):
        lines.append(f'    g.add_edge({a!r}, {b!r})')
    lines.append(f'    g.add_edge({slugs[-1]!r}, END)')
    return lines


def _wire_supervisor(
    specs: list[AgentSpec], verdict_schema: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Coordinator -> specialists in sequence -> a synthesis pass (reusing
    the coordinator's own role) -> END. Returns (extra_node_lines, edge_lines).
    When verdict_schema is given, the synthesis node is the verdict node —
    it's already the graph's final, synthesizing step."""
    supervisor, specialists = specs[0], specs[1:]
    chain = [supervisor.slug] + [s.slug for s in specialists] + ["synthesis"]
    if verdict_schema is not None:
        extra_node = [
            '    g.add_node("synthesis", _verdict_node('
            f'"synthesis", {supervisor.role!r}, model, VERDICT_SCHEMA))'
        ]
    else:
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
    topology: str,
    agents: list[dict],
    model: str = DEFAULT_MODEL,
    verdict_schema: dict[str, str] | None = None,
) -> str:
    """Render a complete, syntactically valid target module as a string.
    Raises TargetBuilderError on invalid topology/agent input — never
    partially generates.

    verdict_schema (optional): when given, the graph's final node
    produces STRUCTURED JSON matching this {field: description} shape
    instead of free text, written to state["verdict"] — this is what
    makes a generated system usable for calibration against real
    historical outcomes (see drift.calibration and
    drift.historical_grounding.STOCK_VERDICT_SCHEMA for the stock-
    reaction-prediction shape those use).

    Not supported for topology="loop": the critic's free-text output
    already drives the generator/critic routing decision
    (APPROVED/REVISE) — swapping it for structured JSON would break
    that, not just add a feature on top of it.
    """
    if verdict_schema is not None and topology == "loop":
        raise TargetBuilderError(
            "verdict_schema isn't supported for topology='loop' — the "
            "critic's free-text output already drives the generator/critic "
            "routing decision; use supervisor or fanout for a structured "
            "verdict instead"
        )

    specs = _build_agent_specs(topology, agents)

    parts: list[str] = [_render_header(topology, specs, model, verdict_schema)]
    parts.append(_NODE_FACTORY_SOURCE)
    if verdict_schema is not None:
        parts.append(_VERDICT_NODE_FACTORY_SOURCE)

    if topology == "loop":
        critic = specs[1]
        parts.append(_LOOP_ROUTER_SOURCE % {
            "critic_slug": critic.slug, "max_iterations": 3,
        })

    build_lines = ["", "", "def build(model: str | None = None) -> object:",
                   "    model = model or DEFAULT_MODEL",
                   "    g = StateGraph(State)"]
    build_lines += _node_lines(specs, topology, verdict_schema)

    if topology == "supervisor":
        extra_node, edge_lines = _wire_supervisor(specs, verdict_schema)
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

    initial_state_dict = '{"task": query, "notes": {}, "_iterations": 0' + (
        ', "verdict": {}}' if verdict_schema is not None else '}'
    )
    parts.append(
        "\n\n"
        "def initial_state(query: str = DEFAULT_QUERY) -> dict:\n"
        f"    return {initial_state_dict}\n"
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# AI-assisted proposal — turn a free-text scenario into a starting roster
# for the Build tab. generate_target_source() above does all the real
# codegen and all the real validation; this only drafts its input. The
# user reviews/edits the draft in the same agent-row UI the manual flow
# already has before ever calling Generate, so a proposal that's slightly
# off (wrong agent count for the topology, awkward names) is never fatal —
# it's a starting point, not a contract.
# ---------------------------------------------------------------------------


class ScenarioProposalError(ValueError):
    """Raised when a scenario can't be turned into a usable draft — empty
    input, an LLM call failure, or a response that doesn't parse into the
    expected shape. Always surfaced as an error, never silently swallowed
    into an empty-but-valid-looking proposal."""


@dataclass
class ProposedTarget:
    """An LLM's DRAFT starting point for the Build tab. Deliberately not
    validated against TOPOLOGY_MIN_AGENTS/TOPOLOGY_MAX_AGENTS here — that
    validation already lives in generate_target_source() and runs when
    the user actually clicks Generate, after they've had a chance to
    review and edit this draft in the UI."""
    topology: str
    agents: list[dict[str, str]] = field(default_factory=list)  # [{"name", "role"}, ...]
    suggested_name: str = ""
    suggested_query: str = ""
    rationale: str = ""

    def to_json(self) -> dict:
        return {
            "topology": self.topology,
            "agents": self.agents,
            "suggested_name": self.suggested_name,
            "suggested_query": self.suggested_query,
            "rationale": self.rationale,
        }


def _proposal_system_prompt() -> str:
    topo_lines = "\n".join(
        f"- {name}: {TOPOLOGY_DESCRIPTIONS[name]}. {TOPOLOGY_AGENT_ORDER_HINTS[name]} "
        f"(needs {TOPOLOGY_MIN_AGENTS[name]}"
        + (f"-{TOPOLOGY_MAX_AGENTS[name]}" if name in TOPOLOGY_MAX_AGENTS else "+")
        + " agents)"
        for name in TOPOLOGY_MIN_AGENTS
    )
    return (
        "You design multi-agent system ROSTERS for a chaos-testing tool. "
        "Given a one-line scenario, propose which coordination topology "
        "fits best and a short roster of agents to simulate it — this is "
        "NOT a final answer to the scenario, it's a cast of agents whose "
        "work/debate would explore it.\n\n"
        "Available topologies:\n"
        f"{topo_lines}\n\n"
        "Pick the topology whose SHAPE genuinely matches how this scenario "
        "should be explored — e.g. a scenario that's naturally a back-and-"
        "forth between a couple of peers suits swarm or loop; one with a "
        "clear central decision-maker delegating to specialists suits "
        "supervisor; one needing several independent angles explored at "
        "once before converging suits fanout.\n\n"
        "The agent order matters and must literally match the role you "
        "give each one — for fanout, agents[0]'s role must BE the act of "
        "splitting the task into angles (not one of the angles itself), "
        "and the LAST agent's role must BE synthesizing everyone else's "
        "output; for loop, agents[0] generates/revises and agents[1] "
        "reviews and renders a verdict.\n\n"
        "Respond with STRICT JSON, no markdown fencing, exactly this shape:\n"
        '{"topology": "<one of the names above>", '
        '"rationale": "<one sentence: why this topology fits>", '
        '"suggested_name": "<short snake_case name>", '
        '"suggested_query": "<a concrete example input for this system>", '
        '"agents": [{"name": "<snake_case>", "role": "<1-2 sentence role '
        'description, written as instructions TO that agent>"}, ...]}'
    )


async def _call_proposal_llm(system: str, user: str, model: str) -> str:
    """Deliberately separate from drift.failures.judge.OpenAIJudge: that
    class degrades a transient failure to '{"failures": []}', which is
    the right call for coordination judging (an empty-findings default is
    always safe) but wrong here — a silently empty proposal would just
    look like an unexplained parse failure downstream. Errors propagate
    instead, so the caller can surface a real message."""
    import os

    from openai import AsyncOpenAI  # lazy import — openai is optional

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
            max_tokens=1200,
        )
    except Exception as e:
        raise ScenarioProposalError(
            f"LLM call failed ({type(e).__name__}): {e}. "
            "Check OPENAI_API_KEY, billing, and model name."
        ) from e
    return resp.choices[0].message.content or "{}"


def _parse_proposal(raw: str) -> ProposedTarget:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ScenarioProposalError(f"model did not return valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ScenarioProposalError("model's JSON response was not an object")

    topology = str(data.get("topology", "")).strip().lower()
    if topology not in TOPOLOGY_MIN_AGENTS:
        raise ScenarioProposalError(
            f"model proposed an unrecognized topology {topology!r}; expected "
            f"one of {sorted(TOPOLOGY_MIN_AGENTS)}"
        )

    raw_agents = data.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ScenarioProposalError("model's proposal had no agents")
    agents: list[dict[str, str]] = []
    for a in raw_agents:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", "")).strip()
        role = str(a.get("role", "")).strip()
        if name and role:
            agents.append({"name": name, "role": role})
    if not agents:
        raise ScenarioProposalError("model's proposal had no usable agents (needs both name and role)")

    return ProposedTarget(
        topology=topology,
        agents=agents,
        suggested_name=str(data.get("suggested_name", "")).strip(),
        suggested_query=str(data.get("suggested_query", "")).strip(),
        rationale=str(data.get("rationale", "")).strip(),
    )


async def propose_target(scenario: str, *, model: str = DEFAULT_MODEL) -> ProposedTarget:
    """Ask an LLM to draft a topology + agent roster for a free-text
    scenario description. Returns a ProposedTarget shaped to drop
    straight into the Build tab's existing review-then-generate flow.

    Never calls generate_target_source() and never writes anything —
    this is a second, upstream step that produces a DRAFT for a human to
    review, not a second code-generation path.
    """
    scenario = scenario.strip()
    if not scenario:
        raise ScenarioProposalError("scenario description is required")
    raw = await _call_proposal_llm(_proposal_system_prompt(), scenario, model)
    return _parse_proposal(raw)
