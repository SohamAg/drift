"""Framework adapters — let users plug an existing agent app into drift.

The shipped @drift.agent / WorldState pattern requires the user to author
their agents in drift's SDK. Most prospective users already have a working
multi-agent app built on LangGraph / CrewAI / OpenAI Agents SDK / Autogen
and won't rewrite it just to try drift. Adapters in this package treat the
user's compiled app as a black box callable and run drift's schema-driven
auto-chaos against it.

The first shipped adapter is `drift.adapters.langgraph`. It needs nothing
from the langgraph package itself — any object with `.invoke(state)` (or
`.ainvoke(state)`) returning a dict works. That's the natural shape of a
compiled langgraph `StateGraph`, but also covers homemade pipelines.
"""
