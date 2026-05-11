I want to build an MVP for a platform that stress-tests autonomous multi-agent AI systems inside evolving environments.

The core idea is:

Current AI eval systems mostly test isolated prompts or short workflows. I want to simulate how multiple interacting agents behave over time under changing conditions, and detect emergent failures like memory drift, coordination breakdowns, escalation loops, contradictory behavior, and cascading hallucinations.

This MVP should NOT try to be production-grade or massively scalable.

I want a clean, modular Python prototype demonstrating the core idea.

Build a minimal simulation framework with the following:

1. WORLD STATE
   Create a shared mutable world state object.

Example fields:

* customer_sentiment
* refund_policy_version
* inventory_delay
* escalation_queue_size
* system_load

The world state should evolve over time.

2. AGENTS
   Create 4 simple agents:

* SupportAgent
* RefundAgent
* EscalationAgent
* PolicyAgent

Each agent:

* has a role/system prompt
* observes relevant parts of world state
* generates actions/responses using an LLM abstraction layer
* can update world state
* maintains lightweight memory

Use clean abstractions so agents can be extended later.

3. SIMULATION LOOP
   Implement a simulation loop:

for timestep in range(N):
- inject events
- agents observe state
- agents act
- world updates
- metrics/failures logged

The loop should support long-running simulations.

4. EVENT INJECTION
   Create an event system.

Example events:

* Black Friday traffic spike
* Refund policy changed
* Inventory API delayed
* Angry customer surge

Events should dynamically alter world state during runtime.

5. FAILURE DETECTION
   This is VERY important.

Implement basic emergent failure detection logic:

* contradictory refund decisions
* escalation loops
* policy inconsistency
* customer sentiment collapse
* repeated hallucinated behavior
* excessive queue growth

The system should log:

* timestep
* involved agents
* failure type
* relevant world state snapshot

6. OBSERVABILITY
   Create:

* structured logs
* event timeline
* world state snapshots
* simple metrics tracking

Keep it lightweight.

7. ARCHITECTURE GOALS
   The codebase should feel like:

* a simulation runtime
* not a chatbot demo

I care about:

* modularity
* extensibility
* clear abstractions
* simulation design

I do NOT care about:

* frontend
* deployment
* authentication
* production infra

8. TECH STACK
   Use:

* Python
* asyncio where useful
* dataclasses or pydantic
* clean folder structure

Use mock LLM calls initially if needed.

9. OUTPUT
   At the end of a simulation run, print:

* event timeline
* detected failures
* final world state
* agent interaction summaries

10. IMPORTANT
    The focus is NOT realistic customer support.

The focus IS:

* evolving world state
* interacting autonomous agents
* long-horizon behavior
* emergent failure discovery

Design the architecture accordingly.

Please generate:

* folder structure
* architecture explanation
* core implementation files
* simulation loop
* example run
* extensibility notes

Keep the MVP small but intellectually clean.
