# ADR-0009: Ship the dashboard inside the package; install from GitHub until a PyPI name and license are chosen

| | |
|---|---|
| Status | Accepted (packaging, install path). **Open: license and PyPI name**, which need the owner's decision |
| Decided | 2026-11-19 16:26 IST — `9472c30` (install path); 16:29 IST — `66a4b80` (packaging) |
| Recorded | 2026-11-19, the same day |
| Owner | Soham Agarwal |

## Context
The plan since June has been an open-source, pip-installable tool. Three
facts surfaced on 2026-11-19:
1. **The web dashboard did not ship.** `web/` sat at the repo root, outside
   the Python package. drift installed from a wheel (which is what
   `pip install git+…` builds) answered `drift serve` with "web/ directory
   not found". Every test ran from a checkout, so no test noticed.
2. **The name `drift` on PyPI belongs to an unrelated project** (versions
   0.0.1–0.0.7). The README told users to `pip install drift[langgraph]`,
   which installs that project.
3. **The repository has no LICENSE file.** Public code with no license is
   legally "all rights reserved", whatever the README intends.

## Decision
- The dashboard moves to `src/drift/web/` and is declared as package
  data.
- Installed outside a checkout, results are written under the directory
  drift was started from, not inside site-packages.
- A CI job builds the wheel, installs it into a clean virtual environment
  and loads the dashboard from outside the repo.
- Documentation installs from GitHub
  (`pip install "drift[langgraph] @ git+https://github.com/SohamAg/drift"`),
  with a warning about the PyPI name, until drift is published.
- **Not decided here:** the license and its copyright holder (Soham
  Agarwal personally, or Sutra Labs LLC once the founder IP assignment is
  signed), and the name to publish under. Both are owner decisions with
  legal weight. They are left open rather than assumed.

## Alternatives considered
- **Add an MIT LICENSE now**, because the plan said MIT. Not done: choosing
  the copyright holder depends on the unsigned IP assignment.
- **Publish under a guessed name.** Not done: a published name cannot be
  taken back cleanly.

## Consequences
- What a user installs is now tested, not only what the repo runs.
- Until the license is added, others have no legal right to use the code.
  This should be the first follow-up.
