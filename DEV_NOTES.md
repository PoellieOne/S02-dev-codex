# Development Notes — SoRa S02 Dev Sandbox

## 1. Context
- Document the intended workflow for engineers exercising this repository during SoRa S02 development.
- Audience: Codex executor plus reviewers who need quick orientation before running or extending any tooling.

## 2. Repository Role
- Local development sandbox only; writable workspace for Codex while broader canon (SoRa/BIOS/Manifests) stays immutable elsewhere.
- `dev` is where Codex executes changes, `staging` is the next gated promotion, and `cloud/gist` artifacts serve archival or external sharing purposes outside this repo.
- Codex may write here because the sandbox explicitly allows workspace-write operations, and instructions direct Codex to produce artifacts that do not overwrite canon.

## 3. Main Components
- `scripts/`: CLI tools, live runners, analysers, and helpers that drive ESP32 communication, log parsing, and visualization.
- `sym_cycles/`: Core Python package implementing CycleBuilder, PhaseTiles, CompassSign, movement awareness, and both realtime/offline pipelines.
- `Legacy/`: Archived realtime state revisions kept for reference; no edits unless a Task Brief explicitly resurrects a legacy version.

## 4. Working Agreements
- Codex works as an executor-only agent: read the Task Brief, inspect relevant files, plan edits, and apply only the minimal change set needed.
- No refactors, no renames, no behavior changes beyond the request, and no touching canonical files unless the brief explicitly says so.
- Promotion to `staging` happens conceptually via Ralph’s manual review and acceptance; Codex prepares the change set here, and Ralph decides when to push or merge upstream.

## 5. Explicitly Not
- This document is not canon; it does not represent BIOS or any other single source of truth.
- It contains no definitive system behavior guarantees and should not be treated as authoritative technical specification or final truth.
