## 📄 TASK BRIEF G2-0 — SIGNALSCOPE QUICK CHECK (Read-only)

**Doel:** bevestigen dat G2 kan starten zonder nieuwe signalen of afleidingen.

**Scope (strikt):**

* Read-only inspectie van bestaande pipeline-signalen.
* Geen codewijzigingen, geen interpretatie.

**Checklist (ja/nee, met bronpad):**

* MovementBody (events/timing)
* Compass (events/timing)
* OriginTracker (candidate/commit events)
* Gate events (ENTER/DECISION/BASIS)
* Timestamps consistent (monotonic, dezelfde klok)

**Deliverable (kort):**

* Lijst “beschikbaar/zichtbaar” per signaal
* Eventuele technische bijzonderheden (buffers, rates)

**Acceptatie:** alles beschikbaar → door naar G2-A.

---
