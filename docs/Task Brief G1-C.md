# 📄 TASK BRIEF G1-C — DASHBOARD READINESS CHECK (Observer UI)

**SoRa — S02 — G1 — Boundary Stress Experiment #1 — Observability Enablement**

## Identiteit

* Opsteller: Gatekeeper (Sophia)
* Ontvanger: Execution Agent
* Doel: dashboard verifieerbaar bruikbaar maken voor Ralph’s handmatige tests
* Scope: PC-side dashboard/visualisatie tooling (bestaand)
* Verboden: geen semantische interpretatie, geen nieuwe control-logica

## Doel

Controleer of het **bestaande dashboard** (pre-BeliefState, “te uitgebreid”) de **minimale velden** toont die Ralph nodig heeft om G1 scenario’s live te kunnen uitvoeren met handbewegingen.

Indien ontbrekend: **minimale uitbreiding** zodat de test uitvoerbaar is.

## Verplichte “Minimum Visible Set” (MVS)

Dashboard moet live zichtbaar maken (zonder interpretatieve labels):

1. **Action Intent** (huidige waarde)
2. **Gate state** (IDLE/OBSERVE/ARMED/ACTIVE/FALLBACK) + transitions
3. **Gate decision output** (bijv. FORCE_FALLBACK / etc.)
4. **Basisvelden** die al bestaan in v0.2 logs (zoals eerder gedefinieerd):

   * coherence
   * lock state
   * rotor present/flag (of equivalent)
   * data_age_ms (of equivalent)

## Randvoorwaarden (bindend)

* Geen nieuwe thresholds / states
* Geen “samenvatting” of patroonbenoeming
* Geen semantische termen in UI
* Geen wijzigingen aan BeliefState / semantische lagen
* Alleen “toon wat er al is”

## Deliverables

1. Korte inventaris: *welk dashboardbestand / script / entrypoint* is het huidige dashboard
2. MVS-checklist: voor elk MVS-item: **aanwezig ja/nee**, en waar zichtbaar
3. Indien “nee”: minimale wijziging + bewijs (screenshot of korte logregel “field displayed”)
4. Reproduceerbare run-instructie: hoe Ralph dashboard start + live pipeline start

## Acceptatiecriteria

* Ralph kan tijdens handbewegingen in één oogopslag intent + gate state + basisvelden volgen
* Geen nieuwe betekenislaag toegevoegd

---
