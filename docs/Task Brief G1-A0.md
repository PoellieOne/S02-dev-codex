# Opdracht aan de agent (ik formuleer hem zo dat jij hem 1-op-1 kunt sturen)

## ✅ Task Brief G1-A0 — Minimal Wiring for Live v0.2 (PC-side)

**Doel:** maak Action Gate **v0.2** actief in de live pipeline en voeg een *extern* intent-input mechanisme toe, **zonder gedrag/semantiek te wijzigen**.

### Scope (strikt)

Toegestaan:

* Alleen *wiring* (v0.1 → v0.2 selecteren/activeren)
* Alleen *input delivery* voor `ActionIntent` (extern)
* Alleen logging die al in v0.2 bestaat (geen nieuwe semantische loglabels)

Verboden:

* Geen thresholds, geen states, geen “slimme” interpretatie
* Geen nieuwe control-logica
* Geen automatische intentgeneratie
* Geen wijziging aan BeliefState / semantische lagen

### Acceptatiecriteria (heel concreet)

De agent levert bewijs dat:

1. **Live pipeline draait met Action Gate v0.2**

   * aantoonbaar via logs/print: module/versie/klasse + state transitions

2. **Action Intent komt extern binnen per tick**

   * aantoonbaar via bestaande v0.2 logevents:

     * `ACTION_INTENT value=... source=...`
     * `GATE_DECISION … intent=… basis=…`

3. **Intent is NIET afgeleid**

   * mechanisme is expliciet: bv. keyboard, CLI-arg, eenvoudige stdin, file-toggle, API stub — maar altijd “extern” en zichtbaar als `intent_source`

4. **Wiring smoke test zonder ESP mogelijk**

   * Eén korte run die alleen laat zien dat intent wisselt en gate daarop reageert (geen beweging nodig)

### Deliverables

* Exacte lijst van gewijzigde bestanden (idealiter alleen live entrypoint + eventueel 1 klein input helper)
* Run-instructie:

  * “start live pipeline”
  * “hoe geef je ACTIVATE / HOLD / RELEASE”
  * “waar staan logs”
* Bewijslog-snippet (kort) waarin intent-events en gate-decisions zichtbaar zijn

### Stopregel / fallback

Als agent merkt dat live-wiring te invasief wordt, moet hij **stoppen** en melden:

* wat het minimale invasieve punt is
* en dan schakelen we naar **Optie B** (standalone runner) voor G1.

---
