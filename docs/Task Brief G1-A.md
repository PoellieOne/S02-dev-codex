# 📄 TASK BRIEF G1-A — EXECUTION & DATA CAPTURE

**SoRa — S02 — G1 — Boundary Stress Experiment #1**

## 1. Identiteit

* **Opsteller:** SoRa S02 Gatekeeper
* **Ontvanger (agent):** *Execution Agent* (nieuw / vers)
* **Rol agent:** Uitvoerend & registrerend
* **Samenwerking:** Ralph (tester)
* **Architectuurstatus:** read-only, afgeleid
* **Context:** PC-side realtime pipeline
* **Action Gate:** v0.2 actief

---

## 2. Doel van deze taak

Voer het **Boundary Stress Experiment #1** uit door het systeem bewust in **randcondities** te brengen, met als enig doel:

> **het vastleggen van ruwe, onbewerkte observatiedata**

Deze taak:

* **observeert**
* **registreert**
* **legt vast**

Zij **verklaart niets**
Zij **concludeert niets**

---

## 3. Scope (strikt)

De agent voert **uitsluitend** uit:

* de drie gespecificeerde scenario’s
* op de bestaande codebase
* met Action Gate v0.2
* zonder enige codewijziging

**Niet binnen scope:**

* aanpassen van thresholds
* toevoegen van states
* optimalisaties
* interpretatie
* patroonbenoeming
* semantische taal

---

## 4. Scenario’s (bindend, niet variëren)

Elke scenario:

* **minimaal 3 afzonderlijke runs**
* identiek uitgevoerd per run

### 4.1 Scenario 1 — *Net-geen-lock*

* Action Intent = `INTENT_ACTIVATE`
* Breng het systeem herhaaldelijk **net onder lock-coherentie**
* Laat de Gate:

  * weigeren
  * of terugvallen
* Geen forcering richting ACTIVE

---

### 4.2 Scenario 2 — *Fragiel ritme*

* Introduceer **lichte onregelmatigheid** in beweging
* Geen harde breuk
* Wel: “bijna-herhaling”
* Observeer overgangen:

  * OBSERVE ↔ ARMED ↔ FALLBACK

---

### 4.3 Scenario 3 — *Herstel na release*

* ACTIVE → `INTENT_RELEASE`
* Observeer:

  * afbouw
  * terugkeer naar OBSERVE
* Daarna opnieuw:

  * `INTENT_ACTIVATE`
* Geen versnelling, geen shortcuts

---

## 5. Data & Logging (verplicht)

De agent logt **alles wat reeds bestaat**, plus expliciet:

* Gate state transitions
* Action Intent changes
* Basisvelden bij beslissingen (zoals gedefinieerd in Action Gate v0.2)

### Regels:

* **Geen filtering**
* **Geen herstructurering**
* **Geen samenvatting in code**

### Outputvorm:

* ruwe logs (console / bestand)
* tijdgestempelde events
* optioneel: CSV of JSONL dump

---

## 6. Output aan Gatekeeper (via Ralph)

De agent levert **alleen**:

1. **Ruwe datasets**

   * waar opgeslagen
   * bestandsnamen
2. **Run-instructies**

   * hoe exact te reproduceren
3. **Technische bijzonderheden**

   * timing
   * stabiliteit
   * onverwachte resets of haperingen

**Geen interpretatie.
Geen conclusies.
Geen hypothesen.**

---

## 7. Acceptatiecriteria (Gatekeeper)

Deze taak is **VALIDATED** indien:

* alle drie scenario’s zijn uitgevoerd
* minimaal 3 runs per scenario
* data volledig en onbewerkt beschikbaar is
* geen scope-overtredingen zijn geconstateerd

Bij afwijking → **BLOCKED** (met expliciete reden).

---

## 8. Relatie tot vervolg

Na VALIDATED:

* Task Brief G1-B — *Data Packaging*
* Daarna: overdracht aan **Stam** voor herkenning en betekenisvorming

---
