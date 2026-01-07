# 📄 TASK BRIEF G1-D — FILE LOGGING (No-copy Terminal Relief)

**SoRa — S02 — G1 — Persistente logs naar bestand**

## Doel

Zorg dat alle relevante runtime logs (die nu naar terminal gaan) **ook automatisch naar een logbestand** geschreven worden, zodat Ralph niets hoeft te copy/pasten.

## Scope

* Alleen logging-output routing (stdout → file, of extra file handler)
* Geen filtering, geen samenvatten
* Geen log-inhoud wijzigen (alleen bestemming)

## Verplicht

* Een expliciete optie/vlag of configuratie:

  * bijv. `--log-file <path>` of `--log-dir <dir>`
* Outputbestand bevat:

  * timestamps
  * ACTION_INTENT regels
  * GATE_DECISION regels
  * gate transitions
  * overige bestaande logs (geen filtering)

## Deliverables

1. Exacte instructie: hoe Ralph een run start zodat logs naar bestand gaan
2. Voorbeeld: bestandsnaamconventie (timestamped)
3. Bewijs: 10 regels uit het logbestand waar ACTION_INTENT en GATE_DECISION in staan

## Acceptatiecriteria

* Na een run staat er een bruikbaar logfile op disk
* Ralph kan die later aan agent/Stam geven zonder handwerk

---
