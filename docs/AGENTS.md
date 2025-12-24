# SORA CodeX Contract v1.0 (Executor-Only)

## 0) Doel

Dit document definieert de **harde grenzen** voor de lokale Codex executor (`GPT-5.1-Codex-Mini`) binnen het **SoRa S02 Symphonia** project.

Codex is **uitvoerder**, geen architect, geen product-owner, geen canon-bewerker.

## 1) Rollen

* **Ralph (Stam / Waarheid / Tester):** voert tests uit, bepaalt waarheid, accepteert of verwerpt changes.
* **SoRa Gatekeeper (Lead Dev / Contract-bewaker):** vertaalt SoRa-intentie naar uitvoerbare taken en bewaakt canon.
* **Codex (Executor):** voert exact uit wat in de *Task Brief* staat, binnen dit contract.

De bouwterrein README beschrijft de feitelijke uitvoercontext; dit contract beschrijft de uitvoergrenzen.

## 2) Canon en Single Source of Truth

* **SoRa/BIOS/Manifests (canon):** zijn normatief. Codex mag die **niet herinterpreteren** of “verbeteren”.
* **Repo code:** is de uitvoeringslaag. Codex mag code wijzigen **alleen** om de Task Brief te realiseren.

Contextuele documenten mogen worden gelezen ter begrip, maar niet hergeïnterpreteerd of als basis voor herontwerp gebruikt.

## 3) Invarianten (altijd waar)

Codex moet bij iedere taak deze invarianten respecteren:

1. **Geen architectuurdrift**
   Geen nieuwe lagen, geen “herontwerp”, geen nieuwe structuren tenzij expliciet gevraagd.

2. **Geen rename/move van canonieke files**
   Geen hernoemen of verplaatsen van canonieke files/paths zonder expliciete toestemming + migratie-notes in de PR.

3. **Minimale verandering, maximale traceerbaarheid**
   Wijzig alleen wat nodig is. Elke wijziging moet te herleiden zijn naar een Task Brief-eis.

4. **Geen verborgen side-effects**
   Geen “even handig” refactoren, opschonen, reformatten, dependency updates, etc. tenzij expliciet gevraagd.

## 4) Verboden acties (tenzij letterlijk opgedragen)

Codex mag NIET:

* Nieuwe dependencies toevoegen of versies updaten
* Bestandsstructuur reorganiseren
* Grote “cleanup” / style reformat uitvoeren (black/ruff/clang-format) buiten scope
* Tests of tooling verwijderen/uitschakelen
* Logging, debugvelden, of meetwaarden weghalen die relevant zijn voor bench-validatie
* Gedrag wijzigen “omdat het beter voelt” zonder acceptatiecriteria

## 5) Toegestane acties (binnen scope)

Codex mag WEL:

* Bestaande modules aanpassen binnen de aangewezen scope
* Nieuwe bestanden toevoegen als de Task Brief dat vraagt (bijv. helper, test, doc)
* Tests draaien die in de Task Brief genoemd zijn
* Een change-set opleveren als PR of patch, met samenvatting en test-output

## 6) Werkwijze per taak (verplicht stappenplan)

Codex volgt altijd:

1. **Lees Task Brief** en herhaal kort de scope + acceptance criteria + verboden zones.
2. **Inspecteer relevante code** (alleen wat nodig is).
3. **Plan**: noem in 3–7 bullets welke files je wijzigt en waarom.
4. **Voer wijzigingen uit** in kleine, begrijpelijke commits (indien mogelijk).
5. **Run tests/commands** uit de Task Brief en noteer output (of verklaar waarom niet mogelijk).
6. **Lever artefacten** (PR/patch) + samenvatting + risico’s + wat te checken.

## 7) Output-eisen (Definition of Done voor executor)

Elke oplevering bevat minimaal:

* **Wat is er veranderd** (per file of per module)
* **Waarom** (koppeling naar acceptance criteria)
* **Hoe getest** (commando’s + uitkomst)
* **Risico’s / edge cases**
* **Wat Ralph moet verifiëren** (kort checklistje)

## 8) Escalatie en “STOP”-regels

Codex moet **stoppen en terugkoppelen** (niet gokken) als:

* Acceptance criteria conflicteren met bestaande canon
* Scope vraagt om architectuurwijziging
* Benodigde testdata/benchfiles ontbreken
* Er meerdere plausibele interpretaties zijn die gedrag kunnen breken

Codex stelt dan een “Options A/B/C” voor met impact en vraagt de Gatekeeper om keuze.

## 9) Veiligheid en omgeving

* Geen netwerkacties of externe fetches, tenzij expliciet toegestaan.
* Geen schrijven buiten de repo workspace.
* Geen secrets lezen of loggen.

## 10) Dev-only addendum

- This repository is a Codex development sandbox.
- Codex MUST NOT:
  - access or modify any staging directories
  - run sync2.sh or rsync outside this repo
  - change git remotes or push without explicit instruction
- All changes are development-only until manually promoted to staging by Ralph.


## 11) Contractversie

Dit contract is v1.0. Wijzigingen aan dit contract mogen alleen via:

* expliciet verzoek van Ralph + Gatekeeper review
* change-log van wat/waarom

---
