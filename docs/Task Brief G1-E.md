# 📄 TASK BRIEF G1-E — SCOREBOARD RUST + FILE LOGGING (Human-Friendly Observability)

## Identiteit

* Opsteller: Gatekeeper (Sophia)
* Ontvanger: Execution Agent
* Context: `scripts/live_symphonia_v2_0.py`
* Log root (canoniek): `testbench/out/`
* Discipline: **geen semantiek**, **geen control-logica**, **geen thresholds/states**

---

## Probleem (feitelijk)

* `--scoreboard` hertekent te vaak → Ralph ervaart dit als “loop” (onwerkbaar tijdens handbewegingen)
* logflags geven extra terminal-output → onrust
* zonder logflags moet Ralph weer copy/pasten → onwenselijk

---

## Doel (bindend)

Maak het mogelijk om:

1. een **rustig scoreboard** te hebben (lage refresh-rate)
2. **alle logs automatisch naar bestand** te schrijven in `testbench/out/`
3. de terminal **stil** te houden (alleen scoreboard, of niets)

Zonder wijziging aan systeemgedrag.

---

## Vereiste features (minimaal)

### A) Scoreboard refresh-rate instelbaar

Voeg één van deze opties toe (kies één stijl, liefst beide niet):

* `--scoreboard-hz <float>` (default bijv. 2; Ralph gebruikt 1)
  **of**
* `--scoreboard-interval-ms <int>` (default bijv. 500; Ralph gebruikt 1000)

Acceptatie: Ralph kan draaien met 1 update/sec en ervaart het als “statisch genoeg”.

### B) Quiet console mode

Voeg optie toe:

* `--quiet-console`

Gedrag:

* Als `--quiet-console` + `--scoreboard`: alleen scoreboard output (geen extra prints)
* Als `--quiet-console` zonder scoreboard: geen stdout spam (alleen errors naar stderr is oké)

### C) File logging altijd beschikbaar

Voeg optie toe (of behoud bestaande, maar maak betrouwbaar):

* `--log-dir <path>` default: `testbench/out`

Gedrag:

* schrijft een timestamped logfile, bv:

  * `testbench/out/live_YYYYMMDD_HHMMSS.log`
* bevat ten minste:

  * timestamps
  * `ACTION_INTENT`
  * `GATE_ENTER`
  * `GATE_BASIS`
  * `GATE_DECISION`
* **geen filtering**, **geen inhoudelijke verandering**, alleen routing

---

## Deliverables

1. Patch (klein) voor `scripts/live_symphonia_v2_0.py` (+ eventueel 1 helper)
2. Exacte run-commando’s voor Ralph:

**Rustig dashboard + logfile:**

```bash
python3 scripts/live_symphonia_v2_0.py \
  --intent-file testbench/out/intent.txt \
  --scoreboard \
  --scoreboard-hz 1 \
  --quiet-console \
  --log-dir testbench/out
```

3. Bewijs:

* naam van logfile die is aangemaakt
* 10 regels uit die logfile met `ACTION_INTENT` en `GATE_DECISION`

---

## Verboden (herhaal)

* geen nieuwe thresholds
* geen nieuwe states
* geen semantische labels
* geen gedrag aanpassen
* geen “interpretatieve” UI

---
