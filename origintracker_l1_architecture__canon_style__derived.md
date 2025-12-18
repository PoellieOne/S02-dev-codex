> NON-CANON / DERIVED / CODE-ANCHORED
> Scope: OriginTracker (L1) via l1_physical_activity.py (+ live_symphonia_v2_0.py voor call-site)
> Bronpad: /home/ralph/PoellieOne/S02-dev-codex

# OriginTracker L1 Architecture — Derived Document

**Versie:** v0.4.5 (code-afgeleid)
**Scope:** L1PhysicalActivity klasse + MDI modes

---

## 1. Doel en afbakening

**Wat je hier leert:** OriginTracker detecteert beweging voordat cycles beschikbaar zijn. Het vertaalt pool-transities naar een awareness-state die van STILL naar MOVEMENT loopt.

### Doel (in code zichtbaar)

OriginTracker is onderdeel van `L1PhysicalActivity` en biedt:

- **Pre-cycle bewegingsdetectie** via MDI (Micro-Displacement Integrator)
- **Two-phase origin tracking** (candidate → commit)
- **Awareness state machine** met vijf states

CODE-EVIDENCE: `class L1PhysicalActivity` in `l1_physical_activity.py` (regel 198)

### Buiten scope

- L2 cycle-gebaseerde richting (dat is `RealtimePipeline`)
- CompassSign v3 (L3 layer)
- Directe event-parsing (dat doet de caller)

---

## 2. Terminologie

**Wat je hier leert:** Deze termen komen direct uit de code. Elk veld of concept is traceerbaar naar een specifieke plek.

| Term | Betekenis | Code-evidence |
|------|-----------|---------------|
| **AwState** | Awareness state enum (STILL/NOISE/PRE_MOVEMENT/PRE_ROTATION/MOVEMENT) | `class AwState(Enum)` regel 28-34 |
| **AwReason** | Reden voor state-transitie (MDI_TRIGGER, CANDIDATE_POOL, etc.) | `class AwReason(Enum)` regel 35-57 |
| **MDI** | Micro-Displacement Integrator — pre-cycle verplaatsingsdetectie | `_process_mdi_step()` regel 255-273 |
| **mdi_micro_acc** | Accumulator voor micro-stappen (float, max 36.0) | `self._mdi_micro_acc` regel 213, 270 |
| **mdi_conf_acc** | EMA-smoothed confidence score | `_update_mdi_conf_acc()` regel 302-310 |
| **theta_hat_rot** | Geschatte rotatie in rotaties (cycles/cycles_per_rot) | `self._theta_hat_rot` regel 445 |
| **pool window** | Sliding window (250ms) voor pool-evidence | `self._pool_window: deque` regel 211 |
| **origin_candidate_set** | Boolean: pool-evidence voldoende voor kandidaat | `self._origin_candidate_set` regel 228 |
| **origin_commit_set** | Boolean: angle-displacement bevestigd | `self._origin_commit_set` regel 230 |
| **disp_acc_deg** | Geaccumuleerde hoekverplaatsing in graden | `self._disp_acc_deg` regel 232 |
| **gap** | Tijd sinds laatste event/cycle (ageE_s, ageC_s) | Berekend in `update()` regel 454-455 |
| **latch** | Mode C fast-start mechanisme | `self._mdi_latch_set` regel 221 |

**Hoe herken je dit in logs:** Zoek naar `aw_state`, `aw_reason`, `mdi_micro_acc`, `mdi_conf_acc` in JSONL output.

---

## 3. Interfaces en Inputs

**Wat je hier leert:** L1 krijgt data via twee methodes. De caller (live runner) roept deze aan per event en per tick.

### record_pool() — Per event

CODE-EVIDENCE: `def record_pool(self, to_pool, sensor: int, t_s: float = None)` regel 243

**Parameters:**
- `to_pool`: int (0=NEU, 1=N, 2=S, 3=invalid)
- `sensor`: int (0=A, 1=B)
- `t_s`: timestamp (optioneel, default: laatste update tijd)

**Wat het doet:**
- Vult `_pool_window` en `_mdi_window` (sliding buffers)
- Roept `_process_mdi_step()` aan voor pool-waarden 0/1/2

### update() — Per tick

CODE-EVIDENCE: `def update(self, wall_time: float, cycles_physical_total: float, events_this_batch: int = 0, ...)` regel 433

**Verplichte parameters:**
- `wall_time`: huidige tijd in seconden
- `cycles_physical_total`: cumulatieve cycles van L2
- `events_this_batch`: aantal events sinds laatste update

**Optionele parameters:**
- `direction_conf`: richting-confidence van L2
- `lock_state`: lock status ("UNLOCKED"/"SOFT_LOCK"/"LOCKED")
- `direction_effective`: effectieve richting ("CW"/"CCW"/"UNDECIDED")

**Call-site voorbeeld** (live_symphonia_v2_0.py regel 430-431):
```python
snap = l1.update(wall_time=now, cycles_physical_total=cy_total, events_this_batch=ev_batch,
                 direction_conf=dc, lock_state=lk, direction_effective=de)
```

**Hoe herken je dit in logs:** Het `ev` veld in JSONL is `events_this_batch`, `cy` is `cycles_physical_total`.

---

## 4. State model

**Wat je hier leert:** Er zijn twee parallelle state machines: L1State (tactiel) en AwState (awareness). De awareness states zijn het primaire output.

### AwState (primair)

CODE-EVIDENCE: `class AwState(Enum)` regel 28-34

| State | Betekenis | Trigger sources (code) |
|-------|-----------|------------------------|
| STILL | Geen activiteit | `activity_score < activity_threshold_low` (regel 596), gap timeout (regel 489-491) |
| NOISE | Activiteit zonder verplaatsing | `activity_score >= activity_threshold_low` (regel 595) |
| PRE_MOVEMENT | MDI detecteert micro-verplaatsing | MDI trigger/latch (regel 509-510) |
| PRE_ROTATION | Pool-evidence kandidaat gezet | `origin_candidate_set` (regel 593), commit zonder movement (regel 592) |
| MOVEMENT | Bevestigde beweging | `disp_from_origin_deg >= movement_confirm_deg` (regel 589), speed confirm (regel 590), lock (regel 591) |

### AwReason (trigger context)

CODE-EVIDENCE: `class AwReason(Enum)` regel 35-57

Belangrijke reasons:
- `MDI_TRIGGER` / `MDI_LATCH`: MDI heeft beweging gedetecteerd
- `CANDIDATE_POOL`: pool-evidence sterk genoeg
- `COMMIT_ANGLE`: hoek-threshold behaald + horizon doorstaan
- `STOP_GAP_TIMEOUT`: geen activiteit gedurende `stop_gap_s`

### L1State (secundair, tactiel)

CODE-EVIDENCE: `class L1State(Enum)` regel 59-64

States: STILL, FEELING, SCRAPE, DISPLACEMENT, MOVING

INFERENTIE: L1State lijkt een legacy/parallel systeem; AwState is de primaire awareness output.

**Hoe herken je dit in logs:** `aw.state` en `aw.reason` in JSONL; `l1.state` voor tactiele state.

---

## 5. Datamodel

**Wat je hier leert:** L1Snapshot bevat alle observeerbare velden. Interne accumulators leven in de klasse-instantie.

### Interne accumulators (L1PhysicalActivity)

CODE-EVIDENCE: `__init__` regel 201-236

| Veld | Type | Doel |
|------|------|------|
| `_mdi_micro_acc` | float | Geaccumuleerde micro-stappen (0..36) |
| `_mdi_tremor_score` | float | Flipflop-penalty (0..1) |
| `_mdi_conf_acc` | float | EMA-smoothed confidence |
| `_mdi_window` | deque | Sliding window voor MDI stats |
| `_pool_window` | deque | Sliding window voor pool evidence |
| `_origin_candidate_set` | bool | Kandidaat actief |
| `_origin_commit_set` | bool | Commit actief |
| `_disp_acc_deg` | float | Totale hoekverplaatsing |
| `_speed_deg_s` | float | EMA speed in graden/sec |
| `_mdi_latch_set` | bool | Mode C latch actief |
| `_mdi_confirmed` | bool | Mode C latch bevestigd |

### L1Snapshot output velden

CODE-EVIDENCE: `@dataclass class L1Snapshot` regel 133-193

**MDI velden:**
- `mdi_mode`: str — actieve mode ("A"/"B"/"C")
- `mdi_ev_win`: int — events in MDI window
- `mdi_micro_acc`: float — accumulator waarde
- `mdi_disp_micro_deg`: float — micro_acc × step_size
- `mdi_conf` / `mdi_conf_acc` / `mdi_conf_used`: float — confidence scores
- `mdi_tremor_score`: float — flipflop indicator
- `micro_t0_s`: Optional[float] — eerste micro-displacement timestamp
- `mdi_latch_set` / `mdi_confirmed`: bool — latch state (Mode C)

**Origin velden:**
- `origin_candidate_set` / `origin_commit_set`: bool
- `origin_time0_s`: Optional[float] — vroegste bewegingstijd
- `origin_time_s`: Optional[float] — commit tijd
- `origin_theta_deg`: Optional[float] — origin hoek

**Displacement velden:**
- `disp_acc_deg`: float — totale accumulatie
- `disp_from_origin_deg`: float — verplaatsing sinds commit
- `speed_deg_s`: float — huidige snelheid
- `early_dir`: str — vroege richtingindicatie

**Hoe herken je dit in logs:** Alle velden zijn direct geserialiseerd in JSONL onder `mdi`, `candidate`, `commit`, `disp` secties.

---

## 6. Lifecycle en Flow

**Wat je hier leert:** Per tick doorloopt update() een vaste volgorde. De volgorde bepaalt wanneer resets vs state-changes gebeuren.

### Per tick (update() flow)

CODE-EVIDENCE: `def update()` regel 433-584

1. **Time delta berekening** (regel 437-439)
   - `dt_s` = now - last_update
   - Hard reset als `dt_s > hard_reset_s`

2. **Cycle/event accounting** (regel 441-452)
   - `delta_cycles` = nieuwe - vorige cycles
   - Update `t_last_cycle_s` en `t_last_event_s`

3. **Age berekening** (regel 454-456)
   - `ageE_s` = tijd sinds laatste event
   - `ageC_s` = tijd sinds laatste cycle

4. **Activity/encoder decay** (regel 462-467)
   - Exponentieel verval van `activity_score`
   - `encoder_conf` update op basis van cycles/events

5. **MDI stats berekening** (regel 472-478)
   - `_compute_pool_stats()` voor pool window
   - `_compute_mdi_stats()` voor MDI window
   - `_compute_mdi_conf()` voor confidence
   - Step size bepalen via `_get_step_size()`

6. **MDI mode evaluatie** (regel 480-483)
   - `_apply_mdi_mode()` retourneert (triggered, reason)
   - Alleen als tremor <= threshold

7. **Gap handling** (regel 488-514)
   - Hard gap: volledige reset
   - Soft gap: origin reset, MDI behouden (als niet actief)
   - Hold decay: speed × 0.9

8. **Origin candidate/commit logic** (regel 516-547)
   - Pool evidence check voor candidate
   - Angle threshold + horizon voor commit

9. **AwState berekening** (regel 560)
   - `_compute_aw()` bepaalt finale state/reason

10. **Snapshot constructie** (regel 567-584)
    - Alle velden verzameld in `L1Snapshot`

**Hoe herken je dit in logs:** De volgorde van velden in JSONL weerspiegelt deze flow: eerst ages, dan mdi, dan candidate/commit, dan aw.

---

## 7. Reset, gaps en guards

**Wat je hier leert:** Reset-gedrag is afhankelijk van gap-type en MDI-activiteit. De `mdi_active` guard voorkomt onterecht verlies van PRE_MOVEMENT.

### Reset triggers

CODE-EVIDENCE: `_reset_origin()` regel 406-431

| Trigger | Conditie | keep_tactile | reset_mdi |
|---------|----------|--------------|-----------|
| STOP_GAP_TIMEOUT | `ageC >= stop_gap_s OR ageE >= stop_gap_s` AND `activity < activity_reset_a0` | False | True |
| NO_DISP_ACTIVE | `ageC >= noise_gap_s` AND `activity >= activity_reset_a0` AND NOT `mdi_active` | True | False |
| MDI_TREMOR | `mdi_tremor > mdi_tremor_max` AND `aw_state == PRE_MOVEMENT` | True | True |
| MDI_HOLD_TIMEOUT | `aw_state == PRE_MOVEMENT` AND NOT `origin_candidate_set` AND `ageE > mdi_hold_s` AND `activity < threshold_low` | False | True |
| CANDIDATE_DROPPED | `origin_candidate_set` AND `pool_chg == 0` AND `activity < threshold_low` | False | True |

### MDI active guard (v0.4.5)

CODE-EVIDENCE: regel 486

```python
mdi_active = mdi_triggered or self._mdi_latch_set or self._aw_state == AwState.PRE_MOVEMENT
```

Dit voorkomt dat een cycle-gap (NO_DISP_ACTIVE) de MDI-state wist terwijl PRE_MOVEMENT actief is.

### Gap config defaults

CODE-EVIDENCE: `L1Config` regel 128-131

- `stop_gap_s`: 0.80 (hard reset threshold)
- `noise_gap_s`: 0.50 (soft reset threshold)
- `movement_hold_s`: 0.25 (hold decay start)
- `activity_reset_a0`: 0.20 (minimum activity voor soft reset)

**Hoe herken je dit in logs:** Reset is zichtbaar als `aw.reason` plotseling `STOP_GAP_TIMEOUT` of `NO_DISP_ACTIVE` wordt, met `aw.state` naar STILL/NOISE.

---

## 8. Origin candidate vs commit

**Wat je hier leert:** De two-phase origin scheidt "begin van beweging detecteren" van "bevestigen dat het echt beweging is".

### Phase 1: Candidate (pool evidence)

CODE-EVIDENCE: regel 519-526

**Trigger conditie:**
```python
strong = pool_chg >= pool_changes_min and len(valid_pools) >= pool_unique_min and pool_vr >= pool_valid_rate_min
```

**Defaults:**
- `pool_changes_min`: 2
- `pool_unique_min`: 2
- `pool_valid_rate_min`: 0.70

**Bij trigger:**
- `origin_candidate_set = True`
- `origin_candidate_time_s = now_s`
- `origin_time0_s = micro_t0_s or now_s` (prefer MDI timestamp)
- `aw_state → PRE_ROTATION`

### Phase 2: Commit (angle displacement)

CODE-EVIDENCE: regel 531-547

**Trigger conditie:**
1. `abs(disp_acc_deg) >= origin_step_deg` (default: 30°)
2. Start horizon timer
3. Wacht `origin_commit_horizon_s` (default: 0.35s)
4. Check rebound: als `abs(disp_acc_deg) < origin_rebound_eps_deg` (default: 10°) → reset horizon

**Bij commit:**
- `origin_commit_set = True`
- `origin_time_s = now_s`
- `origin_theta_hat_rot = theta_hat_rot - disp_acc_deg/360`
- `origin_conf = 0.6`
- `aw_reason = COMMIT_ANGLE`

### Timeline prioriteit

CODE-EVIDENCE: regel 524, 543

`origin_time0_s` wordt gezet als de **vroegste** van:
1. `micro_t0_s` (MDI eerste detectie)
2. `origin_candidate_time_s` (pool evidence)
3. `now_s` (commit tijd)

Dit zorgt dat de "echte start" altijd het eerste loskomen vastlegt.

**Hoe herken je dit in logs:** Check `candidate.set`, `candidate.t0`, `commit.set`, `commit.t0`, `commit.tC` in JSONL.

---

## 9. Observability

**Wat je hier leert:** De snapshot bevat alle velden nodig voor logging en debugging. De runner serialiseert dit naar JSONL.

### Diagnostic velden in L1Snapshot

CODE-EVIDENCE: `L1Snapshot` regel 133-193

| Veld | Doel |
|------|------|
| `ageE_s` / `ageC_s` | Freshness van events/cycles |
| `l2_stale` | Boolean voor cycle-staleness |
| `mdi_ev_win` | Events in MDI window (debug) |
| `mdi_micro_deg_per_step_used` | Actuele step size (Mode B) |
| `mdi_latch_reason` | Reden voor latch state |
| `to_pool_hist` | Histogram van alle pool-waarden |

### Runner logging (live_symphonia_v2_0.py)

CODE-EVIDENCE: `live_symphonia_v2_0.py` regel 472-541

De runner schrijft JSONL met:
- `t_s`: elapsed time
- `ev`: events dit tick
- `cy`: totale cycles
- `mdi`: complete MDI state
- `candidate` / `commit`: origin state
- `aw`: awareness state/reason
- `_cycle_truth`: debug info voor cycle-mismatch detectie

### Scoreboard output

CODE-EVIDENCE: regel 543-575

In scoreboard mode print de runner:
```
PRE_MOVEMENT MDI_TRIGGER             cand=- comm=- [C] μ=  25° ev=4 latch=L✓
  CycleTruth: used=0 cb=0 src=total_cycles_physical
  Flow: tiles=0 w/cycles=0 cycles_in_tiles=0.0 last_tile=-
```

**Hoe herken je dit in logs:** Alle velden uit deze sectie zijn direct terug te vinden in JSONL; scoreboard geeft real-time samenvatting.

---

## 10. Open vragen

**Wat je hier leert:** Deze vragen zijn afleidbaar uit de code maar niet expliciet beantwoord.

1. **MDI mode selectie runtime**: De mode wordt bij init gezet (`mdi_mode` in config). ONBEKEND: kan mode runtime wisselen zonder reset?

2. **Tremor threshold tuning**: `mdi_tremor_max = 0.60` is default. ONBEKEND: hoe is deze waarde bepaald?

3. **Confidence EMA tau**: `mdi_conf_tau_s = 0.30`. ONBEKEND: relatie tot fysieke responsiviteit?

4. **Origin rebound epsilon**: `origin_rebound_eps_deg = 10.0`. ONBEKEND: waarom precies 10°?

5. **Mode A confirm timing**: `mdi_confirm_s_A = 0.25`. ONBEKEND: waarom korter dan Mode C latch timing?

6. **L1State vs AwState**: Beide bestaan parallel. ONBEKEND: wanneer gebruikt een consumer L1State vs AwState?

7. **Cycles_per_rot calibratie**: Default 12.0. ONBEKEND: hoe wordt dit per encoder bepaald?

8. **Pool value 3 handling**: In `record_pool()` wordt pool=3 als "other" geteld. ONBEKEND: wat betekent pool=3 fysiek?

---

## Template Conformance

Dit document volgt de structuur en stijl van het template `S02_OriginTracker_Architecture_v0_4_x__TEMPLATE.md`:

1. **Kopvolgorde**: De 10 secties (Doel, Terminologie, Interfaces, State model, Datamodel, Lifecycle, Reset, Origin, Observability, Open vragen) volgen exact de template-structuur.

2. **"Wat je hier leert" openers**: Elke sectie begint met een korte samenvatting van wat de lezer leert, conform de template-stijl.

3. **Tabel-formaat voor terminologie en states**: Dezelfde compacte tabelstijl als het template, met kolommen voor term/betekenis/evidence.

4. **"Hoe herken je dit in logs" observaties**: Per sectie een concrete hint voor testers, conform de schrijfregel uit het template.

5. **CODE-EVIDENCE / INFERENTIE labels**: Consistent gebruik van bewijslabels om traceerbaarheid naar code te garanderen.

Deze elementen betreffen alleen de **vorm**, niet de **inhoud**. Alle inhoudelijke claims zijn afgeleid uit de broncode.
