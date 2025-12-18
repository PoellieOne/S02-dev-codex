> TEMPLATE ONLY — NOT CANON
> This document is provided solely as a structural and stylistic reference.
> Content MUST NOT be treated as factual source by Codex.

# S02.OriginTracker Architecture v0.4.x

## Functional & Technical Documentation

**Version:** v0.4.4 (ACCEPTED)  
**Date:** 2025-12-14  
**Series:** S02 — Symphonia Motor Awareness  
**Status:** Production-ready for Test-1 validation

---

## 1. Inleiding

### 1.1 Scope binnen Symphonia

OriginTracker is de **L1.5 Awareness Layer** binnen de Symphonia motor-awareness stack. Het vult de kloof tussen L1 tactiele sensing en L2 cycle-based motion tracking door **vroege bewegingsdetectie** mogelijk te maken.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYMPHONIA MOTOR AWARENESS STACK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  L0: Raw Events (Event24)                                                   │
│       ↓                                                                     │
│  L1: PhysicalActivity (tactile: activity, encoder_conf, gap detection)      │
│       ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L1.5: OriginTracker v0.4.x                                          │    │
│  │       • MDI (Micro-Displacement Integrator) — pre-cycle detection   │    │
│  │       • Two-Phase Origin (candidate + commit)                       │    │
│  │       • Awareness State Machine (STILL → MOVEMENT)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                     │
│  L2: CyclesState → TilesState → InertialCompass → MovementBody              │
│       ↓                                                                     │
│  L3: CompassSign v3 → Direction Kernel (CW/CCW/UNDECIDED)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Probleemstelling

De oorspronkelijke awareness stack had een fundamentele beperking:

> **Bewegingsdetectie was afhankelijk van cycles.**

Dit betekende dat bij kleine handverplaatsingen (20-30°) — waar wel events en pool-transities optreden maar nog geen volledige 3-punts cycles — het systeem "blind" bleef. Test-1 (eerste fysieke loskomen detecteren) faalde systematisch.

### 1.3 Oplossing: MDI + Two-Phase Origin

OriginTracker v0.4.x introduceert:

1. **MDI (Micro-Displacement Integrator)**: Detecteert netto verplaatsing op basis van pool-evidence, vóórdat cycles bestaan
2. **Two-Phase Origin**: Scheidt "begin van beweging" (candidate) van "bevestigde verplaatsing" (commit)
3. **PRE_MOVEMENT State**: Nieuwe awareness state die Test-1 laat slagen zonder cycle-afhankelijkheid

### MDI versie-opmerking
MDI is geïntroduceerd in v0.4.3 (pre-cycle micro-displacement) en gestabiliseerd in v0.4.4 door:
- NOISE/NO_DISP_ACTIVE niet langer PRE_MOVEMENT te laten blokkeren
- MDI-state te behouden bij NO_DISP_ACTIVE (reset_mdi=False)
- EMA smoothing voor confidence (mdi_conf_acc) als triggerbasis

---

## 2. Kernconcepten

### 2.1 Het Onderscheid: Voelen ≠ Verplaatsing

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          VOELEN vs VERPLAATSING                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   VOELEN (Tactiel)              VERPLAATSING (Kinematisch)                 │
│   ─────────────────             ──────────────────────────                 │
│   • Events ontvangen            • Netto positieverandering                 │
│   • Activiteit detecteren       • Richting bepalen                         │
│   • Scrape/touch herkennen      • Hoek accumuleren                         │
│                                                                            │
│   Indicator: activity_score     Indicator: mdi_micro_acc, disp_acc_deg     │
│   State: NOISE                  State: PRE_MOVEMENT, MOVEMENT              │
│                                                                            │
│   MDI vormt de BRUG: voelen → netto micro-displacement → verplaatsing      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Awareness State Machine v0.4.x

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AWARENESS STATES v0.4.x                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌──────────────────┐                           │
│     ┌────────┐   activity    │                  │   MDI trigger             │
│     │ STILL  │ ────────────→ │      NOISE       │ ──────────────┐           │
│     │   ○    │               │        ◎         │               │           │
│     └────────┘               └──────────────────┘               ↓           │
│         ↑                            │                   ┌──────────────┐   │
│         │                            │                   │ PRE_MOVEMENT │   │
│     gap timeout                      │ pool evidence     │      ◑       │   │
│         │                            ↓                   └──────────────┘   │
│         │                    ┌──────────────────┐               │           │
│         │                    │  PRE_ROTATION    │←──────────────┘           │
│         │                    │       ◐          │  pool candidate           │
│         │                    └──────────────────┘                           │
│         │                            │                                      │
│         │                            │ angle commit (30°)                   │
│         │                            ↓                                      │
│         │                    ┌──────────────────┐                           │
│         └────────────────────│    MOVEMENT      │                           │
│                              │       ◉          │                           │
│                              └──────────────────┘                           │
│                                                                             │
│  Legend:                                                                    │
│  ○ = geen activiteit    ◎ = voelen zonder verplaatsing                      │
│  ◑ = micro verplaatsing ◐ = origin kandidaat    ◉ = bevestigde beweging     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Timeline Priorities (origin_time0_s)

De "echte start" van beweging wordt bepaald door de vroegste betrouwbare indicator:

```
origin_time0_s = eerste van:
  1. micro_t0_s      ← MDI detecteert eerste netto verplaatsing (VROEGST)
  2. candidate_time  ← Pool evidence triggert candidate
  3. commit_time     ← Angle displacement bevestigt

Dit zorgt ervoor dat O(t0) altijd het "eerste loskomen" vastlegt,
niet het moment waarop cycles toevallig beschikbaar komen.
```

---

## 3. MDI — Micro-Displacement Integrator

### 3.1 Doel

MDI detecteert fysieke verplaatsing op basis van **pool-transities** en **sensor-alternatie**, zonder afhankelijkheid van cycles. Dit maakt Test-1 mogelijk.

### 3.2 Input

```
EVENT24 stream:
  • to_pool: int (0=NEU, 1=N, 2=S)
  • sensor: int (0=A, 1=B)
  • t_abs_us: timestamp
```

### 3.3 Sliding Window

```python
mdi_win_ms = 200.0  # korte window voor responsiviteit

# Ringbuffer entries: (t_s, sensor, to_pool)
mdi_window: deque = deque()
```

### 3.4 Micro Step Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MDI MICRO STEP LOGIC                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Pool Transition Detected?                                                 │
│          │                                                                  │
│          ↓                                                                  │
│   ┌──────────────────┐     YES     ┌──────────────────────┐                 │
│   │ prev_pool ≠ pool │ ──────────→ │ micro_step = +1.0    │                 │
│   └──────────────────┘             └──────────────────────┘                 │
│          │                                   │                              │
│          │                                   ↓                              │
│          │                         ┌──────────────────────┐                 │
│          │                         │ Flipflop Check       │                 │
│          │                         │ (p→q→p in 80ms?)     │                 │
│          │                         └──────────────────────┘                 │
│          │                                   │                              │
│          │                          YES      │      NO                      │
│          │                           ↓       │       ↓                      │
│          │                    ┌──────────┐   │  ┌─────────────┐             │
│          │                    │ step=-0.5│   │  │ step=+1.0   │             │
│          │                    │ trem+=0.15│  │  │ trem decay  │             │
│          │                    └──────────┘   │  └─────────────┘             │
│          │                                   │                              │
│          ↓                                   ↓                              │
│   mdi_micro_acc = clamp(mdi_micro_acc + step, 0..max)                       │
│   mdi_disp_micro_deg = mdi_micro_acc × micro_deg_per_step (10°)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Window Statistics

```python
def _compute_mdi_stats(now_s) -> Tuple[int, Set[int], float, float, float]:
    """
    Returns:
        mdi_pool_changes: aantal pool transities in window
        mdi_unique_pools: set van geziene pools {0,1,2}
        mdi_valid_rate: fractie valide pools (0/1/2) vs totaal
        mdi_sensor_alt_rate: fractie sensor wisselingen (A↔B)
        mdi_tremor_score: flipflop/tremor indicator
    """
```

### 3.6 Confidence Calculation

```python
mdi_conf = weighted_sum(
    0.30 × change_score,      # pool transities (0-4 → 0-1)
    0.20 × unique_score,      # unieke pools (0-3 → 0-1)
    0.20 × valid_score,       # valide pool rate
    0.20 × alt_score,         # sensor alternatie
   -0.30 × tremor_penalty     # flipflop straf
)
```

### 3.7 PRE_MOVEMENT Trigger

**Trigger gebruikt `mdi_conf_used`**
Waar `mdi_conf_used = mdi_conf_acc` (EMA), met fallback naar `mdi_conf` als `mdi_conf_acc` ontbreekt.

Doel: PRE_MOVEMENT mag niet falen door instant-window jitter; we willen stabiele “erkenning” op basis van een kortdurend geheugen.

```python
if mdi_disp_micro_deg >= mdi_trigger_micro_deg and mdi_conf_used >= mdi_conf_min and mdi_tremor_score <= mdi_tremor_max:
    aw_state = PRE_MOVEMENT
    aw_reason = MDI_TRIGGER
```

---

## 4. Two-Phase Origin

### 4.1 Rationale

Enkele events of pool-transities kunnen noise zijn. Daarom gebruikt OriginTracker een two-phase approach:

1. **CANDIDATE**: Pool evidence suggereert beweging
2. **COMMIT**: Angle displacement bevestigt beweging

### 4.2 Phase 1: Candidate (Pool Evidence)

```python
# Pool window (250ms sliding)
pool_evidence_strong = (
    pool_changes_win >= pool_changes_min      # default: 2
    and len(valid_pools) >= pool_unique_min   # default: 2
    and pool_valid_rate_win >= pool_valid_rate_min  # default: 0.70
)

if pool_evidence_strong and not origin_candidate_set:
    origin_candidate_set = True
    origin_candidate_time_s = now_s
    origin_time0_s = micro_t0_s or now_s  # prefer earlier MDI timestamp
    aw_state = PRE_ROTATION
    aw_reason = CANDIDATE_POOL
```

### 4.3 Phase 2: Commit (Angle Displacement)

```python
# Anti-rebound: angle moet stabiel zijn gedurende horizon
if abs(disp_acc_deg) >= origin_step_deg:  # default: 30°
    if commit_horizon_start_s is None:
        commit_horizon_start_s = now_s
    
    horizon_age = now_s - commit_horizon_start_s
    
    # Rebound check
    if abs(disp_acc_deg) < origin_rebound_eps_deg:  # default: 10°
        commit_horizon_start_s = None  # false start
        aw_reason = COMMIT_REBOUND
    
    # Commit after horizon
    elif horizon_age >= origin_commit_horizon_s:  # default: 0.35s
        origin_commit_set = True
        origin_time_s = now_s
        origin_theta_hat_rot = theta_hat_rot - (disp_acc_deg / 360)
        aw_reason = COMMIT_ANGLE
```

### 4.4 Commit Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMIT ANTI-REBOUND                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   disp_acc_deg                                                              │
│        │                                                                    │
│   35°  │           ╭───────────────────────╮                                │
│        │          ╱                         ╲     ← COMMIT na horizon       │
│   30°  │─────────╱───────────────────────────╲────────────────              │
│        │        ╱    origin_step_deg          ╲                             │
│   25°  │       ╱                               ╲                            │
│        │      ╱                                 ╲                           │
│   20°  │     ╱                                   ╲                          │
│        │    ╱                                     ╲                         │
│   10°  │───╱─────────────────────────────────────────────────               │
│        │  ╱      origin_rebound_eps_deg                                     │
│    0°  │ ╱                                                                  │
│        └───────────────────────────────────────────────────→ time           │
│             │←─────── horizon (350ms) ──────→│                              │
│                                                                             │
│   Scenario A: acc stijgt en blijft → COMMIT                                 │
│   Scenario B: acc stijgt maar daalt < 10° → REBOUND (geen commit)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Gap & Reset Rules

### 5.1 Coherent Reset (v0.3.1)

```python
# Hard reset: geen activiteit gedurende stop_gap
if (ageC_s >= stop_gap_s or ageE_s >= stop_gap_s) and activity_score < activity_reset_a0:
    reset_origin("STOP_GAP_TIMEOUT", keep_tactile=False)
    # → STILL

# Soft reset: cycles stale maar nog events
elif ageC_s >= noise_gap_s and activity_score >= activity_reset_a0:
    reset_origin("NO_DISP_ACTIVE", keep_tactile=True, reset_mdi=False)
    # → NOISE (keeps activity)

# Movement hold decay
elif origin_commit_set and movement_hold_s < ageC_s < stop_gap_s:
    aw_state = PRE_ROTATION
    aw_reason = HOLD_DECAY
    speed_deg_s *= 0.9
```

### 5.2 MDI-Specific Resets

```python
# Tremor override
if mdi_tremor_score > mdi_tremor_max and aw_state == PRE_MOVEMENT:
    reset_origin("MDI_TREMOR", keep_tactile=True)

# MDI hold timeout (geen candidate, geen activiteit)
if aw_state == PRE_MOVEMENT and not origin_candidate_set:
    if ageE_s > mdi_hold_s and activity_score < activity_threshold_low:
        reset_origin("MDI_HOLD_TIMEOUT", keep_tactile=False)
```

### 5.3 Reset policy nuance (v0.4.4)
Bij `NO_DISP_ACTIVE` wordt de origin/angle-context gewist, maar MDI blijft behouden:

- `_reset_origin("NO_DISP_ACTIVE", keep_tactile=True, reset_mdi=False)`

Rationale: MDI is pre-cycle proprioceptie; deze mag niet verdwijnen door een cycle-gap, anders kan Test-1 (PRE_MOVEMENT) in de praktijk niet betrouwbaar optreden.

---

## 6. Configuration

### 6.1 Default Parameters

```python
@dataclass
class L1Config:
    # === MDI v0.4.3 ===
    mdi_win_ms: float = 200.0           # sliding window
    mdi_trigger_micro_deg: float = 20.0 # PRE_MOVEMENT threshold
    mdi_conf_min: float = 0.35          # minimum confidence
    mdi_tremor_max: float = 0.60        # maximum tremor
    mdi_hold_s: float = 0.35            # hold duration
    micro_deg_per_step: float = 10.0    # step → degree mapping
    mdi_micro_acc_max: float = 36.0     # max accumulator (360°)
    mdi_flipflop_window_ms: float = 80.0
    
    # === Pool Window (v0.4) ===
    pool_win_ms: float = 250.0
    pool_changes_min: int = 2
    pool_unique_min: int = 2
    pool_valid_rate_min: float = 0.70
    
    # === Angle Commit (v0.4) ===
    origin_step_deg: float = 30.0
    origin_commit_horizon_s: float = 0.35
    origin_rebound_eps_deg: float = 10.0
    
    # === Gap Rules (v0.3.1) ===
    stop_gap_s: float = 0.80
    noise_gap_s: float = 0.50
    movement_hold_s: float = 0.25
    activity_reset_a0: float = 0.20
```

### 6.2 Presets

```python
L1_CONFIG_HUMAN = L1Config()  # standaard

L1_CONFIG_BENCH = L1Config(
    stop_gap_s=1.0,
    noise_gap_s=0.6,
    mdi_win_ms=250.0,
    mdi_trigger_micro_deg=15.0,
)

L1_CONFIG_SENSITIVE = L1Config(  # gevoelige detectie
    mdi_trigger_micro_deg=10.0,
    mdi_conf_min=0.25,
    origin_step_deg=15.0,
)
```

---

## 7. Data Model & Logging

### 7.1 L1Snapshot Fields

```python
@dataclass
class L1Snapshot:
    # === MDI v0.4.3 ===
    mdi_micro_acc: float           # accumulated micro steps
    mdi_disp_micro_deg: float      # micro_acc × 10°
    mdi_conf: float                # [0..1]
    mdi_tremor_score: float        # [0..1]
    mdi_pool_changes: int          # changes in window
    mdi_unique_pools: Set[int]     # {0,1,2}
    mdi_valid_rate: float          # valid pool fraction
    mdi_sensor_alt_rate: float     # A↔B alternation
    micro_t0_s: Optional[float]    # earliest movement
    micro_dir_hint: str            # CW/CCW/UNDECIDED
    
    # === Origin ===
    origin_candidate_set: bool
    origin_candidate_time_s: Optional[float]
    origin_commit_set: bool
    origin_time_s: Optional[float]      # commit time
    origin_time0_s: Optional[float]     # earliest (micro > cand > commit)
    
    # === Displacement ===
    disp_acc_deg: float                 # accumulated angle
    disp_from_origin_deg: float         # from committed origin
    speed_deg_s: float                  # EMA speed
    early_dir: str                      # early direction hint
    
    # === Awareness ===
    aw_state: AwState
    aw_reason: AwReason
```

### 7.2 JSONL Log Format (V4.3)

```json
{
  "t_s": 1.234,
  "ev": 5,
  "cy": 0,
  "ages": {"ageE_s": 0.05, "ageC_s": "INF"},
  "mdi": {
    "micro_acc": 2.5,
    "disp_deg": 25.0,
    "conf": 0.42,
    "tremor": 0.15,
    "pool_chg": 4,
    "unique": [0, 1, 2],
    "t0": 1.123,
    "dir_hint": "CW"
  },
  "candidate": {"set": false, "t0": null},
  "commit": {"set": false, "t0": null, "tC": null},
  "aw": {"state": "PRE_MOVEMENT", "reason": "MDI_TRIGGER"}
}
```

---

## 8. Acceptance Tests

### Test 0 — Touch/Scrape

**Input:** Korte aanrakingen zonder netto verplaatsing

**Expected:**
- `aw_state`: NOISE of STILL
- `mdi_micro_acc`: laag (< 2 steps)
- `mdi_tremor_score`: hoog (> 0.5)
- Geen PRE_MOVEMENT

**Verdict:** PASS als geen false positive PRE_MOVEMENT

### Test 1 — 20-30° Handverplaatsing (zonder rotaties)

**Input:** Langzame handbeweging, ~20-30° verplaatsing

**Expected:**
- `aw_state`: PRE_MOVEMENT
- `aw_reason`: MDI_TRIGGER
- `micro_t0_s`: gezet
- `cycles`: mag 0 zijn

**Verdict:** PASS als PRE_MOVEMENT minimaal één keer tijdens de run is opgetreden (state “seen-at-least-once”), ongeacht de eindstaat van de run (die mag terugvallen naar STILL door STOP_GAP_TIMEOUT).

### Test 1b — Doorbewegen na PRE_MOVEMENT

**Input:** Test 1 gevolgd door verdere rotatie

**Expected:**
- PRE_MOVEMENT → PRE_ROTATION → MOVEMENT
- `origin_time0_s` = `micro_t0_s` (vroegste start)
- Commit delay zichtbaar

**Verdict:** PASS als origin_time0_s behouden blijft

### Test 2 — Stop na Beweging

**Input:** Beweging gevolgd door pauze

**Expected:**
- Gap rules resetten coherent
- PRE_MOVEMENT houdt max `mdi_hold_s` aan
- Val terug naar STILL na timeout

**Verdict:** PASS als geen "stale movement" zonder freshness

---

## 9. Tooling

### 9.1 Live Dashboard

```bash
python3 live_symphonia_v2_0.py --scoreboard --log
```

**Output:**
```
PRE_MOVEMENT MDI_TRIGGER              cand=- comm=- μ=  25° conf=0.42 t0μ=1.23 cy=0
  MDI: chg=4 trem=0.15 | CB: ev=45 cy=0 last=SEQ_NOT_MATCH top=SEQ:30
```

### 9.2 Log Analyzer (V4.3)

```bash
python3 analyze_handencoder_log.py live_origin_*.jsonl
```

**Output:**
```
======================================================================
  V4.3 + MDI LOG ANALYSIS: live_origin_20251214_1234.jsonl
======================================================================

  Records: 1500  Duration: 15.2s
  Events: 250  Cycles: 0

----------------------------------------------------------------------
  MDI (MICRO-DISPLACEMENT INTEGRATOR)
  Max micro displacement: 35.0°
  Avg MDI confidence: 0.48

----------------------------------------------------------------------
  TIMELINE
  First micro_t0:      1.23s
  First PRE_MOVEMENT:  1.25s
  First CANDIDATE:     -
  First COMMIT:        -

----------------------------------------------------------------------
  DIAGNOSIS
======================================================================
  ✅ Test-1 PASS: PRE_MOVEMENT detected
  ⚠️  No cycles detected
     → Pools valid but 0 cycles — CycleBuilder issue
======================================================================
```

---

## 10. Integration met Bestaande Stack

### 10.1 Relatie tot CompassSign v3

OriginTracker (L1.5) en CompassSign v3 (L3) zijn complementair:

| Aspect | OriginTracker | CompassSign v3 |
|--------|---------------|----------------|
| Layer | L1.5 | L3 |
| Input | Events, pools | PhaseTiles v3 |
| Output | Awareness state, origin | Global direction |
| Cycles nodig | Nee (MDI) | Ja |
| Doel | Beweging detecteren | Richting bepalen |

### 10.2 Relatie tot Cycle-Backbone

De Cycle-Backbone (§4 van `symphonia_cycle_backbone_blueprint_v_1_0.md`) blijft de fysieke waarheid. OriginTracker voegt **pre-cycle awareness** toe:

```
Pre-cycle (OriginTracker):
  Events → MDI → PRE_MOVEMENT → origin_time0_s

Cycle-based (Backbone):
  Cycles → Tiles → CompassSign → direction
  
OriginTracker VERVANGT cycles niet, maar DETECTEERT eerder.
```

### 10.3 Dataflow

```
EVENT24
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ↓                                          ↓
L1PhysicalActivity                      RealtimePipeline (L2)
    │                                          │
    ├─ record_pool() ──────→ MDI               ├─ CyclesState
    │                        │                 ├─ TilesState
    ├─ update() ────────────→ OriginTracker    ├─ InertialCompass
    │                        │                 └─ MovementBody
    │                        ↓                         │
    │                   L1Snapshot                     ↓
    │                   (aw_state,                PipelineResult
    │                    micro_t0_s,              (tiles, compass,
    │                    origin_*)                 movement_state)
    │                        │                         │
    └────────────────────────┴─────────────────────────┘
                             │
                             ↓
                    Dashboard / Logger
```

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.3.1 | 2025-12 | Timer semantics (ageC/ageE), pool histogram, V3 log format |
| v0.4 | 2025-12 | Two-phase origin (candidate + commit), pool evidence window |
| v0.4.1 | 2025-12 | CycleBuilder TruthProbe (reject telemetry) |
| v0.4.2 | 2025-12 | canon_event24() normalization, trace buffer |
| v0.4.3 | 2025-12 | **MDI** (Micro-Displacement Integrator), PRE_MOVEMENT state |
| v0.4.4 | 2025-12 | NOISE→PRE_MOVEMENT fix, status ACCEPTED |

---

## 12. Open Punten (v0.4.5+)

1. **PRE_MOVEMENT hysteresis**: Stabiliteit bij korte pauzes
2. **micro direction hint**: Vroege richting stabiliseren
3. **Origin commit timing**: Verfijning op basis van micro_t0 + acc + anti-rebound
4. **L2 integration**: Feedback loop van cycles naar MDI confidence

---

## 13. Conclusie

OriginTracker v0.4.x vormt de **awareness backbone** van Symphonia. Door MDI toe te voegen is bewegingsdetectie niet langer afhankelijk van cycles, waardoor Test-1 (eerste fysieke loskomen) nu slaagt.

Het systeem respecteert het onderscheid tussen **voelen** (tactiel, events) en **verplaatsing** (kinematisch, netto beweging), met MDI als de cruciale brug ertussen.

De two-phase origin (candidate + commit) zorgt voor robuuste bevestiging, terwijl de gap rules coherente resets garanderen.

**Status: Production-ready voor Test-1 validatie.**

---

*Dit document is de canonieke referentie voor L1.5 OriginTracker binnen de Symphonia motor-awareness stack.*
