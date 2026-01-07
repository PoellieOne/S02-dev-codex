## 📄 TASK BRIEF G2-A — FRAME A: LIVE OVERLAP CAPTURE (Sliding Window)

**Doel:** live vastleggen **welke signalen gelijktijdig of tijd-nabij actief zijn**, zonder duiding.

**Observatiekader:** **FRAME A** (sliding window, geen toekomstkennis)

**Parameters (expliciet):**

* Window: bijv. 250–500 ms (agent kiest één, vermeldt welke)
* Resolutie: event-level (bestaande events)
* Outputmarkering: `frame=A`

**Wat vastleggen (per overlap-event):**

* Betrokken **signaalnamen**
* Tijdstempel(s)
* Window-id of start/eind-tijd

**Verboden:**

* Geen labels, geen categorieën
* Geen samenvoeging met Frame B
* Geen conclusies

**Deliverables:**

* Ruwe overlap-events (log/CSV/JSONL)
* Reproduceerbare run-instructies
* Technische bijzonderheden (timing, buffers)

**Acceptatie:**

* Live runs leveren overlap-events of expliciet “geen overlap” (ook geldig)
* Data reproduceerbaar

---
