# SoRa Task Brief v1.0

*(Executor: GPT-5.1-Codex-Mini — lokaal)*

## 0) Identiteit

* **Project**: SoRa S02 · Symphonia
* **Task ID**: `S02-____`
* **Datum**: `YYYY-MM-DD`
* **Executor**: Codex (executor-only)
* **Gatekeeper**: Sophia (OpenAI)
* **Waarheid / Acceptatie**: Ralph

---

## 1) Context (canoniek)

Korte beschrijving van **waar in de architectuur** deze taak zit.

* Architectuurlaag / module:
* Relevante versie(s) (bv. RealtimeStates v1.9 canonical):
* Gerelateerde canon (manifest / README / gist / afspraak):

> *Dit is geen uitleg voor beginners, maar positionering voor correct handelen.*

---

## 2) Doel (wat moet er na afloop waar zijn?)

Beschrijf **het gewenste eindgedrag**, niet de implementatie.

* [ ] Gedrag X is correct / stabiel / aantoonbaar
* [ ] Edge-case Y is afgevangen
* [ ] Geen regressie in Z

> *Als dit doel bereikt is, mag Codex stoppen.*

---

## 3) Scope (wat mag Codex aanraken?)

**Toegestaan:**

* Files:

  * `path/to/file_a.py`
  * `path/to/file_b.py`
* Modules / functies:

  * `function_x`
  * `ClassY.method_z`

**Niet toegestaan:**

* Alles buiten bovenstaande scope
* Canonieke bestanden / architectuur
* Structuur-, naam- of dependency-wijzigingen

---

## 4) Acceptatiecriteria (hard)

De taak is **alleen geslaagd** als:

* [ ] Criterium 1 (observeerbaar gedrag)
* [ ] Criterium 2 (meetbaar via test/log/bench)
* [ ] Criterium 3 (geen ongewenst neveneffect)

> *Geen interpretatie. Geen “lijkt goed”.*

---

## 5) Tests & verificatie

Codex moet (indien mogelijk) deze uitvoeren of expliciet aangeven waarom niet.

* Test / script:

  * `python3 script_x.py`
  * `pytest tests/test_y.py`
* Verwachte uitkomst:

  * geen errors
  * specifieke logregel / waarde

---

## 6) Verboden acties (herhaling = opzet)

Codex mag expliciet **niet**:

* refactoren buiten scope
* files hernoemen / verplaatsen
* formatting opschonen
* logging verwijderen
* dependencies wijzigen
* architectuur “verbeteren”

Bij twijfel: **STOP en escaleren**.

---

## 7) Oplevering (verplicht)

Codex levert:

1. **Samenvatting** (wat & waarom, max 10 regels)
2. **Overzicht gewijzigde files** + kernwijzigingen
3. **Test-uitvoer** (of reden waarom niet uitgevoerd)
4. **Risico’s / aandachtspunten**
5. **Checklist voor Ralph** (wat hij moet verifiëren)

---

## 8) Escalatievoorwaarden

Codex moet stoppen en terugkoppelen als:

* acceptatiecriteria conflicteren met canon
* meerdere interpretaties mogelijk zijn
* scope onvoldoende is voor correcte oplossing
* tests ontbreken of onduidelijk zijn

Dan: voorstel **Optie A / B / C** met impact.

---

## 9) Contract

Deze taak valt onder:
**SORA CodeX Contract v1.0**

Geen afwijkingen zonder expliciete toestemming.

---

### ✔️ Status

* ☐ Task opgesteld
* ☐ Gatekeeper akkoord
* ☐ Codex uitgevoerd
* ☐ Ralph geverifieerd
* ☐ Afgesloten

---

