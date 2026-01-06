# Symphonia S02 — PC-side Bouwterrein (Python)

Dit repository is het **operationele bouwterrein** voor de PC-side
van Symphonia S02.

Het bevat:
- uitvoerende Python-modules (`sym_cycles`)
- live en replay tooling
- analyse- en visualisatiescripts
- een testbench-omgeving
- historische context en ontwerpnotities

Dit bouwterrein is **project-gebonden** en wordt aangestuurd
via de SoRa Gatekeeper.

## Normatieve documenten (verplicht)

Elke code agent die in dit bouwterrein opereert,
moet **alle onderstaande documenten lezen**
voordat uitvoering start.

Deze documenten bevatten:
- contractuele grenzen
- rolafspraken
- eerdere inzichten
- ontwerp- en testcontext

## Documentstatus binnen dit bouwterrein

De bestanden in `docs/` hebben verschillende rollen.

### Normatief (bindend voor agents)
- `docs/AGENTS.md`
  Rollen, verantwoordelijkheden en operationele regels
  voor agents binnen dit bouwterrein.

- `docs/Task Brief v1.0.md`
  Verplichte structuur voor elke uitvoeropdracht.

- Overige `.md` bestanden in `docs/`
  Historische inzichten, ontwerpbesluiten en testervaringen.
  Deze zijn **contextueel bindend** maar **niet herontwerpbaar**.

Deze documenten bepalen:
- wat agents mogen doen
- hoe taken worden uitgevoerd
- wat als correct of fout geldt

### Contextueel (informatief, niet-normatief)

- `docs/DEV_NOTES.md`
- `docs/GEMINI.md`
- overige `.md` bestanden in `docs/`

Deze documenten:
- geven achtergrond, historie en uitleg
- helpen begrip en oriëntatie
- mogen **niet** worden gebruikt als basis voor:
  - herontwerp
  - architectuurwijzigingen
  - aannames buiten de Task Brief

Bij conflict geldt altijd:
**Task Brief → AGENTS.md → Project README → Bouwterrein README**

## Wat dit README niet doet

Dit README:
- geeft **geen** run-instructies
- wijst **geen** specifieke scripts aan om te draaien
- vervangt **geen** Task Brief
- bevat **geen** bootstrap- of identiteit-logica

Welke tests, scripts of benches gebruikt worden,
wordt **altijd** bepaald door de Task Brief.

## Relatie tot Gatekeeper en Ralph

- Agents hebben **geen direct contact** met Ralph.
- Alle aansturing verloopt via de Gatekeeper.
- De Gatekeeper gebruikt dit README als
  feitelijke uitvoercontext bij verificatie.

Ralph gebruikt dit README uitsluitend om:
- globaal te begrijpen wat hier gebeurt
- inzicht te krijgen in stappen en samenhang
- zonder technische beoordeling of code-inzicht
