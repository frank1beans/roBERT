# Giorno 1 – Blocco percorsi e versioning del registry

Questa guida documenta le attività completate per il Giorno 1 della roadmap verso la messa in produzione del workflow di estrazione. L'obiettivo è rendere ripetibile la risoluzione dei percorsi critici (knowledge pack, registry, lexicon) e fissare la versione del registry da utilizzare in produzione.

## 1. Inventario centralizzato dei percorsi

Il modulo `robimb.config` espone la classe `ResourcePaths` che normalizza i percorsi partendo da:

- variabili d'ambiente (`ROBIMB_*`), oppure
- un file di configurazione esterno puntato da `ROBIMB_CONFIG_FILE`.

Per rendere trasparente questo passaggio è stato aggiunto il comando CLI:

```bash
poetry run robimb config paths
```

Il comando restituisce un JSON con tutti i percorsi risolti, indicando per ciascuno se esiste e il tipo di risorsa (file o directory). Esempio di output abbreviato:

```json
{
  "config_source": "environment",
  "paths": {
    "resources_dir": {"path": "/app/resources", "exists": true, "kind": "directory"},
    "registry_path": {"path": "/app/resources/data/properties/registry.json", "exists": true, "kind": "file"}
  },
  "registry": {
    "path": "/app/resources/data/properties/registry.json",
    "status": "ok",
    "detected_version": "0.2.0",
    "version_matches": true
  }
}
```

Per generare un lockfile consumabile dalla CI/CD è disponibile l'azione:

```bash
poetry run robimb config lockfile --output outputs/resource-paths.json
```

Il file prodotto (`outputs/resource-paths.json`) può essere archiviato come artefatto di build così da verificare in modo deterministico cosa verrà deployato.

## 2. Variabili d'ambiente supportate

| Variabile | Descrizione | Valore di default |
| --- | --- | --- |
| `ROBIMB_RESOURCES_DIR` | Directory base delle risorse condivise | `<project_root>/resources` |
| `ROBIMB_DATA_DIR` | Directory dati derivati (registry, prompts, lexicon) | `<resources_dir>/data` |
| `ROBIMB_PACK_DIR` | Cartella con il knowledge pack esportato | `<resources_dir>/pack` |
| `ROBIMB_REGISTRY_PATH` | File JSON del registry delle categorie | `<data_dir>/properties/registry.json` |
| `ROBIMB_LEXICON_DIR` | Radice delle tassonomie per estrazione | `<data_dir>/properties/lexicon` |
| `ROBIMB_PROMPTS_PATH` | File JSON con i prompt deterministici | `<data_dir>/properties/prompts.json` |
| `ROBIMB_BRANDS_PATH` | Lexicon marchi correnti | `<lexicon_dir>/brands.json` |
| `ROBIMB_BRANDS_LEGACY_PATH` | Lexicon marchi legacy | `<lexicon_dir>/brands.txt` |
| `ROBIMB_MATERIALS_PATH` | Lexicon materiali corrente | `<lexicon_dir>/materials.json` |
| `ROBIMB_MATERIALS_LEGACY_PATH` | Lexicon materiali legacy | `<lexicon_dir>/materials.txt` |
| `ROBIMB_STANDARDS_PATH` | Lexicon norme correnti | `<lexicon_dir>/norms.json` |
| `ROBIMB_STANDARDS_BY_CATEGORY` | Mappatura norme per categoria | `<lexicon_dir>/norms_by_category.json` |
| `ROBIMB_PRODUCERS_PATH` | Produttori per categoria | `<lexicon_dir>/producers_by_category.json` |
| `ROBIMB_COLORS_RAL_PATH` | Tavola colori RAL | `<lexicon_dir>/colors_ral.json` |
| `ROBIMB_STANDARDS_PREFIXES_PATH` | Prefissi norme | `<lexicon_dir>/standards_prefixes.json` |
| `ROBIMB_CONFIG_FILE` | File TOML/YAML alternativo con blocco percorsi | _(non definito)_ |

> Suggerimento: valorizzare esplicitamente queste variabili nell'orchestratore (es. Helm chart, GitHub Action) evita derive fra ambienti e consente di accorgersi tempestivamente di path errati.

## 3. Template di configurazione

Per ambienti in cui si preferisce versionare un file di configurazione è stato introdotto `resources/config/production.sample.toml`:

```toml
[paths]
resources = "../resources"
data = "../resources/data"
pack = "../resources/pack"
registry = "../resources/data/properties/registry.json"
lexicon = "../resources/data/properties/lexicon"
prompts = "../resources/data/properties/prompts.json"

[paths.lexicon]
brands = "../resources/data/properties/lexicon/brands.json"
brands_legacy = "../resources/data/properties/lexicon/brands.txt"
materials = "../resources/data/properties/lexicon/materials.json"
materials_legacy = "../resources/data/properties/lexicon/materials.txt"
standards = "../resources/data/properties/lexicon/norms.json"
standards_by_category = "../resources/data/properties/lexicon/norms_by_category.json"
producers = "../resources/data/properties/lexicon/producers_by_category.json"
colors_ral = "../resources/data/properties/lexicon/colors_ral.json"
standards_prefixes = "../resources/data/properties/lexicon/standards_prefixes.json"
```

Per utilizzarlo impostare `ROBIMB_CONFIG_FILE=/percorso/production.toml` prima di lanciare la pipeline.

## 4. Versione del registry fissata

L'inventario registra anche la versione del registry (`metadata.version`) assicurandosi che corrisponda alla baseline di produzione `0.2.0`. In caso di mismatch il comando termina segnalando `"version_matches": false`, permettendo di bloccare il deploy via job CI.

Il registry ufficiale rimane quello presente in `resources/data/properties/registry.json` (schema `property-registry-v1`). Qualsiasi modifica dovrà aggiornare la versione e passare per una nuova approvazione.

---

Con queste azioni il Giorno 1 è completo: i percorsi fondamentali sono inventariati e congelati, mentre la versione del registry è monitorata automaticamente durante i rilasci.
