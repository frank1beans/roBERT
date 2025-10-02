# robimb config - Configurazione

Ispeziona e gestisce la configurazione di robimb.

## Sintassi

```bash
robimb config [SUBCOMMAND]
```

## Subcomandi

### Inspect
Mostra configurazione corrente.

```bash
robimb config inspect
```

Output:
```yaml
registry_path: resources/data/properties/registry.json
qa_model_dir: outputs/qa_models/base
llm_endpoint: null
llm_model: null
null_threshold: 0.25
```

### Set
Imposta valori di configurazione (via variabili d'ambiente).

```bash
export ROBIMB_REGISTRY_PATH=/custom/path/registry.json
export ROBIMB_QA_MODEL_DIR=/custom/qa_model
export ROBIMB_NULL_THRESHOLD=0.3
```

## File di Configurazione

Il sistema supporta file di configurazione TOML:

```toml
# resources/config/production.toml
[paths]
registry = "resources/data/properties/registry.json"
qa_model = "outputs/qa_models/base"

[extraction]
null_threshold = 0.25
llm_timeout = 30

[llm]
endpoint = "http://localhost:8000/extract"
model = "gpt-4o-mini"
```

Usa con:
```bash
robimb --config resources/config/production.toml extract properties ...
```

## Vedi Anche

- [Production Setup](../guides/production_resource_setup.md)
