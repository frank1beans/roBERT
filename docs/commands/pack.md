# robimb pack - Knowledge Pack

Crea knowledge pack per distribuzione in produzione.

## Sintassi

```bash
robimb pack [OPTIONS]
```

## Opzioni

- `--resources-dir PATH`: Directory con risorse (lexicon, schemi, registry)
- `--models-dir PATH`: Directory con modelli trained
- `--output PATH`: Path dell'archivio .tar.gz generato
- `--include-models`: Include anche i modelli (aumenta dimensione)

## Esempio

```bash
robimb pack \
  --resources-dir resources/data/properties \
  --output outputs/knowledge_pack_v1.tar.gz
```

## Contenuto Knowledge Pack

Un knowledge pack include:
- **Lexicon**: brands, materials, norms, colors
- **Schema**: definizioni proprietà per categoria
- **Registry**: mapping proprietà-categoria
- **Prompts**: template per LLM
- **Config**: configurazione produzione

Opzionalmente:
- **Models**: modelli QA fine-tuned

## Distribuzione

```bash
# Estrai knowledge pack in produzione
tar -xzf knowledge_pack_v1.tar.gz -C /opt/robimb/resources/
```

## Vedi Anche

- [Production Setup](../guides/production_resource_setup.md)
