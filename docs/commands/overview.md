# Panoramica CLI - robimb

`robimb` è la CLI principale del progetto roBERT per l'estrazione di proprietà e classificazione BIM.

## Comandi Disponibili

| Comando | Descrizione | Documentazione |
|---------|-------------|----------------|
| `robimb extract` | Estrazione proprietà da descrizioni BIM | [extract.md](extract.md) |
| `robimb convert` | Conversione dataset e generazione label map | [convert.md](convert.md) |
| `robimb train` | Training modelli (label e gerarchico) | [train.md](train.md) |
| `robimb evaluate` | Valutazione performance modelli | [evaluate.md](evaluate.md) |
| `robimb pack` | Creazione knowledge pack | [pack.md](pack.md) |
| `robimb config` | Ispezione configurazione | [config.md](config.md) |

## Installazione

```bash
pip install -e ".[dev]"
```

## Verifica Installazione

```bash
robimb --help
```

## Configurazione

Le opzioni CLI usano valori predefiniti configurabili tramite:
- Variabili d'ambiente
- File di configurazione (vedi `robimb config inspect`)

## Workflow Tipico

1. **Preparazione dataset**: `robimb convert`
2. **Training modello**: `robimb train`
3. **Creazione knowledge pack**: `robimb pack`
4. **Estrazione proprietà**: `robimb extract`
5. **Valutazione risultati**: `robimb evaluate`

## Risorse

- [Makefile](../../Makefile): automazione task comuni
- [scripts/](../../scripts/): script di supporto per analisi
- [examples/](../../examples/): esempi integrazione LLM
