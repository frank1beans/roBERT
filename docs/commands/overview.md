# Panoramica CLI - robimb

`robimb` è la CLI principale del progetto roBERT per l'estrazione di proprietà e classificazione di prodotti edili.

## Comandi Disponibili

| Comando | Descrizione | Documentazione |
|---------|-------------|----------------|
| `robimb prepare` | Preparazione dataset per training | [prepare.md](prepare.md) |
| `robimb train` | Training modelli (span, price) | [train.md](train.md) |
| `robimb predict` | Inferenza modelli (categorie, proprietà, prezzi) | [predict.md](predict.md) |
| `robimb extract` | Estrazione proprietà da descrizioni | [extract.md](extract.md) |
| `robimb convert` | Conversione dataset e generazione label map | [convert.md](convert.md) |
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

1. **Preparazione dataset**: `robimb prepare`
2. **Training modelli**: `robimb train span` / `robimb train price`
3. **Inferenza**: `robimb predict category` / `robimb predict properties` / `robimb predict price`
4. **Valutazione risultati**: `robimb evaluate`
5. **Creazione knowledge pack**: `robimb pack`

## Risorse

- [Makefile](../../Makefile): automazione task comuni
- [scripts/](../../scripts/): script di supporto per analisi
- [examples/](../../examples/): esempi integrazione LLM
