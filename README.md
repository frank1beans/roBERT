# roBERT Documentation Hub

roBERT è una toolkit Python per l'estrazione di proprietà e la classificazione di descrizioni BIM. Il progetto fornisce una CLI completa per preparare dataset, addestrare modelli, impacchettare risorse e orchestrare pipeline di inference con regole, QA encoder e LLM.

## Per iniziare rapidamente

1. **Clona il repository e installa le dipendenze**
   

2. **Verifica l'installazione**
   

3. **Configura i percorsi dei knowledge pack** (opzionale): le opzioni CLI usano valori predefiniti letti da . Imposta le variabili d'ambiente oppure modifica il file  se necessario.

## Indice della documentazione

| Argomento | Descrizione |
|-----------|-------------|
| **Panoramica CLI** | [Guida ai comandi principali di robimb](docs/commands/overview.md) |
| **Dataset e conversione** | [Convertire dataset e generare label map](docs/commands/convert.md) |
| **Estrazione proprietà** | [Pipeline  con tutte le sotto-funzioni](docs/commands/extract.md) |
| **Valutazione modelli** | [Comando ](docs/commands/evaluate.md) |
| **Knowledge pack** | [Comando ](docs/commands/pack.md) |
| **Sampling categorie** | [Comando ](docs/commands/sample-categories.md) |
| **Training avanzato** | [Script  (label e gerarchico)](docs/commands/train.md) |

## Risorse correlate

- : contiene server di esempio per l'integrazione LLM.
- : script di supporto per analisi ed automazioni.
- : suite di regression per assicurare qualità.

Per ulteriori dubbi o proposte apri una issue o unisci una pull request con i tuoi miglioramenti.
