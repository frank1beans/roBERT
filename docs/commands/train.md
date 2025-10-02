# robimb train - Training Modelli

Training di modelli per classificazione BIM (label e gerarchico).

## Sintassi

```bash
robimb train [SUBCOMMAND] [OPTIONS]
```

## Subcomandi

### Label Training
Training di classificatori flat (non gerarchici).

```bash
robimb train label \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --model-name atipiqal/BOB \
  --output-dir outputs/models/label
```

### Hierarchical Training
Training di classificatori gerarchici multi-livello.

```bash
robimb train hierarchical \
  --train-data data/train.jsonl \
  --eval-data data/eval.jsonl \
  --model-name atipiqal/BOB \
  --output-dir outputs/models/hierarchical
```

### TAPT (Task-Adaptive Pre-Training)
Pre-training MLM su dominio BIM.

```bash
robimb train tapt \
  --corpus data/bim_corpus.txt \
  --base-model bert-base-italian-cased \
  --output-dir outputs/models/tapt
```

## Opzioni Comuni

- `--train-data PATH`: Dataset di training
- `--eval-data PATH`: Dataset di validazione
- `--model-name NAME`: Modello base (HuggingFace)
- `--output-dir PATH`: Directory per salvare il modello
- `--batch-size INT`: Batch size (default: 16)
- `--epochs INT`: Numero di epoche (default: 3)
- `--learning-rate FLOAT`: Learning rate (default: 2e-5)

## Script Avanzati

Per training più complessi, usa gli script in [src/robimb/training/](../../src/robimb/training/):

```bash
# Label training avanzato
python -m robimb.training.label_trainer \
  --config configs/label_training.yaml

# Hierarchical training avanzato
python -m robimb.training.hier_trainer \
  --config configs/hier_training.yaml

# TAPT MLM
python -m robimb.training.tapt_mlm \
  --corpus data/corpus.txt \
  --output outputs/tapt_model
```

## Vedi Anche

- [convert.md](convert.md): Preparazione dataset
- [evaluate.md](evaluate.md): Valutazione modelli
