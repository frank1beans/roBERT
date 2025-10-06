# 📚 Documentazione roBERT

**Centro di documentazione completo per roBERT**
Toolkit intelligente per estrazione proprietà e classificazione prodotti edili

---

## 🎯 Per Chi Inizia

**Nuovo a roBERT?** Inizia da qui! 👇

### 📖 Guide Introduttive (Per Tutti)

<table>
<tr>
<td width="33%">

**🚀 [Introduzione](getting-started/README.md)**

Cos'è roBERT e come funziona
*(Non serve esperienza tecnica)*

</td>
<td width="33%">

**📦 [Installazione](getting-started/installation.md)**

Setup passo-passo del sistema
*(Windows, macOS, Linux)*

</td>
<td width="33%">

**🔄 [Workflow Comuni](getting-started/workflows.md)**

Operazioni tipiche e casi d'uso
*(Esempi pratici)*

</td>
</tr>
</table>

---

## 🏗️ Architettura e Progetto

**Per capire come funziona internamente**

### 📊 Panoramiche Sistema

| Documento | Descrizione | Livello |
|-----------|-------------|---------|
| [Overview Visuale](architecture/overview.md) | Diagrammi e flussi visivi | 🟢 Base |
| [Architettura Tecnica](architecture/technical.md) | Design e componenti | 🟡 Intermedio |
| [Pipeline Estrazione](architecture/pipeline.md) | Come funziona l'estrazione | 🟡 Intermedio |
| [Organizzazione Progetto](architecture/organization.md) | Struttura file e directory | 🟢 Base |

---

## 🤖 Modelli Machine Learning

**Per usare e addestrare i modelli**

### 🧠 Modelli Disponibili

<table>
<tr>
<td width="50%">

**Classificazione**
- [roBERTino](models/classification.md) - Classificatore categorie BIM
- [Training Roadmap](models/training-roadmap.md) - Piano training modelli

</td>
<td width="50%">

**Estrazione & Predizione**
- [Span Extractor](models/span-extraction.md) - Estrazione intelligente proprietà
- [Price Regressor](models/price-regression.md) - Predizione prezzi

</td>
</tr>
</table>

---

## ⚙️ Comandi CLI

**Riferimento completo comandi da terminale**

### 📋 Comandi Principali

| Comando | Funzione | Guida |
|---------|----------|-------|
| `robimb extract` | Estrazione proprietà | [extract.md](commands/extract.md) |
| `robimb predict` | Predizione prezzi/categorie | [predict.md](commands/predict.md) |
| `robimb train` | Training modelli ML | [train.md](commands/train.md) |
| `robimb prepare` | Preparazione dataset | [prepare.md](commands/prepare.md) |
| `robimb convert` | Conversione formati | [convert.md](commands/convert.md) |
| `robimb evaluate` | Valutazione performance | [evaluate.md](commands/evaluate.md) |
| `robimb pack` | Creazione knowledge pack | [pack.md](commands/pack.md) |
| `robimb config` | Configurazione sistema | [config.md](commands/config.md) |

📖 [Panoramica Completa](commands/overview.md) - Tutti i comandi disponibili

---

## 📚 Guide Avanzate

**Per sviluppatori e utenti avanzati**

### 🔧 Setup e Produzione

- [Production Setup](guides/production_resource_setup.md) - Configurazione ambiente produzione
- [Orchestration](guides/ORCHESTRATION_IMPLEMENTATION.md) - Domain heuristics e orchestrator
- [Orchestration Improvements](guides/orchestration_improvements.md) - Ottimizzazioni

### 💻 Sviluppo

- [Price Unit Implementation](development/implementation/price-unit.md) - Implementazione unit-aware pricing

---

## 🎓 Quick Reference

### Workflow Base

```bash
# 1. Converti dati
robimb convert --input prodotti.csv --output prodotti.jsonl

# 2. Estrai proprietà
robimb extract properties --input prodotti.jsonl --output risultati.jsonl

# 3. Predici prezzi (opzionale)
robimb predict price --model-dir outputs/price_model --input risultati.jsonl --output con_prezzi.jsonl
```

### Training Modelli

```bash
# Span Extractor
robimb train span --train-data qa_data.jsonl --output-dir outputs/span_model

# Price Regressor
robimb train price --train-data price_data.jsonl --output-dir outputs/price_model --use-properties
```

### Analisi Risultati

```bash
# Statistiche estrazione
python scripts/analysis/extraction_results.py risultati.jsonl

# Validazione dataset
python scripts/testing/sanity_check.py
```

---

## 📦 Risorse Progetto

### Lessici e Schemi

```
resources/data/properties/
├── lexicon/                # Dizionari dominio
│   ├── brands.json        # Marchi e produttori
│   ├── materials.json     # Materiali
│   ├── norms.json         # Normative tecniche
│   └── colors_ral.json    # Codici colore RAL
│
├── schema/                 # Schemi proprietà per categoria
│   ├── controsoffitti.json
│   ├── pavimentazioni.json
│   └── ...
│
└── registry.json          # Mapping categoria → proprietà
```

### Script Utili

```
scripts/
├── analysis/              # Analisi dataset e risultati
├── data_prep/             # Preparazione dati
├── testing/               # Test e validazione
└── setup/                 # Setup ambiente
```

---

## 🗺️ Mappa della Documentazione

### Per Tipo di Utente

**👤 Utente Business (Non Tecnico)**
1. [Introduzione](getting-started/README.md) - Cos'è roBERT
2. [Installazione](getting-started/installation.md) - Setup base
3. [Workflow Comuni](getting-started/workflows.md) - Come usarlo
4. [Comandi Base](commands/extract.md) - Estrazione proprietà

**💻 Sviluppatore**
1. [Architettura Tecnica](architecture/technical.md) - Design sistema
2. [Pipeline](architecture/pipeline.md) - Come funziona l'estrazione
3. [Comandi CLI](commands/overview.md) - Tutti i comandi
4. [Organizzazione](architecture/organization.md) - Struttura codice

**🧑‍🔬 Data Scientist / ML Engineer**
1. [Training Roadmap](models/training-roadmap.md) - Piano training
2. [Span Extractor](models/span-extraction.md) - Modello QA
3. [Price Regressor](models/price-regression.md) - Modello prezzi
4. [Comandi Train](commands/train.md) - Training modelli

**🏢 DevOps / Sysadmin**
1. [Installazione](getting-started/installation.md) - Setup completo
2. [Production Setup](guides/production_resource_setup.md) - Deploy produzione
3. [Config](commands/config.md) - Configurazione sistema

---

## 🔍 Cerca Nella Documentazione

### Per Argomento

**Estrazione Proprietà**
- [Workflow Estrazione](getting-started/workflows.md#1-estrazione-base)
- [Comando Extract](commands/extract.md)
- [Pipeline Architecture](architecture/pipeline.md)

**Training Modelli**
- [Training Roadmap](models/training-roadmap.md)
- [Comando Train](commands/train.md)
- [Span Extractor Training](models/span-extraction.md)

**Prezzi**
- [Price Regressor](models/price-regression.md)
- [Price Unit Implementation](development/implementation/price-unit.md)
- [Comando Predict](commands/predict.md)

**Setup & Config**
- [Installazione](getting-started/installation.md)
- [Production Setup](guides/production_resource_setup.md)
- [Comando Config](commands/config.md)

---

## 🆘 Supporto e Aiuto

### Risorse Community

- 🐛 **[GitHub Issues](https://github.com/atipiqal/roBERT/issues)** - Bug e feature request
- 💬 **[GitHub Discussions](https://github.com/atipiqal/roBERT/discussions)** - Domande e discussioni
- 📖 **[Examples](../examples/README.md)** - Esempi pratici

### FAQ Rapide

**Q: Come inizio?**
A: Leggi [Introduzione](getting-started/README.md) → [Installazione](getting-started/installation.md) → [Workflow](getting-started/workflows.md)

**Q: Dove trovo esempi pratici?**
A: [Workflow Comuni](getting-started/workflows.md) e [examples/](../examples/README.md)

**Q: Come addestrare un modello?**
A: [Training Roadmap](models/training-roadmap.md) e [Comando Train](commands/train.md)

**Q: Problemi installazione?**
A: Vedi [Troubleshooting](getting-started/installation.md#-risoluzione-problemi-comuni)

---

## 📂 Archivio

Documentazione storica e legacy in [archive/](archive/)

## 📝 Note sulla Riorganizzazione

Per dettagli completi sulla riorganizzazione della documentazione (versione 2.0), vedi [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)

---

## 🔗 Link Rapidi

| Risorsa | Link |
|---------|------|
| 📖 README Principale | [../README.md](../README.md) |
| 🚀 Quick Start | [getting-started/README.md](getting-started/README.md) |
| ⚙️ Comandi CLI | [commands/overview.md](commands/overview.md) |
| 🏗️ Architettura | [architecture/technical.md](architecture/technical.md) |
| 🤖 Modelli ML | [models/training-roadmap.md](models/training-roadmap.md) |
| 💼 Esempi | [../examples/README.md](../examples/README.md) |
| 🔧 Scripts | [../scripts/README.md](../scripts/README.md) |

---

**Ultimo aggiornamento:** 2025-10-06
**Versione:** 2.0 (Riorganizzazione completa)
