# 📦 Guida all'Installazione

Guida passo-passo per installare roBERT sul tuo computer.

## 🎯 Cosa Ti Serve

### Requisiti Base
- **Sistema Operativo:** Windows 10/11, macOS, o Linux
- **Python:** Versione 3.8 o superiore
- **Spazio Disco:** Almeno 5GB liberi
- **RAM:** Minimo 8GB (consigliato 16GB)

### Requisiti Opzionali (per performance migliori)
- **GPU NVIDIA:** Con almeno 8GB VRAM (per elaborazione veloce)
- **Internet:** Per scaricare modelli pre-addestrati

## 🚀 Installazione Rapida (Utenti Business)

### Windows

1. **Installa Python**
   - Scarica da [python.org](https://www.python.org/downloads/)
   - Durante installazione, spunta "Add Python to PATH"

2. **Scarica roBERT**
   ```cmd
   cd Desktop
   git clone https://github.com/atipiqal/roBERT.git
   cd roBERT
   ```

3. **Installa dipendenze**
   ```cmd
   pip install -e .
   ```

4. **Verifica installazione**
   ```cmd
   robimb --help
   ```

   Se vedi la lista dei comandi, tutto funziona! ✅

### macOS / Linux

1. **Verifica Python**
   ```bash
   python3 --version
   ```
   Se vedi "Python 3.8" o superiore, sei a posto!

2. **Scarica roBERT**
   ```bash
   cd ~/Desktop
   git clone https://github.com/atipiqal/roBERT.git
   cd roBERT
   ```

3. **Installa dipendenze**
   ```bash
   pip3 install -e .
   ```

4. **Verifica installazione**
   ```bash
   robimb --help
   ```

## 🔧 Installazione Completa (Sviluppatori)

### 1. Clona Repository

```bash
git clone https://github.com/atipiqal/roBERT.git
cd roBERT
```

### 2. Crea Ambiente Virtuale (Consigliato)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installa Dipendenze

**Installazione Base:**
```bash
pip install -e .
```

**Installazione Completa (con strumenti sviluppo):**
```bash
pip install -e ".[dev]"
```

### 4. Configura Token HuggingFace

Alcuni modelli richiedono autenticazione HuggingFace:

1. Crea account su [huggingface.co](https://huggingface.co)
2. Ottieni token da [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Crea file `.env` nella cartella roBERT:
   ```bash
   HF_TOKEN=hf_tuoTokenQui
   ```

### 5. Scarica Modelli Pre-addestrati

**Classificatore (roBERTino):**
```bash
# Download automatico al primo utilizzo
robimb extract properties --help
```

**Span Extractor (se disponibile):**
```bash
# Verifica se esiste il modello
ls outputs/span_model/
```

## 🧪 Test Installazione

### Test Base

```bash
# Verifica comandi disponibili
robimb --help

# Test configurazione
robimb config show
```

### Test Estrazione (con file esempio)

```bash
# Crea file di test
echo '{"text": "Pavimento gres Florim 120x280 cm"}' > test.jsonl

# Esegui estrazione
robimb extract properties --input test.jsonl --output result.jsonl

# Verifica risultato
cat result.jsonl
```

Se vedi proprietà estratte, tutto funziona! ✅

## 🐛 Risoluzione Problemi Comuni

### Errore: "robimb: command not found"

**Causa:** Python non è nel PATH o installazione non riuscita

**Soluzione:**
```bash
# Usa percorso completo
python -m robimb.cli.main --help

# Oppure reinstalla con:
pip install -e . --force-reinstall
```

### Errore: "No module named 'transformers'"

**Causa:** Dipendenze non installate

**Soluzione:**
```bash
pip install transformers torch
```

### Errore: "CUDA out of memory"

**Causa:** GPU con memoria insufficiente

**Soluzione:** Usa CPU invece di GPU
```bash
# Forza uso CPU
export CUDA_VISIBLE_DEVICES=""
robimb extract properties --input data.jsonl --output result.jsonl
```

### Errore: "HuggingFace authentication required"

**Causa:** Token HuggingFace mancante

**Soluzione:**
1. Crea file `.env` con `HF_TOKEN=hf_tuoToken`
2. Oppure usa variabile ambiente:
   ```bash
   export HF_TOKEN=hf_tuoToken
   ```

### Installazione Lenta

**Causa:** Download modelli grandi

**Soluzione:** È normale! I modelli sono ~500MB-2GB. Aspetta il completamento.

## 📊 Verifica Requisiti Hardware

### Test GPU (opzionale)

```python
import torch
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Test RAM

```bash
# Windows
systeminfo | find "Available Physical Memory"

# macOS/Linux
free -h
```

Assicurati di avere almeno 8GB RAM disponibile.

## 🔄 Aggiornamento

### Aggiorna roBERT all'ultima versione

```bash
cd roBERT
git pull origin main
pip install -e . --upgrade
```

### Aggiorna solo dipendenze

```bash
pip install --upgrade transformers torch
```

## 🎓 Ambienti di Installazione

### Solo Utilizzo (Produzione)

```bash
pip install -e .
```

**Include:**
- Modelli ML (transformers, torch)
- Utilities estrazione
- CLI comandi base

### Sviluppo Completo

```bash
pip install -e ".[dev]"
```

**Include tutto + strumenti sviluppo:**
- pytest (testing)
- black, flake8 (linting)
- jupyter (notebook)
- tqdm (progress bars)

### Training Modelli

```bash
pip install -e ".[train]"
```

**Include tutto + librerie training:**
- wandb (tracking esperimenti)
- tensorboard (visualizzazione)
- accelerate (multi-GPU)

## 📁 Struttura Post-Installazione

Dopo installazione completa:

```
roBERT/
├── src/robimb/          # Codice sorgente ✅
├── resources/           # Lessici, schemi ✅
├── outputs/             # Modelli trained (da scaricare)
├── venv/                # Ambiente virtuale (se creato)
├── .env                 # Token HuggingFace (da creare)
└── test.jsonl           # File test (opzionale)
```

## ✅ Checklist Post-Installazione

- [ ] Python 3.8+ installato
- [ ] Repository clonato
- [ ] Ambiente virtuale attivato (opzionale ma consigliato)
- [ ] Dipendenze installate (`pip install -e .`)
- [ ] Comando `robimb --help` funziona
- [ ] File `.env` creato con token HuggingFace
- [ ] Test estrazione eseguito con successo
- [ ] GPU riconosciuta (se disponibile)

## 🎯 Prossimi Passi

Ora che hai installato roBERT, puoi:

1. **[Primi Passi](workflows.md)** - Inizia a usare roBERT
2. **[Comandi CLI](../commands/overview.md)** - Esplora tutti i comandi
3. **[Esempi Pratici](../../examples/README.md)** - Vedi casi d'uso reali

## 🆘 Ancora Problemi?

- **Documentazione Completa:** [docs/README.md](../README.md)
- **FAQ:** [docs/guides/faq.md](../guides/faq.md)
- **GitHub Issues:** [Apri un issue](https://github.com/atipiqal/roBERT/issues)

---

**Precedente:** [Introduzione](README.md) | **Prossimo:** [Primi Passi](workflows.md)
