# 🚀 Quick Start: GPT-4o-mini per Estrazione Proprietà

Guida rapida per usare GPT-4o-mini con roBERT in **5 minuti**.

## ⚡ Setup Veloce

### 1. Installa dipendenza OpenAI

```bash
pip install openai
```

### 2. Configura API Key

```bash
# Ottieni API key da: https://platform.openai.com/api-keys

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-proj-TUA-KEY-QUI"

# Linux/Mac/Git Bash
export OPENAI_API_KEY='sk-proj-TUA-KEY-QUI'
```

### 3. Avvia Server GPT-4o-mini

```bash
python examples/llm_server_gpt4mini.py --port 8000
```

Dovresti vedere:
```
✅ OPENAI_API_KEY configured
🚀 Starting GPT-4o-mini LLM Server on http://0.0.0.0:8000
```

### 4. Test (in altro terminale)

```bash
# Test connessione
curl http://localhost:8000/health

# Deve rispondere:
# {"status":"healthy","model":"gpt-4o-mini","api_key_configured":true}
```

### 5. Estrazione Proprietà

```bash
robimb extract properties \
  --input resources/data/train/classification/raw/dataset_lim.jsonl \
  --output outputs/extracted.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini \
  --no-qa \
  --sample 10
```

### 6. Analisi Risultati

```bash
python scripts/analysis/extraction_results.py outputs/extracted.jsonl
```

## 🎯 Fatto!

Hai appena estratto proprietà usando GPT-4o-mini!

---

## 📝 Configurazione Permanente (.env)

Per evitare di configurare `OPENAI_API_KEY` ogni volta:

### 1. Crea file .env

```bash
# Copia template
cp .env.example .env
```

### 2. Modifica .env

```bash
# Apri con editor
notepad .env  # Windows
nano .env     # Linux/Mac
```

Aggiungi:
```
OPENAI_API_KEY=sk-proj-TUA-KEY-QUI
```

### 3. Carica .env automaticamente

```bash
# Installa python-dotenv
pip install python-dotenv
```

Ora il server caricherà automaticamente `.env` all'avvio!

---

## 🔥 Workflow Produttivo

### Setup Iniziale (una volta)

```bash
# 1. Installa dipendenze
pip install openai python-dotenv

# 2. Configura .env
cp .env.example .env
# Modifica .env con la tua API key

# 3. Verifica
python examples/llm_server_gpt4mini.py --port 8000
# CTRL+C per fermare
```

### Utilizzo Quotidiano

```bash
# Terminal 1: Server (lascia sempre acceso)
python examples/llm_server_gpt4mini.py --port 8000

# Terminal 2: Estrazioni
robimb extract properties \
  --input mio_dataset.jsonl \
  --output outputs/estratto.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --sample 50
```

---

## 💡 Tips & Tricks

### Usa il Makefile

```bash
# Estrazione con shortcut
make extract-llm SAMPLE=20
```

### Aumenta il sample gradualmente

```bash
# Piccolo test
--sample 10    # ~$0.0006 (~60 cent/1000)

# Test medio
--sample 100   # ~$0.006

# Batch completo
--sample 1000  # ~$0.06
```

### Monitora costi

Dashboard: https://platform.openai.com/usage

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Verifica
echo $OPENAI_API_KEY  # deve mostrare sk-proj-...

# Se vuoto, riconfigura
export OPENAI_API_KEY='sk-proj-...'
```

### "Module 'openai' not found"
```bash
pip install openai
```

### Server non risponde
```bash
# Verifica se è in esecuzione
curl http://localhost:8000/health

# Riavvia se necessario
# CTRL+C nel terminal del server
python examples/llm_server_gpt4mini.py --port 8000
```

### Rate limit
```bash
# Riduci --sample o aspetta 1 minuto
```

---

## 📚 Prossimi Passi

1. **Leggi documentazione completa**: [examples/README.md](examples/README.md)
2. **Esplora comandi CLI**: [docs/commands/extract.md](docs/commands/extract.md)
3. **Comprendi architettura**: [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Sperimenta con diversi sample size**
5. **Confronta con rules-only**: `make extract-rules`

---

## 💰 Costi di Riferimento

| Sample Size | Costo Stimato |
|-------------|---------------|
| 10 docs | ~$0.0006 |
| 50 docs | ~$0.003 |
| 100 docs | ~$0.006 |
| 500 docs | ~$0.03 |
| 1000 docs | ~$0.06 |

**Nota**: Costi basati su GPT-4o-mini @ $0.15/1M input + $0.60/1M output tokens

---

**Buona estrazione! 🎉**

Per domande: vedi [examples/README.md](examples/README.md) o apri una issue.
