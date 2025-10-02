# Examples

Esempi e server di integrazione per roBERT.

## 📋 Indice

1. [LLM Server Mock](#llm-server-mock) - Server mock per testing
2. [LLM Server GPT-4o-mini](#llm-server-gpt-4o-mini) - Server con OpenAI GPT-4o-mini
3. [Setup e Configurazione](#setup-e-configurazione)
4. [Integrazione con roBERT](#integrazione-con-robert)

---

## LLM Server Mock

### llm_server_example.py

Server FastAPI mock che simula un endpoint LLM per testing senza costi API.

#### Avvio

```bash
# Avvio standard
python examples/llm_server_example.py --port 8000

# Con auto-reload (sviluppo)
python examples/llm_server_example.py --port 8000 --reload

# Via Makefile
make llm-server LLM_PROVIDER=mock LLM_PORT=8000
```

#### Test

```bash
# Health check
curl http://localhost:8000/health

# Test estrazione
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Testo:\nPiastrella gres porcellanato 60x60 sp. 10mm\nDomanda:\nEstrai dimensioni\nSchema:\n{}",
    "schema": {},
    "model": "mock"
  }'
```

#### Risposta Esempio

```json
{
  "value": "60x60",
  "confidence": 0.88,
  "source": "llm_mock"
}
```

---

## LLM Server GPT-4o-mini

### llm_server_gpt4mini.py

Server FastAPI che usa **OpenAI GPT-4o-mini** per estrazione reale tramite API.

#### Prerequisiti

1. **API Key OpenAI**
   - Ottieni da: https://platform.openai.com/api-keys
   - Costo: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens

2. **Installazione openai**
   ```bash
   pip install openai
   ```

#### Configurazione

**Metodo 1: File .env (raccomandato)**

```bash
# Copia il template
cp .env.example .env

# Modifica .env e aggiungi la tua API key
OPENAI_API_KEY=sk-proj-...
```

**Metodo 2: Variabile d'ambiente**

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-proj-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-proj-...

# Linux/Mac
export OPENAI_API_KEY='sk-proj-...'
```

#### Avvio

```bash
# Avvio standard
python examples/llm_server_gpt4mini.py --port 8000

# Con auto-reload (sviluppo)
python examples/llm_server_gpt4mini.py --port 8000 --reload
```

#### Test

```bash
# Health check (verifica API key)
curl http://localhost:8000/health

# Test estrazione reale
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Testo:\nPiastrella gres porcellanato effetto legno 120x20 cm spessore 9mm classe PEI 4\nDomanda:\nEstrai il materiale\nSchema:\n{\"type\": \"string\"}",
    "schema": {"type": "string"},
    "model": "gpt-4o-mini"
  }'
```

#### Risposta Esempio

```json
{
  "value": "gres porcellanato",
  "confidence": 0.95,
  "source": "gpt-4o-mini",
  "raw_response": "{\"value\": \"gres porcellanato\", \"confidence\": 0.95}"
}
```

---

## Setup e Configurazione

### 1. Installazione Dipendenze

```bash
# Core dependencies (già installate con robimb)
pip install fastapi uvicorn pydantic

# Per GPT-4o-mini
pip install openai

# Opzionale: per gestione .env
pip install python-dotenv
```

### 2. File .env

Crea un file `.env` nella root del progetto:

```bash
# .env
OPENAI_API_KEY=sk-proj-your-key-here
LLM_ENDPOINT=http://localhost:8000/extract
LLM_MODEL=gpt-4o-mini
```

**Importante**: `.env` è già in `.gitignore`, non verrà committato.

### 3. Verifica Setup

```bash
# 1. Verifica API key
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY% # Windows CMD
echo $env:OPENAI_API_KEY # Windows PowerShell

# 2. Avvia server
python examples/llm_server_gpt4mini.py

# 3. In altro terminale, testa
curl http://localhost:8000/health
```

---

## Integrazione con roBERT

### Workflow Completo

#### 1. Avvia Server LLM

```bash
# Terminal 1: Avvia server GPT-4o-mini
python examples/llm_server_gpt4mini.py --port 8000
```

#### 2. Verifica Connessione

```bash
# Terminal 2: Test server
make llm-test
```

#### 3. Estrazione con LLM

```bash
# Solo LLM (senza QA encoder)
robimb extract properties \
  --input resources/data/train/classification/raw/dataset_lim.jsonl \
  --output outputs/extracted_gpt4mini.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --llm-model gpt-4o-mini \
  --no-qa \
  --sample 10

# Pipeline completa (rules + LLM)
make extract-llm SAMPLE=20 LLM_MODEL=gpt-4o-mini
```

#### 4. Analisi Risultati

```bash
python scripts/analysis/extraction_results.py outputs/extracted_gpt4mini.jsonl
```

### Confronto Mock vs GPT-4o-mini

```bash
# Terminal 1: Avvia mock
python examples/llm_server_example.py --port 8001

# Terminal 2: Avvia GPT-4o-mini
python examples/llm_server_gpt4mini.py --port 8000

# Terminal 3: Confronta
# Mock
robimb extract properties \
  --input dataset.jsonl \
  --output outputs/mock.jsonl \
  --llm-endpoint http://localhost:8001/extract \
  --sample 50 --no-qa

# GPT-4o-mini
robimb extract properties \
  --input dataset.jsonl \
  --output outputs/gpt4mini.jsonl \
  --llm-endpoint http://localhost:8000/extract \
  --sample 50 --no-qa

# Analizza differenze
python scripts/analysis/extraction_results.py outputs/mock.jsonl
python scripts/analysis/extraction_results.py outputs/gpt4mini.jsonl
```

---

## 💰 Costi Stimati GPT-4o-mini

| Operazione | Token Input | Token Output | Costo | Totale |
|------------|-------------|--------------|-------|--------|
| 1 estrazione | ~200 | ~50 | $0.00003 + $0.00003 | **~$0.00006** |
| 100 estrazioni | ~20K | ~5K | $0.003 + $0.003 | **~$0.006** |
| 1000 estrazioni | ~200K | ~50K | $0.03 + $0.03 | **~$0.06** |

**Conclusione**: Molto economico per testing e sviluppo! 🎉

---

## 🔒 Sicurezza

### Best Practices

1. **MAI committare .env**
   - Già in `.gitignore`
   - Usa `.env.example` per template

2. **API Key Rotation**
   ```bash
   # Revoca vecchia key su: https://platform.openai.com/api-keys
   # Genera nuova key
   # Aggiorna .env
   ```

3. **Rate Limiting**
   - GPT-4o-mini ha limiti per tier
   - Tier 1: ~500 req/min
   - Usa `--sample` per limitare richieste

4. **Monitoring**
   - Dashboard OpenAI: https://platform.openai.com/usage
   - Controlla costi in tempo reale

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"

```bash
# Verifica
echo $OPENAI_API_KEY

# Se vuoto, configura
export OPENAI_API_KEY='sk-proj-...'

# O usa .env
cp .env.example .env
# Modifica .env con la tua key
```

### "Module 'openai' not found"

```bash
pip install openai
```

### Server non risponde

```bash
# Verifica se è in esecuzione
curl http://localhost:8000/health

# Verifica porta libera
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows
```

### Rate limit exceeded

```bash
# Riduci sample
robimb extract properties ... --sample 10

# Aumenta timeout
robimb extract properties ... --llm-timeout 60
```

---

## 📚 Vedi Anche

- [docs/commands/extract.md](../docs/commands/extract.md) - Documentazione comando extract
- [docs/guides/production_resource_setup.md](../docs/guides/production_resource_setup.md) - Setup produzione
- [Makefile](../Makefile) - Task automatizzati
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Architettura sistema
