# 🚀 Guida Introduttiva a roBERT

**Per chi non ha esperienza tecnica**

## Cos'è roBERT?

roBERT è un sistema intelligente che **legge descrizioni di prodotti edili** e **ne estrae automaticamente le informazioni importanti**.

### 🎯 Esempio Pratico

**Testo in ingresso:**
```
"Pavimento in gres porcellanato Florim, dimensioni 120x280 cm, spessore 6mm"
```

**Cosa fa roBERT:**
- ✅ Riconosce che si tratta di un pavimento in gres
- ✅ Identifica il marchio: Florim
- ✅ Estrae le dimensioni: 120cm × 280cm
- ✅ Trova lo spessore: 6mm
- ✅ Organizza tutto in modo strutturato

**Risultato:**
```json
{
  "categoria": "Pavimenti in gres porcellanato",
  "marchio": "Florim",
  "dimensioni": {
    "lunghezza": 1200,
    "larghezza": 2800,
    "unità": "mm"
  },
  "spessore": {
    "valore": 6,
    "unità": "mm"
  }
}
```

## 🤔 Perché è Utile?

### Problema Comune
Nelle aziende edili si hanno migliaia di descrizioni testuali di prodotti scritte in modo diverso:
- "Pavimento gres Florim 120x280"
- "Rivestimento in gres porcellanato, marca Florim, formato 120x280 cm"
- "Florim - gres 1.2m x 2.8m"

### Soluzione roBERT
roBERT **capisce** tutte queste varianti e le trasforma in dati strutturati uniformi, pronti per essere usati in database, preventivi, cataloghi.

## 🧠 Come Funziona (in Parole Semplici)

roBERT usa **3 componenti intelligenti** che lavorano insieme:

### 1️⃣ Classificatore (roBERTino)
**Cosa fa:** Capisce di che tipo di prodotto si tratta
- È un pavimento? Un sanitario? Un controsoffitto?
- Quale categoria specifica?

**Come funziona:** Ha "imparato" leggendo migliaia di descrizioni di prodotti edili italiani

### 2️⃣ Estrattore di Span
**Cosa fa:** Trova nel testo esattamente dove sono scritte le informazioni
- Dove c'è scritto il marchio?
- Dove sono le dimensioni?
- Dove si parla del materiale?

**Come funziona:** Usa intelligenza artificiale per comprendere il contesto (non cerca solo parole chiave!)

### 3️⃣ Parser Specializzati
**Cosa fa:** Interpreta correttamente i valori trovati
- "120x280 cm" → Lunghezza: 1200mm, Larghezza: 2800mm
- "6mm" → Spessore: 6 millimetri
- "RAL 9010" → Colore: Bianco

**Come funziona:** Usa regole specifiche per ogni tipo di dato

## 📊 Flusso Completo

```
┌─────────────────────────────────────────────┐
│ INPUT: Testo descrizione prodotto          │
│ "Pavimento gres Florim 120x280 cm"         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ PASSO 1: Classificazione                   │
│ → Categoria: "Pavimenti in gres"           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ PASSO 2: Estrazione Informazioni           │
│ → Marchio: "Florim" (posizione 15-21)      │
│ → Dimensioni: "120x280 cm" (pos 22-32)     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ PASSO 3: Interpretazione Valori            │
│ → Lunghezza: 1200mm                         │
│ → Larghezza: 2800mm                         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ OUTPUT: Dati Strutturati                    │
│ Pronti per database/preventivi              │
└─────────────────────────────────────────────┘
```

## 🎓 Vantaggi Rispetto ai Sistemi Tradizionali

### ❌ Sistema Tradizionale (Ricerca per Parole Chiave)
**Problema:** "compreso e compensato" → estrae "compensato" come materiale ❌

**Problema:** "Pavimento Florim, adesivo Mapei" → confonde i marchi ❌

### ✅ roBERT (Intelligenza Artificiale)
**Soluzione:** Capisce il contesto → "compensato" non è un materiale ✅

**Soluzione:** Distingue marchio principale (Florim) da accessori (Mapei) ✅

## 📚 Cosa Può Estrarre?

roBERT può identificare **20+ proprietà** diverse:

### Informazioni Base
- Marchio/Produttore
- Materiale
- Categoria prodotto

### Dimensioni
- Lunghezza, Larghezza, Altezza
- Spessore
- Diametro
- Formato

### Caratteristiche Tecniche
- Classe di resistenza al fuoco (REI)
- Trasmittanza termica (U)
- Isolamento acustico (dB)
- Portata (l/min)
- Colore (RAL)

### Normative
- Norme di riferimento (UNI EN, ISO)
- Certificazioni

## 🚦 Guida Rapida all'Uso

### Per Utenti Business (Senza Codice)

1. **Prepara i tuoi dati**
   - File Excel/CSV con descrizioni prodotti
   - Una descrizione per riga

2. **Converti in formato compatibile**
   ```bash
   robimb convert --input prodotti.csv --output prodotti.jsonl
   ```

3. **Estrai le informazioni**
   ```bash
   robimb extract properties --input prodotti.jsonl --output risultati.jsonl
   ```

4. **Ottieni i risultati**
   - File con tutte le informazioni estratte
   - Pronto per importazione in database/ERP

### Per Sviluppatori

Vedi [Installation Guide](installation.md) e [Workflows](workflows.md)

## 📖 Prossimi Passi

### Se sei un utente business:
1. [Installazione Base](installation.md) - Setup del sistema
2. [Workflow Comuni](workflows.md) - Operazioni tipiche
3. [FAQ](../guides/faq.md) - Domande frequenti

### Se sei uno sviluppatore:
1. [Architettura Tecnica](../architecture/technical.md) - Come funziona internamente
2. [Comandi CLI](../commands/overview.md) - Tutti i comandi disponibili
3. [Training Modelli](../models/training-roadmap.md) - Come addestrare i modelli

## ❓ Domande Comuni

### "Funziona solo in italiano?"
Sì, attualmente roBERT è ottimizzato per descrizioni in italiano del settore edile.

### "Quanto è preciso?"
- Classificazione categorie: ~92% accuratezza
- Estrazione proprietà: ~85-90% accuratezza
- Falsi positivi: <5%

### "Quanto è veloce?"
- Con GPU: ~150ms per prodotto
- Con CPU: ~600ms per prodotto
- 1000 prodotti: 2-10 minuti

### "Posso personalizzarlo per la mia azienda?"
Sì! roBERT può essere addestrato su dataset specifici della tua azienda per migliorare la precisione sui tuoi prodotti.

## 💡 Casi d'Uso Reali

### 1. Catalogazione Prodotti
**Scenario:** Hai 10.000 descrizioni da cataloghi PDF
**Soluzione:** roBERT estrae automaticamente tutte le schede prodotto

### 2. Preventivi Automatici
**Scenario:** Devi quotare rapidamente materiali
**Soluzione:** roBERT identifica proprietà → sistema calcola prezzo

### 3. Confronto Prodotti
**Scenario:** Confrontare caratteristiche di fornitori diversi
**Soluzione:** roBERT normalizza descrizioni diverse → confronto diretto

### 4. Integrazione ERP
**Scenario:** Importare dati da fornitori in formato testuale
**Soluzione:** roBERT converte testo → dati strutturati per ERP

## 🆘 Serve Aiuto?

- **Documentazione Completa:** [docs/README.md](../README.md)
- **Guide Tecniche:** [docs/guides/](../guides/)
- **Comandi Disponibili:** [docs/commands/](../commands/)
- **Esempi Pratici:** [examples/](../../examples/)

---

**Prossimo:** [Installazione](installation.md) | [Workflow Comuni](workflows.md)
