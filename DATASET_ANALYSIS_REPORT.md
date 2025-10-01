# Dataset Analysis Report: dataset_lim.jsonl

**Analysis Date:** 2025-10-01
**Dataset Location:** `c:\Users\f.biggi\Scripts\roBERT\data\train\classification\raw\dataset_lim.jsonl`
**Total Records:** 3,186
**Sample Size Analyzed:** ~200-300 records (systematic sampling)

---

## 1. BRANDS AND MANUFACTURERS (Marche)

### Most Common Brands Found (in order of frequency):

1. **Building Materials Brands:**
   - Knauf (GKB, GKI, Idrolastra) - Very frequent
   - Sch�ck - Common in construction
   - Bassanetti

2. **Sanitaryware & Fixtures:**
   - Ideal Standard - 6 occurrences
   - Grohe - Present
   - Hansgrohe - Present
   - Geberit - Present
   - Duravit - Present (Stark3 model)
   - Globo - Present (Tuttofare, Le Lastre T-Edge models)

3. **Ceramics & Tiles:**
   - Marazzi - 6 occurrences (Cementum, Room, Confetto collections)
   - Emil Group/Emil Ceramica - Present (Pigmento, Medley collections)
   - Florim - 4 occurrences (Match Up model)
   - Living Ceramics - Present (Cuit collection)
   - EcoContract

4. **Doors & Hardware:**
   - Novoferm - 4+ occurrences (Elite Premio, EPN models)
   - Yale - 5 occurrences (fire-resistant models)
   - Iseo - 4 occurrences
   - Bertolotto - 2 occurrences (Effetto Legno)
   - DormaKaba - 2 occurrences (KTV Atrium Flex)

5. **Specialized Equipment:**
   - Dyson - 2 occurrences (Airblade V)
   - DMP Electronics - Present (Tube Soap)
   - Goman - Present (accessibility products)
   - Fantoni - 3 occurrences (4Akustik model)
   - Mapei - Present (Keraflex, Ultracolor Plus)

6. **Screening/Partitions:**
   - Metalscreen - 4+ occurrences (GR, LD models)
   - Profilsystem/ProfilSystem

### Brand Name Format Variations:
- All uppercase: GROHE, GOMAN, FANTONI
- Capitalized: Grohe, Geberit, Marazzi
- Mixed case: Hansgrohe, EcoContract
- Often preceded by: "tipo", "marca", "modello"

---

## 2. DIMENSION PATTERNS (Dimensioni)

### Common Dimension Formats:

#### 2D Dimensions (Width x Height/Length):
```
60x60 cm        - Most common (tiles, panels)
80x210 cm       - Very common (doors)
90x210 cm       - Common (doors)
60x120 cm       - Common (tiles)
20x20 cm        - Present
25x25 cm        - Present
30x40 cm        - Present
40x40 cm        - Present
100x30 cm       - Present
```

#### 3D Dimensions (W x H x D):
```
50x150x50 mm
40x150x40 mm
28x18x1 cm
```

#### Format Variations Found:
- With units: `60x60 cm`, `25x25cm`, `100 x 30 mm`
- Without units: `60x60`, `80x210`, `90x240`
- With spaces: `60 x 60`, `500 x 500`
- With multiplication signs: `20×20`, `60X60`
- Prefixed: `dim. 60x60 cm`, `dimensioni 100x30 cm`, `delle dimensioni 60x60 cm`

#### Diameter Formats:
```
Ø 25 cm
Ø25
diametro 30mm
diametro 30 mm
Ø 16
```

#### Length Formats:
```
lunghezza 60 cm
L=150
L 150
L. 800mm
H 210 cm
H. 120 cm
```

#### Thickness/Spessore (Most Common):
```
spessore 12,5 mm    - Most frequent (plasterboard)
spessore 10 mm
spessore 9,5 mm
spessore 40 mm
spessore 60 mm
sp. 9 mm
sp. 6 mm
```

### Dimension Ranges Commonly Specified:
- "dimensioni non superiori a 150 x 65 cm"
- "fino a 40x20 cm"
- "oltre 40x20 cm e fino a 150x60 cm"

---

## 3. STANDARDS AND NORMS (Norme)

### Most Frequent Standards:

#### General Certification (Nearly Universal):
```
CE                          - 121 occurrences
DOP                         - 107 occurrences
Regolamento n. 305/2011     - 95 occurrences
marcatura CE                - 164 occurrences
Dichiarazione di Produzione di Prodotto DOP - 142 occurrences
```

#### Installation Standards:
```
UNI 10818               - 14 occurrences (window/door installation)
UNI 11673-1             - 14 occurrences (installation joints)
UNI 11493:2013          - 4 occurrences (ceramic tile installation)
```

#### Fire Resistance:
```
EI60                    - 31 occurrences
EI 60                   - 11 occurrences
EI120                   - 6 occurrences
EI 120                  - 3 occurrences
EI30                    - 5 occurrences
REI60                   - 3 occurrences
REI 60                  - 2 occurrences
EN 13501-1              - 3 occurrences (fire classification)
UNI EN 1634-1:2001      - 6 occurrences (fire door testing)
EN 1634-1:2001          - 6 occurrences
```

#### Door/Window Standards:
```
EN 1935                 - 6 occurrences (hinges)
EN 14351-1              - 2 occurrences (windows/doors)
DIN 18650               - 2 occurrences (doors)
```

#### Materials Standards:
```
DIN 7748                - Present (PVC profiles)
UNI EN 14411            - 2 occurrences (ceramic tiles)
EN 14411                - 2 occurrences
UNI EN 10346            - 3 occurrences (steel)
EN 10346                - 3 occurrences
AISI 304                - 1 occurrence (stainless steel)
```

#### Paving/Stone Standards:
```
UNI EN 1339             - 4 occurrences (concrete paving)
EN 1339                 - 4 occurrences
```

#### Quality Standards:
```
ISO 9001-2000           - 2 occurrences
UNI EN ISO 9001-2000    - 1 occurrence
```

#### Other Standards:
```
EN 14195                - 2 occurrences
EN 520                  - 1 occurrence (plasterboard)
DIN 18171               - 2 occurrences
EN 13964:2007           - 2 occurrences (suspended ceilings)
```

---

## 4. MATERIALS (Materiali)

### Most Common Materials (with frequency):

#### Metals:
```
alluminio              - 107 occurrences (most common metal)
acciaio zincato        - 82 occurrences
acciaio                - 46 occurrences
acciaio inox           - 11 occurrences
ottone                 - 6 occurrences
```

#### Wood & Composites:
```
legno                  - 78 occurrences
MDF                    - 2 occurrences
multistrato            - 1 occurrence
```

#### Ceramics & Stone:
```
gres porcellanato      - 6 occurrences
ceramica               - 6 occurrences
gres                   - 1 occurrence
marmo                  - 3 occurrences
granito                - 5 occurrences
pietra                 - 7 occurrences
```

#### Plastics & Synthetics:
```
vetro                  - 33 occurrences
PVC/pvc                - 31 occurrences
plastica               - 21 occurrences
policarbonato          - 1 occurrence
nylon                  - 1 occurrence
poliammide             - 1 occurrence
TPE                    - Present (in descriptions)
```

### Material Combinations & Descriptions:
- "acciaio inox 304"
- "acciaio zincato spessore 15/10"
- "gres porcellanato prima scelta"
- "ceramica smaltata"
- "alluminio anodizzato"
- "vetro temperato"
- "cristallo"
- "legno prima scelta"

---

## 5. COLORS (Colori)

### Most Common Colors:
```
bianco                 - 46 occurrences (most common)
nero                   - 31 occurrences
grigio                 - 26 occurrences
```

### RAL Color Codes:
```
RAL (generic)          - 22 occurrences
RAL 9010               - 5 occurrences (pure white)
RAL 9003               - 4 occurrences (signal white)
RAL 9005               - 3 occurrences (jet black)
RAL 7043               - 2 occurrences (traffic grey)
```

### Surface Finishes:
```
cromato/cromata        - 2 occurrences
lucido                 - Present
satinato               - Present
```

### Color Descriptors Found:
- "colore bianco RAL 9016"
- "colore nero antracite"
- "colori chiari"
- "colori scuri"
- "colore Steel"
- "colore Earl Grey"
- "colore Anthracite"
- "colore Characoal"

### Color Phrases:
- "colore da definire e campionare"
- "finitura cromata"
- "effetto marmorizzato"

---

## 6. TECHNICAL SPECIFICATIONS

### Fire Resistance Classes:
```
classe A1              - 38 occurrences (non-combustible)
classe Bfl             - 3 occurrences
classe 1               - 4 occurrences (reazione al fuoco)
EI ratings             - See Standards section
```

### Performance Classes:

#### Slip Resistance:
```
R10                    - Present
R11                    - 3 occurrences
R9                     - 2 occurrences
R7                     - 4 occurrences
```

#### Water Tightness:
```
classe 7A              - 2 occurrences (tenuta all'acqua)
classe 9A              - 4 occurrences
```

#### Air Permeability:
```
classe 4               - 6 occurrences (permeabilit� all'aria)
```

#### Wind Resistance:
```
classe B2              - 1 occurrence (resistenza al vento)
```

### Thermal Performance:

#### U-values (Trasmittanza termica):
```
Uf = 1.00 W/mqK        - 2 occurrences
Uf = 1.30 W/mqK        - 3 occurrences
Uf = 1.0 W/mqK         - 3 occurrences
U = 1.3 W/mqK          - 3 occurrences
```

### Load/Weight Specifications:
```
portata 130kg          - Present (hinges)
peso specifications    - Rare
carico di rottura      - Present for stone (es. 500 kg/cmq)
```

---

## 7. INSTALLATION TYPES (Tipi di Installazione)

### Most Common Installation Methods:
```
a pavimento            - 44 occurrences (floor-mounted)
scorrevole             - 33 occurrences (sliding)
a parete               - 8 occurrences (wall-mounted)
ad incasso             - 8 occurrences (recessed/built-in)
incasso                - Present
oscillobattente        - 6 occurrences (tilt-and-turn)
sospeso/sospesa        - 5 occurrences (suspended/hung)
ribaltabile            - Present (folding)
```

### Installation Descriptions:
- "posa in opera"
- "posato a colla"
- "posato su massetto"
- "fissaggio a parete"
- "installazione su vano predisposto"

---

## 8. PRODUCT CODES & MODEL NUMBERS

### Product Code Formats Found:

#### Alphanumeric Codes:
```
B0024ULUTR, B0022ULUTR, B0032ULUTR    - Box doccia codes
243710001              - Geberit
26336400               - Hansgrohe
109.790.001            - Geberit
115.882.KJ.1           - Geberit
34294001               - Grohe
23982003               - Grohe
```

#### Simple Numeric Codes:
```
6+ digit codes common
Examples: 243710001, 26336400
```

#### Alphanumeric with Special Characters:
```
X811/01                - Goman
ZB13/01                - Goman
X3565D/01              - Goman
AN-M30/01, AN-M60/01   - Goman accessibility
```

#### Project Codes:
```
PX01, PX02, PX03, etc. - Project-specific codes
Fl108, FL102a, FL107   - Floor-related
CV09                   - Wall-related
HC02                   - Other categories
```

---

## 9. OTHER RECURRING PATTERNS

### Collection/Series Names:
```
Marazzi: Room, Cementum, Confetto, Tinta Unita
Emil: Pigmento, Medley
Living Ceramics: Cuit
Florim: Match Up
Grohe: Essence, Eurosmart
Geberit: Sigma (8, 20, 01)
```

### Common Product Types by Category:

#### Sanitaryware:
- Lavabo (sink/basin)
- Vaso sospeso (wall-hung toilet)
- WC a pavimento (floor toilet)
- Box doccia (shower enclosure)
- Cassetta di risciacquo (flush tank)
- Miscelatore monocomando (single-lever mixer)

#### Doors (Porte/Serramenti):
- Portafinestra (French door)
- Finestra (window)
- Portoncino (entrance door)
- Serramento in PVC/alluminio/legno
- Oscillobattente (tilt-and-turn)
- Scorrevole (sliding)

#### Flooring (Pavimenti):
- Piastrelle in gres porcellanato
- Ceramica smaltata
- Parquet/listoncini
- Laminato HPL

#### Walls (Rivestimenti):
- Rivestimento in gres
- Lastre di marmo
- Lamelle in legno

### Technical Terms Frequently Occurring:
- "compresi e compensati" (included and compensated)
- "marca e modello da definire" (brand and model to be defined)
- "da campionare per approvazione" (to be sampled for approval)
- "a regola d'arte" (according to best practice)
- "assistenze murarie" (masonry assistance)
- "pulizia finale" (final cleaning)
- "trasporto e tiro al piano" (transport and lifting to floor)

### Measurement Units:
- cm (centimetri) - most common
- mm (millimetri) - common
- m (metri) - less common
- mq or m² (metri quadri)
- l/min (liters per minute for flow rates)
- kg (kilograms)
- W/mqK (thermal transmittance)

---

## 10. RECOMMENDATIONS FOR EXTRACTION LOGIC

### High-Confidence Patterns to Extract:

1. **Brands**: Look for patterns like:
   - "tipo [Brand]"
   - "marca [Brand]"
   - Known brand names (Grohe, Geberit, Marazzi, etc.)

2. **Dimensions**: Extract using regex for:
   - `\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\s*(?:cm|mm|m)?`
   - "dim./dimensioni [size]"
   - "spessore [thickness]"
   - "Ø [diameter]"
   - "L=, H= [lengths]"

3. **Standards**: Extract:
   - UNI, EN, ISO followed by numbers
   - CE, DOP
   - EI/REI followed by numbers
   - "Regolamento n. [number]/[year]"

4. **Materials**: Extract common material terms:
   - acciaio (+ variants: inox, zincato)
   - alluminio (+ anodizzato)
   - legno, PVC, ceramica, gres, vetro, marmo, etc.

5. **Colors**: Extract:
   - Color names (bianco, nero, grigio, etc.)
   - RAL codes (RAL \d{4})
   - "colore [name]"

6. **Product Codes**: Extract:
   - After "cod./codice/art."
   - Alphanumeric patterns: [A-Z0-9]{6,}
   - Model-specific formats

7. **Technical Specs**: Extract:
   - "classe [value]"
   - "R[number]" (slip resistance)
   - "U = [value] W/mqK"
   - "EI[number]"

### Medium-Confidence Patterns:
- Model names after "modello/serie/collezione"
- Installation types in descriptive text
- Flow rates (l/min)

### Context-Dependent Patterns:
- Multiple dimensions in one text (distinguish floor vs. door vs. tile)
- Brand inference from model names
- Material combinations

---

## SUMMARY STATISTICS

**Dataset:** c:\Users\f.biggi\Scripts\roBERT\data\train\classification\raw\dataset_lim.jsonl

- **Total Records:** 3,186
- **Top Brand Category:** Building Materials (Knauf, Sch�ck)
- **Most Common Dimension:** 60x60 cm
- **Most Referenced Standard:** CE marking (121 occurrences)
- **Most Common Material:** Alluminio (107 occurrences)
- **Most Common Color:** Bianco (46 occurrences)
- **Most Common Installation:** A pavimento (44 occurrences)

---

**End of Report**
