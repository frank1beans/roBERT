import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Load dataset
dataset_path = Path(r"c:\Users\f.biggi\Scripts\roBERT\data\train\classification\raw\dataset_lim.jsonl")

# Initialize collectors
brands = Counter()
dimensions = Counter()
standards = Counter()
materials = Counter()
colors = Counter()
installation_types = Counter()
flow_rates = Counter()
other_patterns = defaultdict(Counter)

# Regex patterns
brand_patterns = [
    r'\b(Grohe|GROHE|Hansgrohe|HANSGROHE|Geberit|GEBERIT|Ideal Standard|IDEAL STANDARD)\b',
    r'\b(Marazzi|MARAZZI|Emil|EMIL|Florim|FLORIM|Living Ceramics|LIVING CERAMICS)\b',
    r'\b(Dyson|DYSON|DMP|Goman|GOMAN|Fantoni|FANTONI|Mapei|MAPEI)\b',
    r'\b(EcoContract|ECOCONTRACT)\b',
    r'\btipo\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
    r'\bmarca\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
]

dimension_patterns = [
    r'\b\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\s*(?:cm|mm|m)\b',  # 20x20 cm, 100x50x30 mm
    r'\b\d+x\d+(?:x\d+)?\b',  # 20x20 without units
    r'\bØ\s*\d+(?:\s*(?:cm|mm|m))?\b',  # Ø 25 cm, Ø25
    r'\bdiametro\s+\d+\s*(?:cm|mm|m)?\b',  # diametro 30mm
    r'\b[LlHh]\.?\s*=?\s*\d+(?:\s*(?:cm|mm|m))?\b',  # L=150, H 200
    r'\blunghezza\s+\d+\s*(?:cm|mm|m)?\b',
    r'\bspessore\s+\d+(?:[.,]\d+)?\s*(?:cm|mm|m)?\b',
    r'\bdim(?:\.|ensioni)?\s+\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\s*(?:cm|mm|m)?\b',
    r'\b(?:delle\s+)?dimensioni\s+(?:di\s+)?\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\s*(?:cm|mm|m)?\b',
]

standard_patterns = [
    r'\bUNI\s+(?:EN\s+)?(?:ISO\s+)?\d+(?:[:\-\.]\d+)*\b',  # UNI EN 997, UNI 10818
    r'\bEN\s+\d+(?:[:\-\.]\d+)*\b',  # EN 1329
    r'\bISO\s+\d+(?:[:\-\.]\d+)*\b',  # ISO 9001
    r'\bCE\b',
    r'\bDIN\s+\d+\b',  # DIN 7748
    r'\bDOP\b',  # Dichiarazione di Prodotto
    r'\bRegolamento\s+n\.\s*\d+\/\d+\b',  # Regolamento n. 305/2011
    r'\bAISI\s+\d+\b',  # AISI 304
]

material_patterns = [
    r'\b(?:acciaio|ACCIAIO)(?:\s+inox|\s+inossidabile|\s+zincato)?\b',
    r'\b(?:ceramica|CERAMICA|gres|GRES)(?:\s+porcellanato)?\b',
    r'\b(?:plastica|PLASTICA|PVC|pvc|policarbonato|POLICARBONATO)\b',
    r'\b(?:alluminio|ALLUMINIO|ottone|OTTONE|rame|RAME)\b',
    r'\b(?:legno|LEGNO|MDF|multistrato|MULTISTRATO)\b',
    r'\b(?:vetro|VETRO|cristallo|CRISTALLO)\b',
    r'\b(?:marmo|MARMO|granito|GRANITO|pietra|PIETRA|travertino|TRAVERTINO)\b',
    r'\b(?:nylon|NYLON|poliammide|POLIAMMIDE|TPE)\b',
]

color_patterns = [
    r'\b(?:bianco|BIANCO|nero|NERO|grigio|GRIGIO)\b',
    r'\b(?:cromato|CROMATO|cromata|CROMATA)\b',
    r'\b(?:RAL\s+\d+)\b',
    r'\b(?:colore|colori)\s+([a-zA-Z\s]+?)(?:\s+|,|\.|$)',
]

installation_patterns = [
    r'\ba\s+(?:parete|pavimento|soffitto)\b',
    r'\b(?:incasso|INCASSO|ad\s+incasso)\b',
    r'\b(?:sospeso|SOSPESO|sospesa|SOSPESA)\b',
    r'\b(?:scorrevole|SCORREVOLE)\b',
    r'\b(?:ribaltabile|RIBALTABILE)\b',
    r'\b(?:oscillobattente|OSCILLOBATTENTE)\b',
]

flow_rate_patterns = [
    r'\b\d+(?:[.,]\d+)?\s*l\/min\b',  # 5.7 l/min
    r'\bportata.*?\d+(?:[.,]\d+)?\s*l\/min\b',
]

print("Processing dataset...")
total_records = 0
sample_size = 0

with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        total_records += 1
        try:
            record = json.loads(line)
            text = record.get('text', '')

            # Every 15th record for representative sampling
            if total_records % 15 == 0:
                sample_size += 1

                # Extract brands
                for pattern in brand_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                        if match and len(match) > 2:
                            brands[match.strip()] += 1

                # Extract dimensions
                for pattern in dimension_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        dimensions[match.strip()] += 1

                # Extract standards
                for pattern in standard_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        standards[match.strip()] += 1

                # Extract materials
                for pattern in material_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        materials[match.strip().lower()] += 1

                # Extract colors
                for pattern in color_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                        if match and len(match) > 2:
                            colors[match.strip().lower()] += 1

                # Extract installation types
                for pattern in installation_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        installation_types[match.strip().lower()] += 1

                # Extract flow rates
                for pattern in flow_rate_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        flow_rates[match.strip()] += 1

        except json.JSONDecodeError:
            continue

print(f"\n{'='*80}")
print(f"DATASET ANALYSIS RESULTS")
print(f"{'='*80}")
print(f"Total records in dataset: {total_records}")
print(f"Records analyzed (every 15th): {sample_size}")
print(f"{'='*80}\n")

# Report Brands
print(f"\n{'='*80}")
print(f"1. BRANDS / MARCHE (Top 30)")
print(f"{'='*80}")
for brand, count in brands.most_common(30):
    print(f"  {brand:40} : {count:4} occurrences")

# Report Dimensions
print(f"\n{'='*80}")
print(f"2. DIMENSION PATTERNS (Top 40)")
print(f"{'='*80}")
print("\nFormat variations found:")
for dim, count in dimensions.most_common(40):
    print(f"  {dim:40} : {count:4} occurrences")

# Report Standards
print(f"\n{'='*80}")
print(f"3. STANDARDS / NORME (All)")
print(f"{'='*80}")
for std, count in standards.most_common():
    print(f"  {std:40} : {count:4} occurrences")

# Report Materials
print(f"\n{'='*80}")
print(f"4. MATERIALS / MATERIALI (Top 30)")
print(f"{'='*80}")
for mat, count in materials.most_common(30):
    print(f"  {mat:40} : {count:4} occurrences")

# Report Colors
print(f"\n{'='*80}")
print(f"5. COLORS / COLORI (Top 30)")
print(f"{'='*80}")
for col, count in colors.most_common(30):
    print(f"  {col:40} : {count:4} occurrences")

# Report Installation Types
print(f"\n{'='*80}")
print(f"6. INSTALLATION TYPES (All)")
print(f"{'='*80}")
for inst, count in installation_types.most_common():
    print(f"  {inst:40} : {count:4} occurrences")

# Report Flow Rates
print(f"\n{'='*80}")
print(f"7. FLOW RATES / PORTATE (All)")
print(f"{'='*80}")
for flow, count in flow_rates.most_common():
    print(f"  {flow:40} : {count:4} occurrences")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}\n")
