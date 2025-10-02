import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Load dataset
dataset_path = Path(r"c:\Users\f.biggi\Scripts\roBERT\data\train\classification\raw\dataset_lim.jsonl")

# Initialize collectors
product_codes = Counter()
model_numbers = Counter()
certifications = Counter()
thickness_patterns = Counter()
diameter_patterns = Counter()
length_patterns = Counter()
area_patterns = Counter()
technical_specs = defaultdict(Counter)

# Regex patterns
code_patterns = [
    r'\b(?:cod\.|codice|cod|code|art\.|articolo)\s*[:\.]?\s*([A-Z0-9\-\/\.]+)\b',
    r'\b([A-Z]{1,3}\d{3,}[A-Z\d]*)\b',  # Product codes like B0024ULUTR
    r'\b(\d{6,})\b',  # 6+ digit codes
]

model_patterns = [
    r'\bmodello\s+([A-Za-z0-9\s\-]+?)(?:\s+cod|\s+dim|,|\.|$)',
    r'\bserie\s+([A-Za-z0-9\s\-]+?)(?:\s+cod|\s+dim|,|\.|$)',
    r'\bcollezione\s+([A-Za-z0-9\s\-]+?)(?:\s+cod|\s+dim|,|\.|$)',
]

certification_patterns = [
    r'\bmarcatura\s+CE\b',
    r'\bmarcati\s+CE\b',
    r'\bmarcato\s+CE\b',
    r'\bDichiarazione\s+(?:di\s+)?Produzione\s+(?:di\s+)?Prodotto\s+DOP\b',
    r'\breazione\s+al\s+fuoco\s+(?:classe\s+)?([A-Z0-9]+)\b',
    r'\bresistenza\s+al\s+fuoco\s+([A-Z0-9\-]+)\b',
    r'\bEI\s*\d+\b',
    r'\bREI\s*\d+\b',
]

thickness_patterns_regex = [
    r'\bspessore\s+(?:di\s+)?(\d+(?:[.,]\d+)?)\s*(?:cm|mm|m)?\b',
    r'\bsp\.\s*(\d+(?:[.,]\d+)?)\s*(?:cm|mm)?\b',
]

diameter_patterns_regex = [
    r'\bØ\s*(\d+(?:[.,]\d+)?)\s*(?:cm|mm)?\b',
    r'\bdiametro\s+(\d+(?:[.,]\d+)?)\s*(?:cm|mm)?\b',
]

length_patterns_regex = [
    r'\blunghezza\s+(\d+(?:[.,]\d+)?)\s*(?:cm|mm|m)?\b',
    r'\bL\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(?:cm|mm)?\b',
]

area_patterns_regex = [
    r'\b(\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?)\s*(?:cm|mm|m)?\b',
]

# Technical specs
r_value_patterns = [
    r'\bR\s*=?\s*(\d+(?:[.,]\d+)?)\b',
    r'\bresistenza.*?(\d+(?:[.,]\d+)?)\s*(?:kg\/cmq|kg\/cm2|MPa)\b',
]

u_value_patterns = [
    r'\bUf?\s*=\s*(\d+[.,]\d+)\s*W\/(?:mq|m2)?K\b',
    r'\btrasmittanza.*?(\d+[.,]\d+)\s*W\/(?:mq|m2)?K\b',
]

class_patterns = [
    r'\bclasse\s+([A-Z0-9]+)\b',
]

weight_patterns = [
    r'\bpeso\s+(\d+(?:[.,]\d+)?)\s*(?:kg|g)\b',
    r'\bportata\s+(\d+(?:[.,]\d+)?)\s*kg\b',
]

print("Processing dataset for detailed patterns...")
total_records = 0
sample_size = 0

with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        total_records += 1
        try:
            record = json.loads(line)
            text = record.get('text', '')

            # Sample every 10th record
            if total_records % 10 == 0:
                sample_size += 1

                # Extract product codes
                for pattern in code_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if len(match) > 2:  # Filter out very short codes
                            product_codes[match.strip()] += 1

                # Extract model numbers
                for pattern in model_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if len(match.strip()) > 2:
                            model_numbers[match.strip()] += 1

                # Extract certifications
                for pattern in certification_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, str) and match:
                            certifications[match.strip()] += 1
                        elif match:
                            certifications[pattern[:30]] += 1

                # Extract thickness
                for pattern in thickness_patterns_regex:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        thickness_patterns[f"{match} mm/cm"] += 1

                # Extract diameter
                for pattern in diameter_patterns_regex:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        diameter_patterns[f"Ø {match}"] += 1

                # Extract length
                for pattern in length_patterns_regex:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        length_patterns[f"L {match}"] += 1

                # Extract area/dimensions
                for pattern in area_patterns_regex:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        area_patterns[match.strip()] += 1

                # Extract R-values
                for pattern in r_value_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1] if len(match) > 1 else ''
                        if match:
                            technical_specs['R-value'][f"R={match}"] += 1

                # Extract U-values
                for pattern in u_value_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        technical_specs['U-value'][f"U={match} W/mqK"] += 1

                # Extract classes
                for pattern in class_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        technical_specs['classe'][f"classe {match}"] += 1

                # Extract weights
                for pattern in weight_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        technical_specs['weight'][f"{match} kg"] += 1

        except json.JSONDecodeError:
            continue

print(f"\n{'='*80}")
print(f"DETAILED DATASET ANALYSIS")
print(f"{'='*80}")
print(f"Total records: {total_records}")
print(f"Analyzed records (every 10th): {sample_size}")
print(f"{'='*80}\n")

# Report Product Codes
print(f"\n{'='*80}")
print(f"PRODUCT CODES / CODICI ARTICOLO (Top 40)")
print(f"{'='*80}")
for code, count in product_codes.most_common(40):
    print(f"  {code:40} : {count:3} occurrences")

# Report Model Numbers
print(f"\n{'='*80}")
print(f"MODEL NUMBERS / MODELLI (Top 30)")
print(f"{'='*80}")
for model, count in model_numbers.most_common(30):
    print(f"  {model:40} : {count:3} occurrences")

# Report Certifications
print(f"\n{'='*80}")
print(f"CERTIFICATIONS / CERTIFICAZIONI (Top 20)")
print(f"{'='*80}")
for cert, count in certifications.most_common(20):
    print(f"  {cert:40} : {count:3} occurrences")

# Report Thickness Patterns
print(f"\n{'='*80}")
print(f"THICKNESS PATTERNS / SPESSORI (Top 30)")
print(f"{'='*80}")
for thick, count in thickness_patterns.most_common(30):
    print(f"  {thick:40} : {count:3} occurrences")

# Report Diameter Patterns
print(f"\n{'='*80}")
print(f"DIAMETER PATTERNS / DIAMETRI (Top 20)")
print(f"{'='*80}")
for diam, count in diameter_patterns.most_common(20):
    print(f"  {diam:40} : {count:3} occurrences")

# Report Length Patterns
print(f"\n{'='*80}")
print(f"LENGTH PATTERNS / LUNGHEZZE (Top 20)")
print(f"{'='*80}")
for length, count in length_patterns.most_common(20):
    print(f"  {length:40} : {count:3} occurrences")

# Report Area/Dimensions
print(f"\n{'='*80}")
print(f"AREA/DIMENSION PATTERNS (Top 40)")
print(f"{'='*80}")
for area, count in area_patterns.most_common(40):
    print(f"  {area:40} : {count:3} occurrences")

# Report Technical Specs
print(f"\n{'='*80}")
print(f"TECHNICAL SPECIFICATIONS")
print(f"{'='*80}")

for spec_type, specs in technical_specs.items():
    print(f"\n{spec_type.upper()} (Top 15):")
    for spec, count in specs.most_common(15):
        print(f"  {spec:40} : {count:3} occurrences")

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}\n")
