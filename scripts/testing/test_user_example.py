"""Test the exact example from user's XML."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force UTF-8 output
import io
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.robimb.extraction.domain_heuristics import post_process_properties

# Exact text from user's XML
text = """Parete in cartongesso standard-idrolastra, con doppia orditura metallica da 75 mm, isolata, sp. 15 cm

- Lastra in cartongesso standard, sp. 12,5 mm
- Orditura metallica da 75 mm, passo 60 cm
- Orditura metallica da 75 mm, passo 60 cm
- Isolante in lana minerale sp. 60 mm, densità 40 kg/m3
- Isolante in lana minerale sp. 60 mm, densità 40 kg/m3
- Idrolastra, sp. 12,5 mm

Classe di reazione al fuoco: A2-s1,d0
Regolamento CE 305/2011 (CPR)
"""

print("=" * 80)
print("USER EXAMPLE TEST - Double Orditure")
print("=" * 80)
print()

# Post-process properties (as would happen in production)
result = {}
post_process_properties(
    text=text,
    category="opere_da_cartongessista",
    properties_payload=result
)

print("Extracted Properties:")
print()
print(f"DEBUG: Keys in result: {list(result.keys())}")
print()

if "stratigrafia_lastre" in result:
    strat = result["stratigrafia_lastre"]
    print(f"✓ stratigrafia_lastre found")
    print(f"  Source: {strat.get('source')}")
    print(f"  Confidence: {strat.get('confidence')}")

    value = strat.get('value')
    if value:
        print(f"\n  Value:")
        print(f"    Lastre: {len(value.get('lastre', []))}")
        for lastra in value.get('lastre', []):
            print(f"      - {lastra['id']}: {lastra['spessore_mm']} mm ({lastra['tipologia']})")

        print(f"    Orditure: {len(value.get('orditure', []))}")
        for orditura in value.get('orditure', []):
            print(f"      - {orditura['id']}: {orditura['larghezza_mm']} mm")

        print(f"    Isolanti: {len(value.get('isolanti', []))}")
        for isolante in value.get('isolanti', []):
            mat = isolante.get('materiale', 'N/A')
            dens = isolante.get('densita_kg_m3', 'N/A')
            print(f"      - {isolante['id']}: {isolante['spessore_mm']} mm ({mat}, {dens} kg/m³)")

        print(f"    Sequenza: {value.get('sequenza')}")

    raw = strat.get('raw')
    print(f"\n  Raw field type: {type(raw).__name__}")
    if isinstance(raw, str):
        print(f"  ✓ Raw is string (valid for schema)")
        # Verify it's valid JSON
        try:
            parsed = json.loads(raw)
            print(f"  ✓ Raw is valid JSON")
        except:
            print(f"  ✗ Raw is NOT valid JSON")
    else:
        print(f"  ✗ Raw is {type(raw).__name__} (should be string)")
else:
    print("✗ stratigrafia_lastre NOT found")

if "classe_reazione_al_fuoco" in result:
    print(f"\n✓ classe_reazione_al_fuoco: {result['classe_reazione_al_fuoco']['value']}")

if "normativa_riferimento" in result:
    print(f"✓ normativa_riferimento: {result['normativa_riferimento']['value']}")

print()
print("=" * 80)
print("VALIDATION CHECKS")
print("=" * 80)

# Only run checks if stratigrafia was extracted
if "stratigrafia_lastre" in result and result["stratigrafia_lastre"].get('value'):
    strat = result["stratigrafia_lastre"]
    value = strat.get('value', {})

    checks = [
        ("2 lastre extracted", len(value.get('lastre', [])) == 2),
        ("2 orditure extracted", len(value.get('orditure', [])) == 2),
        ("2 isolanti extracted", len(value.get('isolanti', [])) == 2),
        ("All orditure are 75mm", all(o['larghezza_mm'] == 75.0 for o in value.get('orditure', []))),
        ("All isolanti are 60mm", all(i['spessore_mm'] == 60.0 for i in value.get('isolanti', []))),
        ("Raw field is string", isinstance(strat.get('raw'), str)),
        ("Reaction class A2-s1,d0", result.get('classe_reazione_al_fuoco', {}).get('value') == 'A2-s1,d0'),
    ]
else:
    print("✗ Cannot run checks - stratigrafia_lastre not found")
    checks = []

all_pass = True
for check_name, check_result in checks:
    if check_result:
        print(f"✓ {check_name}")
    else:
        print(f"✗ {check_name}")
        all_pass = False

print()
if all_pass:
    print("=" * 80)
    print("ALL CHECKS PASSED ✓")
    print("=" * 80)
else:
    print("=" * 80)
    print("SOME CHECKS FAILED ✗")
    print("=" * 80)
