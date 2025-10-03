"""Comprehensive cartongesso extraction tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.robimb.extraction.cartongesso import extract_cartongesso_features, summarize_cartongesso_features

TEST_CASES = [
    {
        "name": "Single orditure + insulation",
        "text": """Parete in cartongesso standard, con orditura metallica da 100 mm, isolata

- Lastra in cartongesso standard, sp. 12,5 mm
- Orditura metallica da 100 mm, passo 60 cm
- Isolante in lana di roccia sp. 50 mm, densità 30 kg/m3
- Lastra in cartongesso standard, sp. 12,5 mm

Classe EI60
Classe di reazione al fuoco: A1
""",
        "expected": {
            "lastre": 2,
            "orditure": 1,
            "isolanti": 1,
            "orditura_width": 100.0,
            "isolante_thickness": 50.0,
        }
    },
    {
        "name": "Double orditure + double insulation",
        "text": """Parete in cartongesso standard-idrolastra, con doppia orditura metallica da 75 mm, isolata

- Lastra in cartongesso standard, sp. 12,5 mm
- Orditura metallica da 75 mm, passo 60 cm
- Orditura metallica da 75 mm, passo 60 cm
- Isolante in lana minerale sp. 60 mm, densità 40 kg/m3
- Isolante in lana minerale sp. 60 mm, densità 40 kg/m3
- Idrolastra, sp. 12,5 mm

Classe di reazione al fuoco: A2-s1,d0
""",
        "expected": {
            "lastre": 2,
            "orditure": 2,
            "isolanti": 2,
            "orditura_width": 75.0,
            "isolante_thickness": 60.0,
        }
    },
    {
        "name": "Triple layers (ignifuga)",
        "text": """Controsoffitto in cartongesso ignifugo

- Ignilastra sp. 12,5 mm
- Orditura metallica C 50/27 mm
- Ignilastra sp. 15 mm
- Ignilastra sp. 12,5 mm

REI 120
Classe A1
""",
        "expected": {
            "lastre": 3,
            "orditure": 1,
            "isolanti": 0,
            "orditura_width": 50.0,
        }
    },
    {
        "name": "No frame (direct application)",
        "text": """Controparete con lastre direttamente incollate

- Lastra in cartongesso standard, sp. 10 mm
- Lastra in cartongesso standard, sp. 12,5 mm

Classe B-s1,d0
""",
        "expected": {
            "lastre": 2,
            "orditure": 0,
            "isolanti": 0,
        }
    },
]

def run_test(test_case):
    """Run a single test case."""
    print("=" * 80)
    print(f"TEST: {test_case['name']}")
    print("=" * 80)

    features = extract_cartongesso_features(test_case['text'])
    if not features:
        print("✗ FAIL - No features extracted")
        return False

    summary = summarize_cartongesso_features(features, text=test_case['text'])
    stratigrafia = summary.get("stratigrafia")

    if not stratigrafia:
        print("✗ FAIL - No stratigrafia")
        return False

    expected = test_case['expected']
    all_ok = True

    # Check counts
    actual_lastre = len(stratigrafia['lastre'])
    actual_orditure = len(stratigrafia['orditure'])
    actual_isolanti = len(stratigrafia['isolanti'])

    print(f"Lastre: {actual_lastre} (expected: {expected['lastre']})")
    if actual_lastre != expected['lastre']:
        print(f"  ✗ FAIL - Expected {expected['lastre']}, got {actual_lastre}")
        all_ok = False
    else:
        print("  ✓ OK")

    print(f"Orditure: {actual_orditure} (expected: {expected['orditure']})")
    if actual_orditure != expected['orditure']:
        print(f"  ✗ FAIL - Expected {expected['orditure']}, got {actual_orditure}")
        all_ok = False
    else:
        print("  ✓ OK")

    print(f"Isolanti: {actual_isolanti} (expected: {expected['isolanti']})")
    if actual_isolanti != expected['isolanti']:
        print(f"  ✗ FAIL - Expected {expected['isolanti']}, got {actual_isolanti}")
        all_ok = False
    else:
        print("  ✓ OK")

    # Check specific values if expected
    if 'orditura_width' in expected and actual_orditure > 0:
        first_width = stratigrafia['orditure'][0]['larghezza_mm']
        if first_width != expected['orditura_width']:
            print(f"  ✗ FAIL - Orditura width: expected {expected['orditura_width']}, got {first_width}")
            all_ok = False

    if 'isolante_thickness' in expected and actual_isolanti > 0:
        first_thickness = stratigrafia['isolanti'][0]['spessore_mm']
        if first_thickness != expected['isolante_thickness']:
            print(f"  ✗ FAIL - Isolante thickness: expected {expected['isolante_thickness']}, got {first_thickness}")
            all_ok = False

    # Display sequenza
    print(f"\nSequenza: {stratigrafia['sequenza']}")

    if all_ok:
        print("\n✓ PASS")
    else:
        print("\n✗ FAIL")

    print()
    return all_ok

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE CARTONGESSO EXTRACTION TESTS")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        if run_test(test_case):
            passed += 1
        else:
            failed += 1

    print("=" * 80)
    print(f"RESULTS: {passed} PASS, {failed} FAIL")
    print("=" * 80)

    sys.exit(0 if failed == 0 else 1)
