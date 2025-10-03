"""Test double orditure extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.robimb.extraction.cartongesso import extract_cartongesso_features, summarize_cartongesso_features

# Example from user's XML with double orditure
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
print("TEST: Double Orditure Extraction")
print("=" * 80)
print(f"Text:\n{text}")
print()

features = extract_cartongesso_features(text)

if features:
    print(f"✓ Features extracted")
    print(f"  Layers: {len(features.layers)}")
    for i, layer in enumerate(features.layers, 1):
        print(f"    {i}. {layer.type} - {layer.thickness_mm} mm")
    print(f"  Frame width (legacy): {features.frame_width_mm}")
    print(f"  Insulation (legacy): {features.insulation_material} - {features.insulation_thickness_mm} mm")
    print()

    summary = summarize_cartongesso_features(features, text=text)
    stratigrafia = summary.get("stratigrafia")

    if stratigrafia:
        print("Stratigrafia:")
        print(f"  Lastre: {len(stratigrafia['lastre'])}")
        for lastra in stratigrafia['lastre']:
            print(f"    - {lastra['id']}: {lastra['spessore_mm']} mm ({lastra['tipologia']})")

        print(f"  Orditure: {len(stratigrafia['orditure'])}")
        for orditura in stratigrafia['orditure']:
            print(f"    - {orditura['id']}: {orditura['larghezza_mm']} mm")

        print(f"  Isolanti: {len(stratigrafia['isolanti'])}")
        for isolante in stratigrafia['isolanti']:
            mat = isolante.get('materiale', 'N/A')
            dens = isolante.get('densita_kg_m3', 'N/A')
            print(f"    - {isolante['id']}: {isolante['spessore_mm']} mm ({mat}, {dens} kg/m³)")

        print(f"  Sequenza: {stratigrafia['sequenza']}")
        print()

        # Check expectations
        expected_orditure = 2
        expected_isolanti = 2
        actual_orditure = len(stratigrafia['orditure'])
        actual_isolanti = len(stratigrafia['isolanti'])

        print("=" * 80)
        print("VALIDATION:")
        print("=" * 80)
        if actual_orditure == expected_orditure:
            print(f"✓ Orditure count correct: {actual_orditure}")
        else:
            print(f"✗ Orditure count wrong: expected {expected_orditure}, got {actual_orditure}")

        if actual_isolanti == expected_isolanti:
            print(f"✓ Isolanti count correct: {actual_isolanti}")
        else:
            print(f"✗ Isolanti count wrong: expected {expected_isolanti}, got {actual_isolanti}")

        # Check all orditure are 75mm
        all_75 = all(ord['larghezza_mm'] == 75.0 for ord in stratigrafia['orditure'])
        if all_75:
            print(f"✓ All orditure are 75 mm")
        else:
            print(f"✗ Not all orditure are 75 mm")

        # Check all isolanti are 60mm
        all_60 = all(iso['spessore_mm'] == 60.0 for iso in stratigrafia['isolanti'])
        if all_60:
            print(f"✓ All isolanti are 60 mm")
        else:
            print(f"✗ Not all isolanti are 60 mm")
    else:
        print("✗ No stratigrafia in summary")
else:
    print("✗ No features extracted")
