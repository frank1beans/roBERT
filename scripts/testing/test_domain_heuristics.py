"""Test domain heuristics integration with real extraction examples."""
import json
import sys
from pathlib import Path

# Test examples from user's extraction results
TEST_EXAMPLES = [
    {
        "text": "Miscelatore monocomando per lavabo con aeratore anticalcare integrato. Finitura cromata. Cartuccia in ceramica 35 mm. Portata 5 l/min.",
        "categoria": "sanitari",
        "expected": {
            "materiale": "metallo",  # NOT ceramica (cartuccia ceramica != materiale)
            "portata_l_min": 5.0,
        }
    },
    {
        "text": "Box doccia con cristallo temperato 8mm, profili in alluminio anodizzato, apertura scorrevole.",
        "categoria": "sanitari",
        "expected": {
            "materiale": "vetro_temperato",  # cristallo = vetro
            "spessore_mm": 8.0,
        }
    },
    {
        "text": "Lavabo sospeso in ceramica bianca, dimensioni 60x45 cm, installazione a parete",
        "categoria": "sanitari",
        "expected": {
            "materiale": "ceramica",
            "tipologia_installazione": "a_parete",
            "larghezza_cm": 60,
            "profondita_cm": 45,
        }
    },
    {
        "text": "Miscelatore AISI 304 per doccia, finitura spazzolata",
        "categoria": "sanitari",
        "expected": {
            "materiale": "acciaio_inox",  # AISI 304 = acciaio inox
        }
    },
]


def test_heuristics():
    """Test domain heuristics on real examples."""
    try:
        from robimb.extraction.domain_heuristics import (
            infer_material,
            infer_installation_type,
            validate_material_consistency,
        )
    except ImportError as e:
        print(f"[ERRORE] Impossibile importare modulo domain_heuristics: {e}")
        return False

    print("=" * 80)
    print("TEST DOMAIN HEURISTICS")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, example in enumerate(TEST_EXAMPLES, 1):
        print(f"\n[TEST {i}] {example['text'][:60]}...")
        print(f"Categoria: {example['categoria']}")

        text = example["text"]
        categoria = example["categoria"]
        expected = example["expected"]

        # Test material inference
        if "materiale" in expected:
            result = infer_material(text, categoria)
            if result:
                extracted = result["value"]
                expected_val = expected["materiale"]

                if extracted == expected_val:
                    print(f"  [OK] Materiale: {extracted} (confidence: {result['confidence']:.2f})")
                    passed += 1
                else:
                    print(f"  [FAIL] Materiale: {extracted} != {expected_val}")
                    failed += 1

                # Test validation
                validation = validate_material_consistency(extracted, text, categoria)
                if not validation["is_valid"]:
                    print(f"  [WARN] Validazione: {validation['warnings']}")
            else:
                print(f"  [FAIL] Materiale non estratto (atteso: {expected['materiale']})")
                failed += 1

        # Test installation type inference
        if "tipologia_installazione" in expected:
            result = infer_installation_type(text)
            if result:
                extracted = result["value"]
                expected_val = expected["tipologia_installazione"]

                if extracted == expected_val:
                    print(f"  [OK] Installazione: {extracted} (confidence: {result['confidence']:.2f})")
                    passed += 1
                else:
                    print(f"  [FAIL] Installazione: {extracted} != {expected_val}")
                    failed += 1
            else:
                print(f"  [FAIL] Installazione non estratta (atteso: {expected['tipologia_installazione']})")
                failed += 1

    print("\n" + "=" * 80)
    print(f"RISULTATI: {passed} OK, {failed} FAIL")
    print("=" * 80)

    return failed == 0


def test_ceramic_mixer_issue():
    """Test the specific ceramic/mixer issue from user's results."""
    from robimb.extraction.domain_heuristics import (
        validate_material_consistency,
    )

    print("\n" + "=" * 80)
    print("TEST ISSUE: Miscelatore con 'ceramica' (dovrebbe essere metallo)")
    print("=" * 80)

    text = "Miscelatore monocomando con cartuccia in ceramica"
    material_value = "ceramica"
    categoria = "sanitari"

    validation = validate_material_consistency(material_value, text, categoria)

    print(f"Testo: {text}")
    print(f"Materiale estratto: {material_value}")
    print(f"Validazione:")
    print(f"  - is_valid: {validation['is_valid']}")
    print(f"  - confidence_adjustment: {validation['confidence_adjustment']}")
    print(f"  - warnings: {validation['warnings']}")

    # Should detect inconsistency
    assert not validation["is_valid"], "Dovrebbe rilevare inconsistenza"
    assert validation["confidence_adjustment"] < 0, "Dovrebbe ridurre confidence"
    assert len(validation["warnings"]) > 0, "Dovrebbe avere warning"

    print("\n[OK] Issue correttamente rilevato!")
    return True


if __name__ == "__main__":
    success = True

    try:
        success &= test_heuristics()
        success &= test_ceramic_mixer_issue()
    except Exception as e:
        print(f"\n[ERRORE] Test fallito: {e}")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
