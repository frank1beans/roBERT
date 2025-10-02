"""Test domain heuristics on real extraction examples from user."""
import json
import sys

# Real examples from user's extraction results
REAL_EXAMPLES = [
    {
        "id": "PX09",
        "text": "Miscelatore monocomando per lavabo tipo Grohe, serie Essence, taglia S. "
                "Fornitura e posa di miscelatore monocomando per lavabo tipo Grohe Essence Taglia S "
                "cod. 34294001, provvisto di cartuccia a dischi ceramici da 28 mm con tecnologia "
                "Grohe SilkMove per risparmio energetico",
        "categoria": "apparecchi_sanitari_accessori",
        "current_extraction": {
            "materiale": "ceramica",  # WRONG! Should be metallo
            "source": "qa_llm",
            "confidence": 0.9
        },
        "expected": {
            "materiale": "metallo",  # Mixers are metal, not ceramic (ceramic is the cartridge)
        }
    },
    {
        "id": "PX18",
        "text": "Fornitura e posa in opera di seggiolino a doghe ribaltabile a parete bianco "
                "Tipo GOMAN Art. X811/01 per bagno disabili.",
        "categoria": "apparecchi_sanitari_accessori",
        "current_extraction": {
            "materiale": None,  # MISSING
        },
        "expected": {
            "materiale": "legno",  # "doghe" = wood slats
        }
    },
    {
        "id": "Piletta_PVC",
        "text": "Fornitura e posa in opera di piletta di scarico Locali Tecnici, in pvc dim. 25x25cm.",
        "categoria": "apparecchi_sanitari_accessori",
        "current_extraction": {
            "materiale": None,  # MISSING
        },
        "expected": {
            "materiale": "plastica_pvc",  # "in pvc" should be detected
        }
    },
    {
        "id": "PX06b",
        "text": "Fornitura e posa in opera di maniglione lineare di sostegno, marca e modello da definire, lunghezza 60 cm.",
        "categoria": "apparecchi_sanitari_accessori",
        "current_extraction": {
            "materiale": None,  # MISSING
        },
        "expected": {
            "materiale": "metallo",  # Support handrails are typically metal
        }
    },
    {
        "id": "PX11",
        "text": "Fornitura e posa in opera di Doccetta tipo Hansgrohe Crometta Vario green, cromata, cod. art. 26336400.",
        "categoria": "apparecchi_sanitari_accessori",
        "current_extraction": {
            "materiale": None,  # MISSING
        },
        "expected": {
            "materiale": "metallo_cromato",  # "cromata" = chrome-plated metal
        }
    },
]


def test_real_examples():
    """Test domain heuristics on real extraction failures."""
    try:
        from robimb.extraction.domain_heuristics import (
            infer_material,
            validate_material_consistency,
        )
    except ImportError as e:
        print(f"[ERRORE] Impossibile importare domain_heuristics: {e}")
        return False

    print("=" * 80)
    print("TEST EURISTICHE SU ESEMPI REALI")
    print("=" * 80)

    passed = 0
    failed = 0
    fixed = 0

    for example in REAL_EXAMPLES:
        print(f"\n[TEST {example['id']}]")
        print(f"Testo: {example['text'][:80]}...")

        text = example["text"]
        categoria = example["categoria"]
        current = example["current_extraction"]
        expected = example["expected"]

        # Test material inference
        if "materiale" in expected:
            result = infer_material(text, categoria)
            expected_val = expected["materiale"]

            if result:
                extracted = result["value"]
                current_val = current.get("materiale")

                print(f"  Attuale:  {current_val}")
                print(f"  Euristica: {extracted} (conf: {result['confidence']:.2f}, source: {result['source']})")
                print(f"  Atteso:    {expected_val}")

                if extracted == expected_val:
                    print(f"  [OK] PASS - Euristica corregge l'estrazione!")
                    passed += 1
                    if current_val != expected_val:
                        fixed += 1
                        print(f"  [FIX] {current_val} -> {extracted}")
                else:
                    print(f"  [FAIL] Euristica: {extracted} != atteso: {expected_val}")
                    failed += 1

                # If current extraction was wrong, validate
                if current_val and current_val != expected_val:
                    validation = validate_material_consistency(current_val, text, categoria)
                    if not validation["is_valid"]:
                        print(f"  [WARN] Validazione rileva problema: {validation['warnings']}")
                        print(f"         Confidence adjustment: {validation['confidence_adjustment']}")
            else:
                print(f"  [FAIL] Euristica non estrae nulla (atteso: {expected_val})")
                failed += 1

    print("\n" + "=" * 80)
    print(f"RISULTATI: {passed} PASS, {failed} FAIL")
    print(f"FIX APPLICATI: {fixed} estrazioni corrette dalle euristiche")
    print("=" * 80)

    return failed == 0


def analyze_improvement():
    """Analyze improvement from heuristics."""
    print("\n" + "=" * 80)
    print("ANALISI MIGLIORAMENTO")
    print("=" * 80)

    total_tests = len(REAL_EXAMPLES)
    missing_before = sum(1 for ex in REAL_EXAMPLES if ex["current_extraction"].get("materiale") is None)
    wrong_before = sum(1 for ex in REAL_EXAMPLES
                      if ex["current_extraction"].get("materiale") not in [None, ex["expected"]["materiale"]])

    print(f"\nPRIMA delle euristiche:")
    print(f"  - Totale esempi: {total_tests}")
    print(f"  - Materiale mancante: {missing_before} ({missing_before/total_tests*100:.0f}%)")
    print(f"  - Materiale errato: {wrong_before} ({wrong_before/total_tests*100:.0f}%)")
    print(f"  - Totale problemi: {missing_before + wrong_before} ({(missing_before + wrong_before)/total_tests*100:.0f}%)")

    print(f"\nDOPO le euristiche (se tutti i test passano):")
    print(f"  - Materiale mancante: 0 (0%)")
    print(f"  - Materiale errato: 0 (0%)")
    print(f"  - Totale problemi: 0 (0%)")

    improvement = (missing_before + wrong_before) / total_tests * 100
    print(f"\n[+] MIGLIORAMENTO ATTESO: {improvement:.0f}% -> 0% = {improvement:.0f}% di riduzione errori")


if __name__ == "__main__":
    success = True

    try:
        analyze_improvement()
        success = test_real_examples()
    except Exception as e:
        print(f"\n[ERRORE] Test fallito: {e}")
        import traceback
        traceback.print_exc()
        success = False

    sys.exit(0 if success else 1)
