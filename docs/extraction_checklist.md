# Checklist estrazione proprietà prioritari
Generata automaticamente il 2025-09-23 21:09:58Z a partire da `data/properties_registry_extended.json`.
Per ogni categoria sono elencati gli slot prioritari strutturati (numerici, enum, stringhe) con le regex di supporto e i normalizzatori suggeriti.

## Apparecchi sanitari e accessori|Accessori per l'allestimento di servizi igienici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_accessorio` | enum | `\bdispenser|porta\s*salviet?te|porta\s*rotolo|specchi[oi]|maniglion[ei]|appendin[oi]\b` | `lower` |
| `materiale` | enum | `\bAISI\s*30[46]|alluminio|ABS|vetro\b` | `lower` |

## Apparecchi sanitari e accessori|Apparecchi sanitari
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bwc|bidet|lavab[oi]|piatto\s*doccia|urinale\b` | `lower` |
| `ceramica_trattata` | bool | `\b(trattamento\s*anticalcare|ceramica\s*trattata|smalto\s*attivo)\b` | `map_yes_no_multilang` |
| `scarico_litri` | float | `\b(3\.?0?|4\.?5|6\.?0?|7\.?5|9\.?0?)\s*l\b|\b(3|4\.5|6|7\.5|9)\s*litri\b` | `comma_to_dot`, `to_float` |

## Apparecchi sanitari e accessori|Cassette di scarico
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `installazione` | enum | `\bincasso|esterna|monoblocco\b` | `lower` |
| `portata_scarico_litri` | enum | `\b(3\s*/\s*6|4\.?5\s*/\s*9|dual\s*flush)\b` | `lower`, `split_structured_list` |
| `comando` | enum | `\bplacca\s*(meccanica|pneumatica)|sensore\b` | `lower` |

## Arredi standard|Altri arredi standard
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Arredi standard|Arredi per esterno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `trattamento_superficiale` | enum | `\bzincatura|polver[ei]|oliat[oa]\b` | `lower` |

## Arredi standard|Arredi per strutture ricettive
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_reazione_fuoco` | enum | `\bClasse\s*(1IM|2IM)\b` | `lower` |

## Arredi standard|Arredo bagno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bmobile\s*sospeso|a\s*terra|colonna|specchier[ae]|piano\s*lavabo\b` | `lower` |

## Arredi standard|Arredo per uffici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `postazione_tipo` | enum | `\boperativ[ao]|direzional[ei]|meeting|bench\b` | `lower` |

## Arredi su misura|Arredi su misura in altri materiali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `posizione_installazione` | enum | `\bintern[oi]|estern[oi]\b` | `lower` |

## Arredi su misura|Arredi su misura in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `sistema_apertura` | enum | `\bbattent[ei]|scorrevol[ei]|vasistas|push\s*pull\b` | `lower` |

## Arredi su misura|Arredi su misura insegne e simboli
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `illuminazione` | enum | `\bLED|retroilluminat[ao]|retro\s*illuminat[ao]\b` | `lower` |

## Assistenze murarie|Assistenze murarie ai subappaltatori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `numero_addetti` | int | `\b(\d{1,2})\s*addett[oi]\b` | `to_int` |
| `durata_ore` | float | `\b(\d{1,3}(?:[.,]\d)?)\s*ore\b` | `comma_to_dot`, `to_float` |
| `tipologia_supporto` | enum | `\b(aperture|scasso|riprese\s*intonaco|tracce|fori)\b` | `lower` |

## Assistenze murarie|Assistenze murarie alla posa di impianti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `impianto` | enum | `\b(elettric[oi]|idraulic[oi]|hvac|climatizzazione|speciali)\b` | `lower` |
| `tracce_ml` | float | `\b(\d{1,4})\s*ml\s*tracce\b|\btracce\s*(\d{1,4})\s*ml\b` | `comma_to_dot`, `to_float` |
| `ripristini_m2` | float | `\b(\d{1,5})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2)\s*ripristi?n[oi]?\b`<br>`\b(\d{1,5})\s*(mq|m2)\s*ripristi?n[oi]?\b` | `comma_to_dot`, `to_float` |

## Cantierizzazioni|Impianti di cantiere
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `potenza_elettrica_kVA` | float | `\b(\d{1,3}(?:[.,]\d)?)\s*(?:kVA|kva|kilovolt[\\s-]*ampere)\b`<br>`\b(\d{1,3}(?:[.,]\d)?)\s*kVA\b` | `comma_to_dot`, `to_float` |
| `quadri_elettrici_n` | int | `\b(\d{1,2})\s*quadri?\b` | `to_int` |
| `approvvigionamento_idrico` | enum | `\b(allaccio\s*rete|autobotte|pozzo)\b` | `lower` |

## Cantierizzazioni|Installazione cantiere
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `superficie_area_m2` | float | `\b(\d{2,5})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2)\b`<br>`\b(\d{2,5})\s*(mq|m2)\b` | `comma_to_dot`, `to_float` |
| `durata_giorni` | int | `\b(\d{1,3})\s*(giorni?|(?:gg|giorni))\b`<br>`\b(\d{1,3})\s*(giorni?|gg)\b` | `to_int` |
| `baraccamenti_moduli_n` | int | `\b(\d{1,3})\s*modul[oi]\b` | `to_int` |

## Cantierizzazioni|Noli
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_mezzo` | enum | `\bgru\s*torr?e|autogr[uù]|piattaforma\s*aerea|miniescavatore|escavatore|sollevatore\s*telescopico\b` | `lower` |
| `portata_t` | float | `\b(\d{1,3}(?:[.,]\d)?)\s*t\b` | `comma_to_dot`, `to_float` |
| `durata_giorni` | int | `\b(\d{1,3})\s*giorni?\b` | `to_int` |

## Cantierizzazioni|Pulizie di cantiere
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `superficie_m2` | float | `\b(\d{2,6})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2)\b`<br>`\b(\d{2,6})\s*(mq|m2)\b` | `comma_to_dot`, `to_float` |
| `tipologia_pulizia` | enum | `\bsgrosso\b|\bfino\b|\bpost[- ]demolizione\b|\bpost[- ]posa\b` | `lower` |
| `numero_passaggi` | int | `\b(\d)\s*passa(?:gg|giorni)[i]\b`<br>`\b(\d)\s*passagg[i]\b` | `to_int` |

## Condotti e canne fumarie|Canne Shunt
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `sezioni_servite_n` | int | `\b(\d{1,2})\s*sezion[i]\b` | `to_int` |
| `altezza_condotto_m` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Condotti e canne fumarie|Canne fumarie
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `diametro_mm` | float | `\bØ\s*(\d{2,3})\s*cm\b|\bdiametro\s*(\d{2,3})\s*cm\b`<br>`\bØ\s*(\d{2,3})\s*mm\b|\bdiametro\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Condotti e canne fumarie|Canne per esalazioni
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `diametro_mm` | float | `\bØ\s*(\d{2,3})\s*cm\b|\bdiametro\s*(\d{2,3})\s*cm\b`<br>`\bØ\s*(\d{2,3})\s*mm\b|\bdiametro\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `destinazione` | enum | `\bcucine|bagni|autorimess[ea]|laboratori\b` | `lower` |

## Condotti e canne fumarie|Comignoli e pezzi speciali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bcomignol[oi]|eolico|antipio(?:gg|giorni)ia|curv[ae]|tee[s]?|riduzion[ei]\b`<br>`\bcomignol[oi]|eolico|antipioggia|curv[ae]|tee[s]?|riduzion[ei]\b` | `lower` |
| `sezione_passaggio_cm2` | float | `\b(\d{2,4})\s*cm2\b|\bsezione\s*(\d{2,4})\s*cm\^?2\b` | `comma_to_dot`, `to_float` |

## Controsoffitti|Botole d'ispezione e accessori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `dimensione_cm` | text | `\b\d{2,3}\s*[x×]\s*\d{2,3}\s*cm\b` | `split_structured_list` |
| `tenuta_fuoco` | enum | `\bEI\s?(30|60|90)\b` | `lower` |

## Controsoffitti|Controsoffitti a Baffles e ispezionabili
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `passo_mm` | float | `\bpasso\s*(\d{2,3})\s*cm\b`<br>`\bpasso\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `altezza_baffle_mm` | float | `\b(h|altezza)\s*(\d{2,3})\s*cm\b`<br>`\b(h|altezza)\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Controsoffitti|Controsoffitti a doghe in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Controsoffitti|Controsoffitti in PVC o materiali plastici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Controsoffitti|Controsoffitti in PVC o plastici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Controsoffitti|Controsoffitti in altri materiali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Controsoffitti|Controsoffitti in cartongesso
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_lastra_mm` | float | `\b(\d{1,3}(?:[.,]\d+)?)\s*(?:mm|cm)\b` | `comma_to_dot`, `to_float`, `if_cm_to_mm` |

## Controsoffitti|Controsoffitti in fibre minerali e acustici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `dimensione_pannello` | text | `\b60\s*[x×]\s*60\s*cm\b|\b120\s*[x×]\s*60\s*cm\b` | — |
| `prestazione_acustica` | enum | `\bαw\s*0\.(6|7|8|9)|\bαw\s*1\.0\b` | `lower` |

## Controsoffitti|Controsoffitti metallici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Controsoffitti|Velette di raccordo
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Demolizioni e rimozioni|Demolizione di fabbricati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `metodo` | enum | `\bmeccanic[ao]|manuale|pinza|martello\s*demolitore|filo\s*di\s*taglio\b` | `lower` |
| `volume_m3` | float | `\b(\d{2,5})\s*(?:m3|m³|metri\\s*cub(?:i|ici))\b|\b(\d{2,5})\s*mc\b`<br>`\b(\d{2,5})\s*m3\b|\b(\d{2,5})\s*mc\b` | `comma_to_dot`, `to_float` |

## Demolizioni e rimozioni|Demolizione elementi civili
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `quantita_unita` | float | `\b(\d{1,5})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2|ml|pz)\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|(?:pz|pz\\.|pezzi))\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|pz)\b` | `comma_to_dot`, `to_float`, `normalize_unit_symbols` |
| `spessore_cm` | float | `\bspessore\s*(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Demolizioni e rimozioni|Demolizione elementi strutturali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{2,3})\s*cm\b` | `comma_to_dot`, `to_float` |
| `metodo` | enum | `\bcarota(?:gg|giorni)i?o|taglio\s*(a\s*)?disco|filo\s*di\s*taglio|martellone|idrodemolizion[ei]\b`<br>`\bcarotaggi?o|taglio\s*(a\s*)?disco|filo\s*di\s*taglio|martellone|idrodemolizion[ei]\b` | `lower` |

## Demolizioni e rimozioni|Oneri per trasporto e discarica
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `CER_codice` | text | `\bCER\s*\d{2}\s*\d{2}\s*\d{2}\b|\b\d{2}\s*\d{2}\s*\d{2}\b` | — |
| `quantita_t` | float | `\b(\d{1,4}(?:[.,]\d)?)\s*t\b|\b(\d{1,4}(?:[.,]\d)?)\s*tonnellat[ae]\b` | `comma_to_dot`, `to_float` |
| `impianto_autorizzato` | bool | `\bautorizzat[oa]\b|\bAIA\b|\bEND\b` | `map_yes_no_multilang` |

## Demolizioni e rimozioni|Rimozione di impianti tecnologici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_impianto` | enum | `\belettric[oi]|idric[oi]|hvac|climatizzazione|gas|speciali|fotovoltaic[oi]\b` | `lower` |
| `quantita_unita` | float | `\b(\d{1,5})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2|ml|pz)\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|(?:pz|pz\\.|pezzi))\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|pz)\b` | `comma_to_dot`, `to_float`, `normalize_unit_symbols` |
| `bonifica_preliminare` | bool | `\bbonific[ae]\b|\bsvuotamento\b|\bneutralizzazion[ei]\b` | `map_yes_no_multilang` |

## Impianti elevatori|Allestimenti e personalizzazioni di impianti elevatori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_allestimento` | enum | `\brivestiment[oi]|paviment[oi]|soffitt[oi]|corriman[oi]|illuminazione\b` | `lower` |

## Impianti elevatori|Impianti ascensori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `portata_kg` | float | `\b(\d{3,4})\s*kg\b` | `comma_to_dot`, `to_float` |
| `corsa_m` | float | `\bcorsa\s*(\d{1,2})\s*m\b` | `comma_to_dot`, `to_float` |
| `velocita_m_s` | float | `\b(0\.[3-9]|[1-2](?:\.[0-5])?)\s*m/s\b` | `comma_to_dot`, `to_float` |

## Impianti elevatori|Montacarichi
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `portata_kg` | float | `\b(\d{3,4})\s*kg\b` | `comma_to_dot`, `to_float` |
| `dimensioni_cabina_cm` | text | `\b(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm\b` | `split_structured_list` |

## Impianti elevatori|Montapersone
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `portata_kg` | float | `\b(\d{2,3})\s*kg\b` | `comma_to_dot`, `to_float` |
| `corsa_m` | float | `\bcorsa\s*(\d{1,2})\s*m\b` | `comma_to_dot`, `to_float` |
| `velocita_m_s` | float | `\b0\.[1-6]\s*m/s\b` | `comma_to_dot`, `to_float` |

## Impianti elevatori|Piattaforme elevatrici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `portata_kg` | float | `\b(\d{2,3,4})\s*kg\b` | `comma_to_dot`, `to_float` |
| `corsa_m` | float | `\bcorsa\s*(\d{1,2})\s*m\b` | `comma_to_dot`, `to_float` |

## Impianti elevatori|Scale mobili
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `larghezza_gradinata_mm` | float | `\b(600|800|1000)\s*cm\b`<br>`\b(600|800|1000)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `velocita_m_s` | float | `\b0\.(45|5|65|75)\s*m/s\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Cappe di completamento
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Massetti alleggeriti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `densita_kgm3` | float | `\b(\d{3,4})\s*kg/?(?:m3|m³|metri\\s*cub(?:i|ici))\b`<br>`\b(\d{3,4})\s*kg/?m3\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Massetti pendenzati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `pendenza_%` | float | `\b(\d(?:[.,]\d)?)\s*%\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Sottofondi pavimentazioni
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Teli di separazione
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `grammatura_gm2` | float | `\b(\d{2,3})\s*g/?m2\b` | `comma_to_dot`, `to_float` |

## Massetti, sottofondi, drenaggi, vespai|Vespai aerati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_cm` | float | `\b(\d{2})\s*cm\b|\bH\s*(\d{2})\s*cm\b` | `comma_to_dot`, `to_float` |

## Movimenti di terra|Rinterri e forniture di terreno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `compattazione_Proctor_%` | float | `\b(Proctor|SPD)\s*(\d{2})\s*%\b|\b95\s*%\s*Mod\.?\s*Proctor\b` | `comma_to_dot`, `to_float` |

## Movimenti di terra|Scavi e trasporti a discarica
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `volume_scavo_m3` | float | `\b(\d{1,6})\s*(?:m3|m³|metri\\s*cub(?:i|ici))\b|\b(\d{1,6})\s*mc\b`<br>`\b(\d{1,6})\s*m3\b|\b(\d{1,6})\s*mc\b` | `comma_to_dot`, `to_float` |
| `profondita_m` | float | `\bprofondit[aà]\s*(\d(?:[.,]\d)?)\s*m\b|\bscavo\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Opere da cartongessista|Accessori per cartongessi
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_accessorio` | enum | `\bbotol[ae]|paraspigolo|staffe|pendinatura|tassell[oi]|nastr[oi]\s*giunto\b` | `lower` |

## Opere da cartongessista|Contropareti in cartongesso resistente al fuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EI_min` | enum | `\bEI\s?(30|60|90|120)\b` | `format_EI_from_last_int`, `to_ei_class` |
| `lastre_per_lato` | int | `\b([1-3])\s*lastre\b|\b(doppia|tripla)\s+lastra\b` | `to_strati_count` |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da cartongessista|Contropareti in cartongesso standard e idrorepellente
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `intercapedine_isolata` | bool | `\b(lana\s+di\s+roccia|lana\s+di\s+vetro|EPS|XPS|PIR)\b` | `map_yes_no_multilang` |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da cartongessista|Contropareti in lastre di fibrocemento
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_lastra_mm` | float | `\b(\d{1,3}(?:[.,]\d+)?)\s*(?:mm|cm)\b` | `comma_to_dot`, `to_float`, `if_cm_to_mm` |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da cartongessista|Pareti in cartongesso resistente al fuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EI_min` | enum | `\bEI\s?(30|60|90|120)\b` | `format_EI_from_last_int`, `to_ei_class` |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |
| `lastre_per_lato` | int | `\b([1-3])\s*lastre\b|\b(doppia|tripla)\s+lastra\b` | `to_strati_count` |

## Opere da cartongessista|Pareti in cartongesso standard e idrorepellente
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |
| `lastre_per_lato` | int | `\b([1-4])\s*lastre\b|\b(doppia|tripla|quadrupla)\s+lastra\b` | `to_strati_count` |
| `spessore_lastra_mm` | float | `\b(\d{1,3}(?:[.,]\d+)?)\s*(?:mm|cm)\b` | `comma_to_dot`, `to_float`, `if_cm_to_mm` |

## Opere da cartongessista|Pareti in lastre di fibrocemento
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_lastra_mm` | float | `\b(\d{1,3}(?:[.,]\d+)?)\s*(?:mm|cm)\b` | `comma_to_dot`, `to_float`, `if_cm_to_mm` |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da cartongessista|Setto autoportante cartongesso resistente al fuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |
| `altezza_setto_m` | float | `\b(h|altezza)\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Opere da cartongessista|Setto autoportante in cartongesso standard e idrorepellente
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da cartongessista|Setto autoportante in lastre di fibrocemento
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_orditura_mm` | enum | `\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*cm\b`<br>`\b(U|CW|UW)\s?(50|75|100|125)\b|\bprofil[oi]\s*(50|75|100|125)\s*mm\b` | `lower` |

## Opere da fabbro|Cancelli e recinzioni
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_m` | float | `\b(h|altezza)\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Opere da fabbro|Carpenterie metalliche
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `profilo` | enum | `\b(IPE|HEA|HEB|UPN)\b|\bangolar[ei]|tubolar[ei]\b` | `lower` |
| `classe_acciaio` | enum | `\bS(235|275|355)\b` | `lower` |
| `trattamento_protettivo` | enum | `\bzincatura|verniciatur[ae]|intumescent[ei]\b` | `lower` |

## Opere da fabbro|Grigliati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `maglia_mm` | text | `\b(\d{1,2})\s*[x×]\s*(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*[x×]\s*(\d{1,2})\s*mm\b` | `split_structured_list` |
| `spessore_piattina_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da fabbro|Parapetti metallici, ringhiere e inferriate
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_parapetto_mm` | float | `\b(9\d{2}|1[01]\d{2}|1200)\s*cm\b`<br>`\b(9\d{2}|1[01]\d{2}|1200)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `tamponamento` | enum | `\bbarre|vetro|lamiera\s*forata|rete\b` | `lower` |

## Opere da fabbro|Portoni metallici e porte basculanti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `dimensione_luce_cm` | text | `\b(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm\b` | `split_structured_list` |
| `motorizzazione` | enum | `\bmotorizzat[oa]|manuale|BMS\b` | `lower` |

## Opere da fabbro|Trattamenti per strutture in acciaio
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_secco_um` | float | `\b(\d{2,4})\s*µm\b|\b(\d{2,4})\s*micron\b` | `comma_to_dot`, `to_float` |
| `classe_resistenza_fuoco` | enum | `\bR(30|60|90|120)\b` | `lower`, `format_EI_from_last_int`, `to_ei_class` |

## Opere da facciatista e da cappottista|Cappotti termici finiti a intonachino
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da facciatista e da cappottista|Cappotti termici finiti con rivestimenti ceramici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Opere da facciatista e da cappottista|Facciata vetrata a doppia pelle
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Opere da facciatista e da cappottista|Facciata vetrata montanti e traversi
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `prestazione_termica_UW` | float | `\bU[wf]?\s*=?\s*(0\.[6-9]|1\.[0-9]|2\.[0-5])\b` | `comma_to_dot`, `to_float` |

## Opere da facciatista e da cappottista|Facciata vetrata riportata a cellule
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Opere da facciatista e da cappottista|Sistemi di facciata prefabbricati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `prestazione_termica_U` | float | `\bU\s*=?\s*0\.(1\d|[2-9]\d?)\b` | `comma_to_dot`, `to_float` |

## Opere da facciatista e da cappottista|Sistemi di facciata ventilata
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_isolante_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da falegname|Boiserie in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_pannello_mm` | float | `\b(1\d|2[0-5])\s*cm\b`<br>`\b(1\d|2[0-5])\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da falegname|Opere in legno custom
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `dimensioni_notevoli` | bool | `\bfuori\s*standard|oversize|su\s*misura\b` | `map_yes_no_multilang` |

## Opere da falegname|Persiane e scuri in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bpersian[ae]|scur[oi]|grigliat[oi]|orientabil[ei]\b` | `lower` |

## Opere da falegname|Porte in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_anta_mm` | float | `\b(3[8-9]|[4-5]\d|60)\s*cm\b`<br>`\b(3[8-9]|[4-5]\d|60)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da florovivaista|Alberature (prima seconda e terza grandezza)
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `categoria_grandezza` | enum | `\b(I|II|III)\s*grandezz?a\b` | `lower` |
| `altezza_piante_m` | float | `\b(h|altezza)\s*(\d{1,2}(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |
| `circonferenza_fusto_cm` | float | `\b(circonferenza|circ\.)\s*(\d{2,3})\s*cm\b|\bC(\d{2,3})\b` | `comma_to_dot`, `to_float` |

## Opere da florovivaista|Altro materiale vegetale (terricci pacciamature)
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `confezione_litri` | float | `\b(\d{2,4})\s*L\b|\b(\d{2,4})\s*litri\b` | `comma_to_dot`, `to_float` |

## Opere da florovivaista|Arbusti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_piante_cm` | float | `\b(h|altezza)\s*(\d{2,3})\s*cm\b` | `comma_to_dot`, `to_float` |
| `contenitore_litri` | float | `\b(\d{1,2}|[1-8]\d)\s*L\b|\b(\d{1,2}|[1-8]\d)\s*litri\b` | `comma_to_dot`, `to_float` |
| `sempreverde` | bool | `\bsempreverde\b|\bcaduc[ao]\b` | `map_yes_no_multilang` |

## Opere da florovivaista|Tappezzanti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `densita_impianto_pt_m2` | float | `\b(\d{1,2})\s*p[tz]\s*/\s*m2\b|\b(\d{1,2})\s*\b[pP]iante\s*/\s*m2\b` | `comma_to_dot`, `to_float` |
| `altezza_cm` | float | `\b(\d{1,2})\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere da intonacatore e stuccatore|Intonaco intumescente
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_resistenza_fuoco` | enum | `\bR(30|60|90|120)\b` | `lower`, `format_EI_from_last_int`, `to_ei_class` |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da intonacatore e stuccatore|Intonaco per esterno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `finitura` | enum | `\bgraffiato|fratazzato|lisciato\b` | `lower` |

## Opere da intonacatore e stuccatore|Intonaco per interno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `finitura` | enum | `\bfratazzato|lisciato|rasato\b` | `lower` |

## Opere da lattoniere|Canali di gronda
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `sezione` | enum | `\bsemi?circolar[ei]|quadra|ogivale\b` | `lower` |
| `sviluppo_lamiera_mm` | float | `\bsviluppo\s*(\d{3})\s*cm\b|\b(\d{3})\s*cm\s*sviluppo\b`<br>`\bsviluppo\s*(\d{3})\s*mm\b|\b(\d{3})\s*mm\s*sviluppo\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da lattoniere|Pezzi speciali per lattonerie
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bangol[oi]|dilatazion[ei]|testat[ei]|imbocch[io]i|raccord[oi]|coprigiunt[oi]\b` | `lower` |
| `sviluppo_lamiera_mm` | float | `\bsviluppo\s*(\d{2,3})\s*cm\b`<br>`\bsviluppo\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da lattoniere|Scossaline
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `sviluppo_lamiera_mm` | float | `\bsviluppo\s*(\d{2,3})\s*cm\b`<br>`\bsviluppo\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `lunghezza_barra_m` | float | `\b(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Opere da lattoniere|Tubi pluviali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `sezione` | enum | `\btond[ao]|quadr[ao]|rettangolar[ei]\b` | `lower` |
| `diametro_mm` | float | `\bØ\s*(\d{2,3})\s*cm\b|\bdiametro\s*(\d{2,3})\s*cm\b`<br>`\bØ\s*(\d{2,3})\s*mm\b|\bdiametro\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da serramentista|Avvolgibili, controtelai, cassonetti e persiane
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bavvolgibil[ei]|persian[ae]|cassonett[oi]|controtelai[oi]|scur[oi]\b` | `lower` |
| `motorizzazione` | enum | `\bmotore\b|\bradio\b|\bmanuale\b|\bBMS\b` | `lower` |

## Opere da serramentista|Porte blindate, portoni e bussole
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_antieffrazione` | enum | `\bRC[2-4]\b` | `lower` |
| `trasmittanza_UW` | float | `\bU[wW]?\s*=?\s*(0\.[7-9]|1\.[0-9]|2\.[0-5])\b` | `comma_to_dot`, `to_float` |

## Opere da serramentista|Porte metalliche
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_lamiera_mm` | float | `\b(0\.[8-9]|1\.[0-9])\s*cm\b|\bspessore\s*(\d(?:\.\d)?)\s*cm\b`<br>`\b(0\.[8-9]|1\.[0-9])\s*mm\b|\bspessore\s*(\d(?:\.\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da serramentista|Porte tagliafuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EI_min` | enum | `\bEI\s?(30|60|90|120)\b` | `format_EI_from_last_int`, `to_ei_class` |
| `omologazione` | bool | `\bomologat[oa]|certificat[oa]\b` | `map_yes_no_multilang` |

## Opere da serramentista|Serramenti in PVC
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `trasmittanza_UW` | float | `\bU[wW]\s*=?\s*(0\.[7-9]|1\.[0-9])\b` | `comma_to_dot`, `to_float` |
| `vetro_Ug` | float | `\bU[gG]\s*=?\s*(0\.[5-9]|1\.[0-3])\b` | `comma_to_dot`, `to_float` |

## Opere da serramentista|Serramenti in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `trasmittanza_UW` | float | `\bU[wW]\s*=?\s*(0\.[8-9]|1\.[0-9])\b` | `comma_to_dot`, `to_float` |
| `vetro_Ug` | float | `\bU[gG]\s*=?\s*(0\.[5-9]|1\.[0-3])\b` | `comma_to_dot`, `to_float` |

## Opere da serramentista|Serramenti in legno e alluminio
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `trasmittanza_UW` | float | `\bU[wW]\s*=?\s*(0\.[7-9]|1\.[0-7])\b` | `comma_to_dot`, `to_float` |

## Opere da serramentista|Serramenti metallici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `materiale` | enum | `\balluminio\s*a?\s*taglio\s*termico|acciaio\s*a?\s*taglio\s*termico|ferro\s*freddo\b` | `lower` |
| `trasmittanza_UW` | float | `\bU[wW]\s*=?\s*(0\.[8-9]|[1-2](?:\.[0-9])?)\b` | `comma_to_dot`, `to_float` |

## Opere da serramentista|Sistemi di partizione trasparenti, porte e parapetti vetrati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_vetro_mm` | float | `\b(8|10|12|16|\d{2})\s*cm\b|\b(44\.[1-2])\b`<br>`\b(8|10|12|16|\d{2})\s*mm\b|\b(44\.[1-2])\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `classe_sicurezza` | enum | `\b[12]B[12]\b` | `lower` |

## Opere da tappezziere|Carta da parati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `reazione_al_fuoco` | enum | `\b(B|C)-s[12],d0\b` | `lower`, `split_structured_list` |

## Opere da tappezziere|Tende da interno manuali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\brullo|pacchetto|pannello|plissettat[ae]|veneziana\b` | `lower` |
| `larghezza_luce_m` | float | `\b(larghezza|luce)\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Opere da tappezziere|Tende da interno motorizzate
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `larghezza_luce_m` | float | `\b(larghezza|luce)\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |
| `automazione` | enum | `\bradio\b|\bBMS\b|\binterruttore\b|\bfilo\s*bus\b` | `lower` |

## Opere da verniciatore|Lavorazioni decorative
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `manodopera_strati` | int | `\b(\d)\s*man[ií]?\b|\b(\d)\s*strat[oi]\b` | `to_int` |

## Opere da verniciatore|Preparazione delle superfici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `grado_preparazione` | enum | `\bsabbiatura\s*SA\s*2\.?5|carte(?:gg|giorni)iat[oa]|spazzolatura|lava(?:gg|giorni)i?o\b`<br>`\bsabbiatura\s*SA\s*2\.?5|carteggiat[oa]|spazzolatura|lavaggi?o\b` | `lower` |
| `primer` | enum | `\bprimer\b|\bfondo\b|\banticorrosiv[oi]\b` | `lower` |

## Opere da verniciatore|Tinteggiature intumescenti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_resistenza_fuoco` | enum | `\bR(30|60|90|120)\b` | `lower`, `format_EI_from_last_int`, `to_ei_class` |
| `spessore_secco_um` | float | `\b(\d{3,4})\s*µm\b|\b(\d{3,4})\s*micron\b` | `comma_to_dot`, `to_float` |
| `sistema_omologato` | bool | `\bomologat[oa]|certificat[oa]\b` | `map_yes_no_multilang` |

## Opere da verniciatore|Tinteggiature su agglomerati edili (murature, cartongessi ecc.)
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `ciclo_mani` | int | `\b(\d)\s*man[ií]?\b|\b(\d)\s*strat[oi]\b` | `to_int` |

## Opere da verniciatore|Tinteggiature su legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `ciclo` | enum | `\bimpregnante\b|\bfondo\s*\+\s*finitura\b|\boliatur[ae]\b|\bvernice\b` | `lower` |
| `protezione_esterna` | bool | `\bestern[oi]|UV|marino\b` | `map_yes_no_multilang` |

## Opere da verniciatore|Tinteggiature su metallo
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_secco_tot_um` | float | `\b(\d{2,3})\s*µm\b` | `comma_to_dot`, `to_float` |
| `ambiente_corrosivita` | enum | `\bC[2-5]\b` | `lower` |

## Opere da vetraio|Lavorazioni su vetri e serramenti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_vetro` | enum | `\bfloat|extrachiaro|temprat[oa]|stratificat[oa]|camera\b` | `lower` |
| `spessore_vetro_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere da vetraio|Pensiline vetrate
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `vetro_stratig_mm` | text | `\b(\d{2})\+(\d{2})(?:\+(\d{2}))?\s*cm\b`<br>`\b(\d{2})\+(\d{2})(?:\+(\d{2}))?\s*mm\b` | — |
| `sporgenza_cm` | float | `\b(\d{2,3})\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere da vetraio|Vetrazioni e accessori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_tot_mm` | float | `\b(\d{2})\s*cm\b|\b(\d{2})-\d{2}-\d{2}\b`<br>`\b(\d{2})\s*mm\b|\b(\d{2})-\d{2}-\d{2}\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di bonifica e analisi di laboratorio|Altre attività di bonifica e analisi di laboratorio
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Opere di bonifica e analisi di laboratorio|Bonifica materiali pericolosi
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `materiale_pericoloso` | enum | `\bamianto|eternit|piombo|PCB|idrocarburi\b` | `lower` |
| `metodo_bonifica` | enum | `\brimozion[ei]|incapsulament[oi]|confinament[oi]\b` | `lower` |

## Opere di coibentazione|Isolanti acustici
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `Rw_dB` | float | `\bR[wW]?\s*=?\s*(\d{2})\s*dB\b` | `comma_to_dot`, `to_float` |

## Opere di coibentazione|Isolanti termici in copertura
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `lambda_WmK` | float | `[λΛ]\s*=?\s*0[.,]\d{3}\b|\b0[.,]\d{3}\s*W/?mK\b` | `comma_to_dot`, `to_float` |
| `spessore_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di coibentazione|Isolanti termici su solai e pareti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `lambda_WmK` | float | `[λΛ]\s*=?\s*0[.,]\d{3}\b|\b0[.,]\d{3}\s*W/?mK\b` | `comma_to_dot`, `to_float` |
| `spessore_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di impermeabilizzazione|Accessori per l'impermeabilizzazione
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_accessorio` | enum | `\bbocchettone|parafango|angolar[ei]|scossalina|sfiato|parapassat[ae]\b` | `lower` |

## Opere di impermeabilizzazione|Barriere al vapore
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `grammatura_gm2` | float | `\b(\d{2,3})\s*g/?m2\b|\b(\d{2,3})\s*g\s*m-?2\b` | `comma_to_dot`, `to_float` |

## Opere di impermeabilizzazione|Impermeabilizzazioni bituminose
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `strati` | int | `\b(\d)\s*strat[oi]\b` | `to_int` |
| `spessore_tot_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di impermeabilizzazione|Impermeabilizzazioni liquide
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `chimica` | enum | `\bPMMA|poliuretanica|epossidic[ae]|cementizi[ae]\b` | `lower` |
| `consumo_kgm2` | float | `\b(\d(?:[.,]\d)?)\s*kg/?m2\b` | `comma_to_dot`, `to_float` |

## Opere di impermeabilizzazione|Impermeabilizzazioni resine
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `resina_tipo` | enum | `\bPMMA\b|\bPU\b|\bEP\b|poliuretanica|epossidic[ae]` | `lower` |
| `spessore_tot_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di impermeabilizzazione|Impermeabilizzazioni sintetiche
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `membrana` | enum | `\bPVC\b|\bTPO\b|\bEPDM\b` | `lower` |
| `spessore_mm` | float | `\b(1\.[2-9]|2\.[0-4])\s*cm\b`<br>`\b(1\.[2-9]|2\.[0-4])\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `fissaggio` | enum | `\b(meccanico|incollato|zavorrato)\b` | `lower` |

## Opere di pavimentazione|Pavimenti in altri materiali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b`<br>`\b(\d{1,2}(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `posa` | enum | `\bincollat[oa]|flottant[ei]|meccanic[oa]|gettata\s*in\s*opera\b` | `lower` |

## Opere di pavimentazione|Pavimenti in gomma o PVC
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `formato` | enum | `\brotol[oi]|quadrott[ei]|doghe|\bLVT\b` | `lower` |
| `spessore_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `strato_usura_mm` | float | `\bstrato\s*usura\s*(0\.[2-9]\d?)\s*cm\b`<br>`\bstrato\s*usura\s*(0\.[2-9]\d?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di pavimentazione|Pavimenti in gres e ceramica
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `formato` | text | `\b\d{2,3}\s*[x×]\s*\d{2,3}\s*(cm|mm)\b` | `split_structured_list` |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `classe_antiscivolo` | enum | `\bR(9|10|11|12|13)\b` | `lower` |

## Opere di pavimentazione|Pavimenti in legno e laminato
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_legno` | enum | `\bprefinito|multistrato|massello|laminato\b` | `lower` |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `posa` | enum | `\bincollat[oa]|flottant[ei]|chiodat[oa]|\bclic\b` | `lower` |

## Opere di pavimentazione|Pavimenti in moquette e zerbini
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `formato` | enum | `\btelo|quadrott[ei]|plank\b` | `lower` |
| `spessore_tot_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di pavimentazione|Pavimenti in pietra
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_pietra` | enum | `\bmarmo|granito|travertino|ardesia|quarzite|basalto|pietra\s*calcarea\b` | `lower` |
| `spessore_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere di pavimentazione|Pavimenti industriali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_calcestruzzo` | enum | `\bRck\s*(25|30|35)|fibro\b` | `lower` |
| `spessore_cm` | float | `\b(\d{1,2})\s*cm\b` | `comma_to_dot`, `to_float` |
| `indurente_quarzo` | bool | `\bindurente\s*(al)?\s*quarzo\b` | `map_yes_no_multilang` |

## Opere di pavimentazione|Pavimenti sopraelevati e flottanti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_strutturale_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di pavimentazione|Zoccolini e accessori per pavimentazioni
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di rivestimento|Altri rivestimenti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b`<br>`\b(\d{1,2}(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `finitura_superficie` | enum | `\b(opaca|satinata|lucida|strutturata)\b` | `lower` |

## Opere di rivestimento|Rivestimenti in gomma o PVC
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `formato` | enum | `\brotol[oi]|quadrott[ei]|doghe|\bLVT\b` | `lower` |
| `spessore_mm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b`<br>`\b(\d(?:[.,]\d)?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `strato_usura_mm` | float | `\bstrato\s*usura\s*(0\.[2-9]\d?)\s*cm\b`<br>`\bstrato\s*usura\s*(0\.[2-9]\d?)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Opere di rivestimento|Rivestimenti in gres e ceramica
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `formato` | text | `\b\d{2,3}\s*[x×]\s*\d{2,3}\s*(cm|mm)\b` | `split_structured_list` |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `classe_antiscivolo` | enum | `\bR(9|10|11|12|13)\b` | `lower` |

## Opere di rivestimento|Rivestimenti in legno
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_legno` | enum | `\bprefinito|multistrato|massello|lamellare\b` | `lower` |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `posa` | enum | `\bincollat[oa]|flottant[ei]|chiodat[oa]\b` | `lower` |

## Opere di rivestimento|Rivestimenti in pietra
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipo_pietra` | enum | `\bmarmo\b|\bgranito\b|\btravertino\b|\bardesia\b|\bquarzite\b|\bbasalto\b|\bpietra\s*calcarea\b` | `lower` |
| `spessore_lastre_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `finitura_superficie` | enum | `\blucidat[oa]|levigat[oa]|bocciardat[oa]|spazzolat[oa]|fiacmat[oa]|sabatat[oa]|anticat[oa]\b`<br>`\blucidat[oa]|levigat[oa]|bocciardat[oa]|spazzolat[oa]|fiammat[oa]|sabatat[oa]|anticat[oa]\b` | `lower` |

## Opere di sicurezza|Apparecchi di sicurezza (reti, cartellonistica ecc.)
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `estensione` | float | `\b(\d{1,5})\s*((?:mq|m²|m2|metri\\s*quad(?:ri|rati))|m2|ml|pz)\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|(?:pz|pz\\.|pezzi))\b`<br>`\b(\d{1,5})\s*(mq|m2|ml|pz)\b` | `comma_to_dot`, `to_float` |
| `classe_norma` | text | `\bEN\s*13374|UNI|D\.Lgs\.?\s*81/08\b` | `split_structured_list` |

## Opere di sicurezza|Baraccamenti
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `moduli_n` | int | `\b(\d{1,3})\s*modul[oi]\b` | `to_int` |
| `dotazioni` | enum | `\buffici|spogliatoi|servizi|mensa|deposito\b` | `lower` |

## Opere di sicurezza|Mezzi di cantiere
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_mezzo` | enum | `\bgru\b|\bPLE\b|sollevatore|escavator[ei]|miniescavator[ei]|autocarro\b` | `lower` |
| `durata_giorni` | int | `\b(\d{1,3})\s*giorni?\b` | `to_int` |

## Opere di sicurezza|Opere Provvisionali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_lavoro_m` | float | `\b(h|altezza)\s*(\d{1,2}(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |
| `classe_carico_ponti` | enum | `\bClasse\s*[1-5]\b|\bUNI\s*EN\s*12811\b` | `lower` |

## Opere in pietra naturale|Copertine e pezzi speciali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere in pietra naturale|Davanzali e soglie
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere in pietra naturale|Materiali semilavorati (sola fornitura)
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_lastre_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `formato_lastra_cm` | text | `\b(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm\b` | `split_structured_list` |

## Opere in pietra naturale|Rivestimenti di scale
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_alzata_cm` | float | `\balzata\s*(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `spessore_pedata_cm` | float | `\bpedata\s*(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere murarie|Blocchi in calcestruzzo cellulare aerato autoclavato
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `densita_kgm3` | float | `\b(\d{3})\s*kg/?(?:m3|m³|metri\\s*cub(?:i|ici))\b|\bρ\s*=\s*(\d{3})\b`<br>`\b(\d{3})\s*kg/?m3\b|\bρ\s*=\s*(\d{3})\b` | `comma_to_dot`, `to_float` |
| `classe_resistenza` | enum | `\b(2\.5|3\.5|5\.0)\b` | `lower` |

## Opere murarie|Blocchi in calcestruzzo vibrocompresso
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `percentuale_foratura` | float | `\b(\d{1,2})\s*%\s*foratura\b` | `comma_to_dot`, `to_float` |
| `classe_resistenza` | enum | `\bR\s?(7\.5|10|12|15)\b` | `lower` |

## Opere murarie|Elementi in laterizio
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\b(forato|alveolato|pieno|porizzato|poroton)\b` | `lower` |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `percentuale_foratura` | float | `\b(\d{1,2})\s*%\s*foratura\b` | `comma_to_dot`, `to_float` |

## Opere murarie|Murature in altri materiali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `materiale` | enum | `\b(pietrame|pietra\s*naturale|sasso|tufo|calcestruzzo\s*pieno|legno\s*massello|adobe)\b` | `lower` |
| `spessore_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `tipo_legante` | enum | `\b(calce|cemento|terra\s*cruda|resine)\b` | `lower` |

## Opere stradali, fognature e sistemazioni esterne|Complementi edili per illuminazione esterna
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `altezza_palo_m` | float | `\b(h|altezza)\s*(\d{1,2}(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |
| `grado_protezione_IP` | enum | `\bIP(5[5-9]|6[5-8])\b` | `lower` |

## Opere stradali, fognature e sistemazioni esterne|Ghiaie sabbie e aggregati
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `granulometria_mm` | text | `\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*mm\b` | `split_structured_list` |

## Opere stradali, fognature e sistemazioni esterne|Manti stradali in asfalto e bitumi
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_strato_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `bitume_classe` | enum | `\b(50/70|70/100|modificato|PMB|MOD)\b` | `lower`, `split_structured_list` |

## Opere stradali, fognature e sistemazioni esterne|Marciapiedi e accessori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_allettamento_cm` | float | `\b(\d)\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere stradali, fognature e sistemazioni esterne|Massicciate stradali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_cm` | float | `\b(\d{2})\s*cm\b` | `comma_to_dot`, `to_float` |
| `portanza_Evd_MPa` | float | `\bEvd\s*(\d{2,3})\s*MPa\b` | `comma_to_dot`, `to_float` |

## Opere stradali, fognature e sistemazioni esterne|Pavimentazione in autobloccanti o masselli
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_massello_cm` | float | `\b(\d(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `schema_posa` | enum | `\bspina\s*di\s*pesce|corsi\s*diritt[ei]|a\s*cerchi|a\s*el{1,2}e\b` | `lower`, `split_structured_list` |

## Opere stradali, fognature e sistemazioni esterne|Segnaletica orizzontale e verticale
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_pellicola` | enum | `\bRA?\s?2|Classe\s*[123]\b` | `lower` |
| `larghezza_traccia_cm` | float | `\b(\d{1,2})\s*cm\b` | `comma_to_dot`, `to_float` |

## Opere stradali, fognature e sistemazioni esterne|Sistema di raccolta e smaltimento acque meteoriche
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `diametro_tubi_mm` | float | `\bØ\s*(\d{3})\s*cm\b|\bDN\s*(\d{2,3})\b`<br>`\bØ\s*(\d{3})\s*mm\b|\bDN\s*(\d{2,3})\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `portata_l_s` | float | `\b(\d{1,3}(?:[.,]\d)?)\s*l/?s\b|\b(\d{1,3}(?:[.,]\d)?)\s*L/s\b` | `comma_to_dot`, `to_float` |
| `griglia_classe_carico` | enum | `\b(A15|B125|C250|D400|E600|F900)\b` | `lower` |

## Pareti mobili, attrezzate, impacchettabili|Pareti impacchettabili
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `Rw_dB` | float | `\bR[wW]?\s*=?\s*(3[2-9]|4[0-9]|5[0-8])\s*dB\b` | `comma_to_dot`, `to_float` |
| `altezza_max_m` | float | `\b(h|altezza)\s*(\d(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Pareti mobili, attrezzate, impacchettabili|Pareti mobili opache
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `Rw_dB` | float | `\bR[wW]?\s*=?\s*(3[2-9]|4[0-9]|5[0-5])\s*dB\b` | `comma_to_dot`, `to_float` |
| `spessore_pannello_mm` | float | `\b(\d{2,3})\s*cm\b`<br>`\b(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Pareti mobili, attrezzate, impacchettabili|Pareti mobili vetrate
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_vetro_mm` | float | `\b(10|12|16|18|20|21)\s*cm\b`<br>`\b(10|12|16|18|20|21)\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `profili_visibili` | enum | `\btutto\s*vetro|minimale|standard\b` | `lower` |

## Presidi antincendio|Altri presidi antincendio
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |

## Presidi antincendio|Estintori
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `capacita_kg` | float | `\b(\d{1,2})\s*kg\b` | `comma_to_dot`, `to_float` |

## Presidi antincendio|Portoni tagliafuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EI_min` | enum | `\bEI(30|60|90|120)\b` | `format_EI_from_last_int`, `to_ei_class` |
| `dimensione_luce_cm` | text | `\b(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm\b` | `split_structured_list` |

## Presidi antincendio|Sigillature
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_mm` | float | `\b(\d{1,2})\s*cm\b`<br>`\b(\d{1,2})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |

## Presidi antincendio|Tende tagliafuoco
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EI_min` | enum | `\bEI(60|90|120)\b` | `format_EI_from_last_int`, `to_ei_class` |
| `larghezza_m` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*m\b` | `comma_to_dot`, `to_float` |

## Sistemi oscuranti per facciate|Schermature fisse e brisè soleil
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `passo_lamelle_mm` | float | `\bpasso\s*(\d{2,3})\s*cm\b`<br>`\bpasso\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
| `orientamento_lamelle` | enum | `\b(orizzontal[ei]|vertical[ei]|variabile)\b` | `lower` |

## Sistemi oscuranti per facciate|Schermature mobili
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_movimento` | enum | `\borientabil[i]|scorrevol[i]|impacchettabil[i]|avvolgibil[i]\b` | `lower` |
| `automazione` | enum | `\bmanuale|motorizzat[oa]|BMS\b` | `lower` |

## Sistemi oscuranti per facciate|Tende da sole e alla veneziana
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia` | enum | `\bvenezian[ae]|tenda\s*da\s*sole|zip\s*screen\b` | `lower` |
| `larghezza_luce_m` | float | `\b(larghezza|luce)\s*(\d(?:[.,]\d)?)\s*m\b|\b(\d(?:[.,]\d)?)\s*m\s*(?:di\s*)?luce\b` | `comma_to_dot`, `to_float` |

## Sistemi per verde pensile|Verde pensile estensivo
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_pacchetto_cm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b` | `comma_to_dot`, `to_float` |
| `peso_saturo_kNm2` | float | `\b(0\.[8-9]|1\.[0-8])\s*kN/?m2\b|\b(80-180)\s*kg/?m2\b` | `comma_to_dot`, `to_float` |
| `specie_vegetali` | enum | `\bsedum|erbacee|muscine[ee]|miste\b` | `lower` |

## Sistemi per verde pensile|Verde pensile intensivo
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `spessore_pacchetto_cm` | float | `\b(\d{2,3})\s*cm\b` | `comma_to_dot`, `to_float` |
| `peso_saturo_kNm2` | float | `\b(2\.[0-9]|[3-5]\.[0-9]|6\.0)\s*kN/?m2\b|\b(200-600)\s*kg/?m2\b` | `comma_to_dot`, `to_float` |
| `irrigazione` | enum | `\ba\s*goccia|spruzzo|centralina\b` | `lower` |

## Strutture in altri materiali|Strutture in altri materiali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `materiale` | enum | `\blamellar[ei]|massicci[oi]|acciaio\s*le(?:gg|giorni)ero|fibra\s*di\s*carbonio|composit[oi]|bamboo\b`<br>`\blamellar[ei]|massicci[oi]|acciaio\s*leggero|fibra\s*di\s*carbonio|composit[oi]|bamboo\b` | `lower` |
| `trattamento` | enum | `\bimpregnant[ei]|resina|verniciatur[ae]\b` | `lower` |

## Strutture in carpenteria metallica|Strutture in carpenteria metallica
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `profilo` | enum | `\b(IPE|HEA|HEB|UPN)\b|\bangolar[ei]|tubolar[ei]\b` | `lower` |
| `classe_acciaio` | enum | `\bS(235|275|355)\b` | `lower` |
| `trattamento` | enum | `\bzincatura|verniciatur[ae]|intumescent[ei]\b` | `lower` |

## Strutture in cemento armato in opera|Strutture in cemento armato in opera
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_calcestruzzo` | enum | `\bC(25/30|28/35|30/37|32/40|35/45|40/50)\b` | `lower`, `split_structured_list` |
| `armatura_tipo` | enum | `\bB450[AC]\b|\bfibra\b` | `lower` |
| `copriferro_cm` | float | `\bcopriferro\s*(\d)\s*cm\b` | `comma_to_dot`, `to_float` |

## Tetti, manti di copertura e opere accessorie|Accessi in copertura
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `dimensione_luce_cm` | text | `\b\d{2,3}\s*[x×]\s*\d{2,3}\s*cm\b` | `split_structured_list` |

## Tetti, manti di copertura e opere accessorie|Accessori per fotovoltaico
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_accessorio` | enum | `\bmorsett[oi]|staffe|binar[io]i|ganc[hi]\b|passacavo|fermapannello` | `lower` |
| `compatibilita_modulo` | enum | `\b30[-–]35\s*cm\b|\b40\s*cm\b|vetro[- ]vetro|universale\b`<br>`\b30[-–]35\s*mm\b|\b40\s*mm\b|vetro[- ]vetro|universale\b` | `lower` |

## Tetti, manti di copertura e opere accessorie|Dispositivi anticaduta e di protezione
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `classe_EN795` | enum | `\bEN\s*795\s*([A-E])\b|\bclasse\s*([A-E])\b` | `lower` |

## Tetti, manti di copertura e opere accessorie|Manti di copertura con elementi puntuali
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `tipologia_elemento` | enum | `\bcopp[io]i|tegole|ardesia|scandol[ae]\b` | `lower` |
| `pendenza_min_%` | float | `\bpendenza\s*min\.?\s*(\d{1,2})\s*%\b|\b(\d{1,2})\s*%\s*minima\b` | `comma_to_dot`, `to_float` |

## Tetti, manti di copertura e opere accessorie|Manti di copertura con pannelli o lastre
| Proprietà | Tipo | Regex | Normalizzatori |
| --- | --- | --- | --- |
| `materiale` | enum | `\blamiera\s*grecat[ae]|sandwich|policarbonato|fibrocemento\b` | `lower` |
| `spessore_mm` | float | `\b(\d{1,2}(?:[.,]\d)?)\s*cm\b|\bpannello\s*(\d{2,3})\s*cm\b`<br>`\b(\d{1,2}(?:[.,]\d)?)\s*mm\b|\bpannello\s*(\d{2,3})\s*mm\b` | `comma_to_dot`, `to_float`, `cm_to_mm?` |
