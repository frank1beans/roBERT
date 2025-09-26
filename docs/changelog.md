# Changelog

## 2025-02-16
- Rigenerati i dataset train/val in `data/train/classification/processed` con il registry ed estrattori aggiornati, confermando la coerenza delle distribuzioni label rispetto al run precedente.
- Aggiunto il report `property_coverage.json` con le metriche di copertura degli slot estratti per train e val.

## 2025-02-15
- Allineati gli estrattori delle proprietà con lo schema del dataset rimuovendo 80 slot non più referenziati per i super-ambiti "Massetti, sottofondi, drenaggi, vespai", "Tetti, manti di copertura e opere accessorie", "Opere di impermeabilizzazione", "Sistemi oscuranti per facciate" e "Controsoffitti".
- Aggiornato il report `data/properties/unused_properties_report.json` con il nuovo inventario degli `property_id` non coperti dal dataset di training.
