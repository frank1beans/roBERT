# Property Extraction Test Report

## Sample per category
```
robimb sample-categories --dataset data/train/classification/raw/dataset.jsonl --output outputs/dataset_per_category.jsonl
```

### CLI Output
```
{
  "output": "outputs/dataset_per_category.jsonl",
  "num_records": 146,
  "category_field": "cat"
}
```

## Property extraction on sampled dataset
```
robimb convert --train-file outputs/dataset_per_category.jsonl --ontology data/wbs/ontology.json --label-maps outputs/per_category/label_maps.json --out-dir outputs/per_category
```

### CLI Output
```
{
  "train_dataset": "outputs/per_category/train_processed.jsonl",
  "val_dataset": "outputs/per_category/val_processed.jsonl",
  "label_maps": "outputs/per_category/label_maps.json",
  "mask_matrix": "outputs/per_category/mask_matrix.npy",
  "mask_report": "outputs/per_category/mask_report.json",
  "reports_dir": "outputs/per_category/reports"
}
```

## Outputs
- `outputs/dataset_per_category.jsonl`
- `outputs/per_category/train_processed.jsonl`
- `outputs/per_category/val_processed.jsonl`
- `outputs/per_category/label_maps.json`
- `outputs/per_category/mask_matrix.npy`
- `outputs/per_category/mask_report.json`
- `outputs/per_category/reports/`

## Dataset Summary
```
{
  "train": {
    "num_records": 116,
    "avg_text_length": 782.0948275862069,
    "median_text_length": 647.0,
    "p95_text_length": 1871.0
  },
  "val": {
    "num_records": 30,
    "avg_text_length": 727.4333333333333,
    "median_text_length": 585.0,
    "p95_text_length": 1498.1499999999999
  }
}
```

## Property extraction on full training dataset
```
robimb convert --train-file data/train/classification/raw/dataset.jsonl --ontology data/wbs/ontology.json --label-maps outputs/label_maps.json --out-dir outputs
```

### CLI Output
```
{
  "train_dataset": "outputs/train_processed.jsonl",
  "val_dataset": "outputs/val_processed.jsonl",
  "label_maps": "outputs/label_maps.json",
  "mask_matrix": "outputs/mask_matrix.npy",
  "mask_report": "outputs/mask_report.json",
  "reports_dir": "outputs/reports"
}
```

### Dataset Summary
```
{
  "train": {
    "num_records": 39280,
    "avg_text_length": 847.1959521384929,
    "median_text_length": 640.0,
    "p95_text_length": 2320.0
  },
  "val": {
    "num_records": 9821,
    "avg_text_length": 851.1460136442317,
    "median_text_length": 643.0,
    "p95_text_length": 2367.0
  }
}
```

### Property extraction metrics
- Train set records with extracted properties: 28 106 / 39 280 (71.6%)
- Validation set records with extracted properties: 7 007 / 9 821 (71.3%)
- Most common extracted property slots (train):
  1. `opere_da_serramentista.__global__.modello` (4 333 matches)
  2. `apparecchi_sanitari_e_accessori.__global__.modello` (2 998 matches)
  3. `opere_da_serramentista.__global__.dimensioni_lxh_mm` (2 802 matches)
  4. `controsoffitti.__global__.modello` (2 763 matches)
  5. `opere_di_pavimentazione.__global__.modello` (2 477 matches)
- Most common extracted property slots (validation):
  1. `opere_da_serramentista.__global__.modello` (1 075 matches)
  2. `apparecchi_sanitari_e_accessori.__global__.modello` (716 matches)
  3. `opere_da_serramentista.__global__.dimensioni_lxh_mm` (683 matches)
  4. `controsoffitti.__global__.modello` (662 matches)
  5. `opere_di_pavimentazione.__global__.modello` (659 matches)
