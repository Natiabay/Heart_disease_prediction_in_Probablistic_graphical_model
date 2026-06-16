# Data — UCI Heart Disease (multi-source)

Combined datasets used for training and evaluation:

| Source | UCI file |
|--------|----------|
| Cleveland | `processed.cleveland.data` |
| Hungarian | `processed.hungarian.data` |
| Switzerland | `processed.switzerland.data` |
| VA Long Beach | `processed.va.data` |

Downloaded automatically from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) on first run.

**Target:** `num` → binary `HeartDisease` (No / Yes)  
**Features:** discretized clinical attributes (age, chest pain, cholesterol, etc.)

Raw files are cached under `data/raw/`. Processed CSV: `data/heart_disease_discretized.csv`.
