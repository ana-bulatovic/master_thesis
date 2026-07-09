# Master rad — NLP Pipeline (Master Thesis)

Projekat za primenu LLM modela (Ollama) na zadatke:
- detekcija sarkazma (iSarcasmEval)
- klasifikacija sentimenta (Amazon Reviews 2023)
- apstraktivna sumarizacija (Amazon Reviews 2023 + XSum)

## Struktura projekta

```
Master rad/
├── config/                 # YAML konfiguracije
│   ├── config.yaml         # pipeline i modeli
│   └── datasets.yaml       # dataset parametri
├── src/                    # izvorni kod
│   ├── config.py
│   ├── paths.py
│   ├── data/               # učitavanje i preprocessing datasetova
│   ├── pipeline/           # sarcasm, sentiment, summarization
│   ├── llm/                # Ollama klijent
│   └── evaluation/         # metrike
├── scripts/                # pokretanje
│   ├── prepare_datasets.py
│   ├── main.py
│   ├── run_ui.py
│   └── verify_setup.py
├── ui/                     # Streamlit interfejs
├── data/
│   ├── raw/                # preuzeti izvorni fajlovi
│   └── processed/          # pripremljeni JSONL splitovi
├── docs/                   # dokumentacija i tema rada
├── bat/                    # Ollama batch skripte
└── results/                # izlazi eksperimenata
```

## Brzi start

```bash
pip install -r requirements.txt
python scripts/verify_setup.py
python scripts/prepare_datasets.py --dataset all

# Ollama u drugom terminalu: ollama serve

# Evaluacija na pravim podacima (počni sa malim limitom)
python scripts/run_evaluation.py --task all --limit 5

# Demo pipeline na ugrađenim primerima
python scripts/main.py
python scripts/run_ui.py
```

Detaljan vodič za analizu: `docs/ANALIZA.md`  
Kompletan pregled projekta: `docs/PREGLED_PROJEKTA.md`

## Priprema datasetova

```bash
# Svi datasetovi
python scripts/prepare_datasets.py --dataset all

# Pojedinačno
python scripts/prepare_datasets.py --dataset sarcasm
python scripts/prepare_datasets.py --dataset sentiment
python scripts/prepare_datasets.py --dataset summarization-amazon
python scripts/prepare_datasets.py --dataset summarization-xsum
```

Standardni format zapisa:

```json
{
  "text": "...",
  "label": "...",
  "summary": "..."
}
```

Polja koja nisu relevantna za task imaju vrednost `null`.

## Detaljnija dokumentacija

Pogledaj `docs/README.md` i `docs/QUICKSTART.md`.
