# Šta se radi u projektu — vodič za analizu

## 1. Cilj projekta

Master rad ispituje koliko dobro **lokalni LLM (Llama 2 preko Ollama)** rešava tri NLP zadatka na pripremljenim datasetovima, koristeći različite **prompting strategije** (zero-shot, few-shot, chain-of-thought).

```
Tekst (iz dataset-a)
       │
       ▼
┌──────────────────┐
│  LLM (Llama 2)   │  ← prompt: zero-shot / few-shot / CoT
└──────────────────┘
       │
       ▼
  Predikcija (label ili sažetak)
       │
       ▼
  Uporedba sa ground truth
       │
       ▼
  Metrike (F1, ROUGE, ...)
```

---

## 2. Tri zadatka

| Zadatak | Dataset | Ulaz | Očekivani izlaz | Metrike |
|---------|---------|------|-----------------|---------|
| **Sarkazam** | iSarcasmEval | tweet/tekst | sarcastic / non-sarcastic | Accuracy, F1 |
| **Sentiment** | Amazon Reviews 2023 | recenzija | positive / negative / neutral | Accuracy, F1 |
| **Sumarizacija** | Amazon + XSum | članak/recenzija | kratak sažetak | ROUGE-1/2/L |

---

## 3. Pipeline varijante (za rad)

```
Varijanta A: samo sumarizacija (baseline)
Varijanta B: sentiment → sumarizacija (sentiment-aware)
Varijanta C: sarkazam → sumarizacija (sarcasm-aware)
Varijanta D: sarkazam + sentiment → sumarizacija (puna)
```

U eksperimentima porediš da li prethodni koraci poboljšavaju kasnije.

---

## 4. Faze rada (šta si uradila i šta sledi)

### Faza 1 — Priprema podataka ✅ (gotovo)
```bash
python scripts/prepare_datasets.py --dataset all
```
Rezultat: `data/processed/` sa JSONL fajlovima.

### Faza 2 — Evaluacija na test podacima ← SADA
```bash
ollama serve
python scripts/run_evaluation.py --task all --limit 10
```
Rezultat: metrike + log + JSON sa predikcijama.

### Faza 3 — Upoređivanje prompting tehnika
```bash
python scripts/run_evaluation.py --task sarcasm --limit 20 --technique zero-shot
python scripts/run_evaluation.py --task sarcasm --limit 20 --technique few-shot
python scripts/run_evaluation.py --task sarcasm --limit 20 --technique chain-of-thought
```

### Faza 4 — Analiza i pisanje rada
- Tabele rezultata po tasku i promptu
- Error analysis (primeri pogrešnih predikcija)
- Zaključak: koji prompt/model najbolji

---

## 5. Gde su rezultati

```
results/
├── logs/
│   └── run_20260708_030000.log    ← čitljiv log svakog pokretanja
└── evaluation_20260708_030000.json  ← svi podaci + metrike + primeri
```

---

## 6. Preporučeni red pokretanja

```bash
# 1. Ollama (drugi terminal)
ollama serve

# 2. Brzi test — 5 primera po tasku (~5-15 min sa llama2)
python scripts/run_evaluation.py --task all --limit 5

# 3. Jedan task detaljnije
python scripts/run_evaluation.py --task sarcasm --limit 20

# 4. Sumarizacija
python scripts/run_evaluation.py --task summarization --source amazon --limit 5
python scripts/run_evaluation.py --task summarization --source xsum --limit 5

# 5. Sa sentiment-aware i sarcasm-aware sumarizacijom
python scripts/run_evaluation.py --task summarization --source amazon --limit 5 --with-sentiment --with-sarcasm
```

**Savet:** `--limit 5` prvo, jer je llama2 spor. Kad vidiš da radi, povećaj na 20–50 za prave eksperimente.

---

## 7. Šta ide u master rad

| Poglavlje | Sadržaj |
|-----------|---------|
| Metodologija | Dataseti, modeli, prompting, pipeline |
| Eksperimenti | Tabele: task × prompt × metrika |
| Rezultati | Grafikoni, primeri |
| Diskusija | Zašto model greši, poređenje varijanti |
| Zaključak | Koji pristup preporučuješ |
