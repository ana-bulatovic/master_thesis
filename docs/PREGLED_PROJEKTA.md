# Pregled projekta — sve što se radi u master radu

Ovaj dokument opisuje kompletan tok rada: cilj, arhitekturu, datasete, pipeline, skripte, evaluaciju i rezultate. Namena je da bude jedan centralni vodič kroz ceo projekat.

---

## 1. Cilj projekta

Master rad istražuje primenu **lokalnih velikih jezičkih modela (LLM)** preko **Ollama** platforme na tri povezana NLP zadatka:

1. **Detekcija sarkazma**
2. **Klasifikacija sentimenta**
3. **Apstraktivna sumarizacija**

Ključno pitanje rada: **da li eksplicitno modelovanje sarkazma i sentimenta poboljšava kvalitet sumarizacije** u odnosu na nezavisnu (baseline) sumarizaciju.

Projekat implementira višestepeni pipeline u kome se rezultati ranijih koraka mogu prosleđivati kasnijim, uz mogućnost isključivanja pojedinih koraka radi poređenja varijanti.

---

## 2. Šema rada (od podataka do rezultata)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PRIPREMA PODATAKA                                           │
│     Sirovi dataseti → preprocessing → JSONL u data/processed/   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. PIPELINE (LLM preko Ollama)                                 │
│     Tekst → sarkazam? → sentiment? → sumarizacija             │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. EVALUACIJA                                                  │
│     Predikcije vs. ground truth → Accuracy, F1, ROUGE           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. ANALIZA I PISANJE RADA                                      │
│     Tabele, grafikoni, error analysis, zaključci                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tri NLP zadatka

| Zadatak | Ulaz | Izlaz | Dataset | Metrike |
|---------|------|-------|---------|---------|
| **Sarkazam** | tweet / tekst recenzije | `sarcastic` / `non-sarcastic` | iSarcasmEval | Accuracy, Precision, Recall, F1 |
| **Sentiment** | tekst recenzije | `positive` / `negative` / `neutral` | Amazon Reviews 2023 | Accuracy, Precision, Recall, F1 |
| **Sumarizacija** | duži tekst (recenzija ili vest) | kratak sažetak (1 rečenica) | Amazon Reviews 2023 + XSum | ROUGE-1, ROUGE-2, ROUGE-L |

### Kako se labeli dobijaju

- **Sarkazam** — ručno anotiran u iSarcasmEval skupu (SemEval 2022).
- **Sentiment** — izveden iz Amazon ocene (1–5 zvezdica):
  - 1–2 → negativan
  - 3 → neutralan
  - 4–5 → pozitivan
- **Sumarizacija (Amazon)** — naslov recenzije služi kao referentni sažetak.
- **Sumarizacija (XSum)** — kratki sažetak vesti iz BBC XSum skupa.

---

## 4. Dataseti

### 4.1. iSarcasmEval (sarkazam)

- **Izvor:** [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval)
- **Lokacija:** `data/raw/sarcasm/`
- **Pripremljeno:** `data/processed/sarcasm/`
- **Split:** predefinisan train/test iz SemEval takmičenja
- **Veličina (trenutno):** ~3443 train, ~1387 test primera

### 4.2. Amazon Reviews 2023 (sentiment + sumarizacija)

- **Izvor:** McAuley-Lab/Amazon-Reviews-2023 (kategorija All_Beauty)
- **Lokacija:** `data/raw/` (preuzima se automatski)
- **Pripremljeno:**
  - `data/processed/sentiment/` — train / validation / test
  - `data/processed/summarization/amazon/` — train / validation / test
- **Veličina (trenutno):** do 10 000 uzoraka → ~9613 posle čišćenja

### 4.3. XSum (sumarizacija vesti)

- **Izvor:** EdinburghNLP/xsum (Hugging Face)
- **Pripremljeno:** `data/processed/summarization/xsum/`
- **Veličina:** do 5000 uzoraka
- **Napomena:** vesti retko sadrže sarkazam — koristan za poređenje domene (vesti vs. recenzije)

### 4.4. Standardni format zapisa (JSONL)

Svaki red u `data/processed/` je JSON objekat:

```json
{
  "text": "tekst recenzije ili članka",
  "label": "sarcastic | positive | null ...",
  "summary": "referentni sažetak ili null"
}
```

Polja koja nisu relevantna za dati task imaju vrednost `null`.

Konfiguracija datasetova: `config/datasets.yaml`

---

## 5. Pipeline — kako tekst prolazi kroz sistem

### 5.1. Glavna klasa: `NLPPipeline`

Fajl: `src/pipeline/pipeline.py`

Za svaki ulazni tekst pipeline radi sledeće:

1. **Detekcija sarkazma** (opciono) → `is_sarcastic: true/false`
2. **Klasifikacija sentimenta** (uvek) → `positive/negative/neutral`
3. **Sumarizacija** u jednoj ili više varijanti (zavisno od konfiguracije)

### 5.2. Varijante sumarizacije (za eksperimente)

| Varijanta | Opis | Polje u rezultatu |
|-----------|------|-------------------|
| **A — Baseline** | Sumarizacija bez dodatnog konteksta | `summarization_without_sentiment` |
| **B — Sentiment-aware** | Prompt uključuje predviđeni sentiment | `summarization_with_sentiment` |
| **C — Sarcasm-aware** | Prompt uključuje da li je tekst sarkastičan | `summarization_with_sarcasm` |
| **D — Puna** | Prompt uključuje i sentiment i sarkazam | `summarization_with_sentiment_and_sarcasm` |

U promptu za sumarizaciju kontekst izgleda ovako:

```
The review is sarcastic.
The review expresses a negative sentiment.
```

### 5.3. Uključivanje / isključivanje koraka

U `config/config.yaml`:

```yaml
pipeline:
  use_sarcasm_detection: true              # da li se detektuje sarkazam
  use_sentiment_for_summarization: true    # da li sentiment ide u prompt sumarizacije
  use_sarcasm_for_summarization: true      # da li sarkazam ide u prompt sumarizacije
```

Ako je `use_sarcasm_for_summarization: true`, sarkazam se detektuje čak i kada je `use_sarcasm_detection: false` (potrebno za sumarizaciju).

---

## 6. Prompting tehnike

Svi zadaci (sarkazam, sentiment, sumarizacija) podržavaju tri prompting strategije:

| Tehnika | Opis |
|---------|------|
| **zero-shot** | Samo instrukcija, bez primera |
| **few-shot** | Instrukcija + nekoliko primera ulaz/izlaz |
| **chain-of-thought** | Instrukcija + korak-po-korak razmišljanje pre odgovora |

Podrazumevano: `few-shot` (podešeno u `config/config.yaml`).

Za eksperimente se menja preko:

```bash
python scripts/run_evaluation.py --task sarcasm --technique zero-shot
python scripts/run_evaluation.py --task sarcasm --technique few-shot
python scripts/run_evaluation.py --task sarcasm --technique chain-of-thought
```

ili u Streamlit UI-ju (sidebar → Technique).

---

## 7. Modeli (Ollama)

Projekat koristi **lokalne LLM modele** preko Ollama servera (`http://localhost:11434`).

### Podržani / planirani modeli za rad

- Meta Llama 3.1
- Alibaba Qwen 2.5
- Mistral
- TinyLlama (za brze testove)

### Konfiguracija modela

Fajl: `config/config.yaml`

```yaml
model:
  name: tinyllama:latest      # podrazumevani model
  temperature: 0.7
  num_predict: 128            # maks. dužina generisanog teksta

pipeline:
  sarcasm_model: tinyllama:latest
  sentiment_model: tinyllama:latest
  summarization_model: tinyllama:latest
```

Svaki zadatak može koristiti **različit model** (npr. veći model za sumarizaciju, manji za klasifikaciju).

### Pokretanje Ollama

```bash
# Windows — batch fajl
bat\start_ollama.bat

# ili ručno
ollama serve
```

Preuzimanje modela:

```bash
ollama pull llama3.1
ollama pull tinyllama
```

---

## 8. Struktura projekta

```
Master rad/
├── config/
│   ├── config.yaml          # modeli, pipeline opcije, prompting
│   └── datasets.yaml        # parametri datasetova
│
├── src/                     # izvorni kod
│   ├── config.py            # učitavanje YAML konfiguracije
│   ├── paths.py             # putanje projekta
│   ├── data/
│   │   ├── dataset_loader.py    # preuzimanje i priprema datasetova
│   │   └── preprocessing.py     # čišćenje teksta, mapiranje labela
│   ├── llm/
│   │   └── ollama_client.py     # komunikacija sa Ollama API-jem
│   ├── pipeline/
│   │   ├── pipeline.py            # glavni orkestrator
│   │   ├── sarcasm_detector.py    # detekcija sarkazma
│   │   ├── sentiment_classifier.py
│   │   └── summarizer.py          # apstraktivna sumarizacija
│   └── evaluation/
│       └── evaluator.py           # metrike (F1, ROUGE, BERTScore)
│
├── scripts/                 # skripte za pokretanje
│   ├── prepare_datasets.py  # priprema podataka
│   ├── run_evaluation.py    # evaluacija na test skupovima
│   ├── main.py              # demo pipeline na primerima
│   ├── run_ui.py            # pokretanje Streamlit UI-ja
│   ├── verify_setup.py      # provera instalacije
│   ├── examples.py          # primeri korišćenja u kodu
│   └── fix_ollama.py        # pomoć za Ollama probleme
│
├── ui/
│   └── app.py               # Streamlit web interfejs
│
├── data/
│   ├── raw/                 # sirovi preuzeti fajlovi
│   └── processed/           # pripremljeni JSONL splitovi
│
├── results/                 # izlazi eksperimenata
│   ├── logs/                # čitljivi logovi pokretanja
│   └── evaluation_*.json    # metrike + predikcije + primeri
│
├── bat/                     # Windows batch skripte za Ollama
├── docs/                    # dokumentacija
├── Literatura/              # PDF radovi za rad
└── requirements.txt         # Python zavisnosti
```

---

## 9. Skripte — šta koja radi

### `scripts/prepare_datasets.py`

Preuzima, čisti i čuva datasete u `data/processed/`.

```bash
python scripts/prepare_datasets.py --dataset all
python scripts/prepare_datasets.py --dataset sarcasm
python scripts/prepare_datasets.py --dataset sentiment
python scripts/prepare_datasets.py --dataset summarization-amazon
python scripts/prepare_datasets.py --dataset summarization-xsum
```

**Kada se pokreće:** jednom na početku, ili kad se menjaju parametri u `datasets.yaml`.

---

### `scripts/run_evaluation.py`

Glavna skripta za **eksperimente na test podacima**. Pokreće LLM na pravim datasetovima i računa metrike.

```bash
# Svi taskovi (po 10 primera)
python scripts/run_evaluation.py --task all --limit 10

# Pojedinačni taskovi
python scripts/run_evaluation.py --task sarcasm --limit 20
python scripts/run_evaluation.py --task sentiment --limit 20
python scripts/run_evaluation.py --task summarization --source amazon --limit 10
python scripts/run_evaluation.py --task summarization --source xsum --limit 10

# Varijante sumarizacije (baseline + sentiment + sarkazam + puna)
python scripts/run_evaluation.py --task summarization --source amazon --limit 10 --with-sentiment --with-sarcasm

# Promena prompting tehnike
python scripts/run_evaluation.py --task sarcasm --limit 20 --technique chain-of-thought
```

**Argumenti:**

| Argument | Opcije | Opis |
|----------|--------|------|
| `--task` | sarcasm, sentiment, summarization, all | Koji zadatak evaluirati |
| `--source` | amazon, xsum | Izvor za sumarizaciju |
| `--split` | train, validation, test | Koji split koristiti |
| `--limit` | broj | Koliko primera (počni sa 5–10) |
| `--technique` | zero-shot, few-shot, chain-of-thought | Prompting strategija |
| `--with-sentiment` | flag | Dodatno pokreni sentiment-aware sumarizaciju |
| `--with-sarcasm` | flag | Dodatno pokreni sarcasm-aware sumarizaciju |

**Izlaz:** `results/evaluation_YYYYMMDD_HHMMSS.json` + `results/logs/run_*.log`

---

### `scripts/main.py`

Demo pipeline na **ugrađenim primerima** recenzija (na srpskom).

```bash
python scripts/main.py                  # pun pipeline na 5 primera
python scripts/main.py individual       # pojedinačni taskovi
python scripts/main.py evaluate         # demo metrika
```

**Izlaz:** `results/pipeline_results.json`

---

### `scripts/run_ui.py`

Pokreće **Streamlit web interfejs** za interaktivno testiranje.

```bash
python scripts/run_ui.py
```

Otvara se u browseru. Omogućava:
- unos recenzija (pojedinačno, batch, upload fajla)
- izbor modela i prompting tehnike
- uključivanje/isključivanje pipeline komponenti
- prikaz rezultata i export (JSON, CSV)

---

### `scripts/verify_setup.py`

Proverava da li su svi potrebni fajlovi prisutni i da li Ollama radi.

```bash
python scripts/verify_setup.py
```

---

### `scripts/examples.py`

Primeri korišćenja pipeline-a u sopstvenom kodu (CSV, JSON, custom config).

---

## 10. Evaluacija i metrike

### Klasifikacija (sarkazam, sentiment)

Implementirano u `src/evaluation/evaluator.py`:

- **Accuracy** — udeo tačnih predikcija
- **Precision** — preciznost (weighted)
- **Recall** — odziv (weighted)
- **F1** — harmonijska sredina precision i recall (weighted)

### Sumarizacija

- **ROUGE-1** — preklapanje unigrama
- **ROUGE-2** — preklapanje bigrama
- **ROUGE-L** — najduža zajednička podsekvenca
- **BERTScore** — semantička sličnost (podržano u kodu, podrazumevano isključeno u evaluaciji zbog brzine)

### Format rezultata evaluacije

```json
{
  "run_id": "20260708_031257",
  "model": "tinyllama:latest",
  "prompting_technique": "few-shot",
  "limit": 10,
  "tasks": [
    {
      "task": "sarcasm",
      "metrics": { "accuracy": 0.7, "f1": 0.68, ... },
      "samples": [ { "text": "...", "reference": "...", "predicted": "...", "correct": true } ]
    },
    {
      "task": "summarization",
      "variant": "with_sarcasm",
      "source": "amazon",
      "metrics": { "rouge": { "rouge1_f": 0.25, ... } },
      "samples": [ ... ]
    }
  ]
}
```

---

## 11. Eksperimentalni plan (za master rad)

### Faza 1 — Priprema podataka ✅

```bash
python scripts/prepare_datasets.py --dataset all
```

### Faza 2 — Bazna evaluacija

```bash
ollama serve
python scripts/run_evaluation.py --task all --limit 10
```

### Faza 3 — Poređenje prompting tehnika

Za svaki task, tri tehnike × N primera:

```bash
for technique in zero-shot few-shot chain-of-thought; do
  python scripts/run_evaluation.py --task sarcasm --limit 50 --technique $technique
done
```

### Faza 4 — Poređenje pipeline varijanti sumarizacije

```bash
python scripts/run_evaluation.py --task summarization --source amazon --limit 30 --with-sentiment --with-sarcasm
python scripts/run_evaluation.py --task summarization --source xsum --limit 30 --with-sentiment --with-sarcasm
```

Uporedi ROUGE metrike za varijante: baseline, +sentiment, +sarcasm, +oba.

### Faza 5 — Poređenje modela

Promeni `config/config.yaml` (npr. `llama3.1`, `mistral`, `qwen2.5`) i ponovi Fazu 3 i 4.

### Faza 6 — Analiza i pisanje

- Tabele: task × prompt × metrika
- Tabele: sumarizacija × varijanta × ROUGE
- Error analysis: primeri gde model pogrešno detektuje sarkazam ili loše sumarizuje
- Diskusija: uticaj sarkazma na sentiment i sumarizaciju

---

## 12. Šta ide u master rad (poglavlja)

| Poglavlje | Sadržaj iz projekta |
|-----------|---------------------|
| **Uvod** | Motivacija, opis problema, ciljevi |
| **Teorija** | LLM, prompting, sarkazam, sentiment, sumarizacija |
| **Metodologija** | Dataseti, modeli, pipeline, prompting tehnike |
| **Implementacija** | Arhitektura (`src/`), Ollama integracija, konfiguracija |
| **Eksperimenti** | Tabele rezultata po tasku, promptu, varijanti |
| **Rezultati** | Metrike, grafikoni, primeri |
| **Diskusija** | Zašto model greši, uticaj sarkazma, poređenje varijanti |
| **Zaključak** | Koji pristup/model/prompt je najbolji |

Tema rada (detaljnije): `docs/tema_za_master_rad.txt`

---

## 13. Brzi start — redosled komandi

```bash
# 1. Instalacija
pip install -r requirements.txt
python scripts/verify_setup.py

# 2. Priprema podataka (jednom)
python scripts/prepare_datasets.py --dataset all

# 3. Ollama (drugi terminal)
ollama serve
ollama pull tinyllama    # ili llama3.1

# 4. Brzi test evaluacije
python scripts/run_evaluation.py --task all --limit 5

# 5. Demo pipeline
python scripts/main.py

# 6. Web UI (opciono)
python scripts/run_ui.py
```

---

## 14. Gde su rezultati

```
results/
├── logs/
│   └── run_20260708_031257.log       # čitljiv log svakog pokretanja
├── evaluation_20260708_031257.json   # pun JSON sa metrikama i primerima
└── pipeline_results.json             # izlaz iz main.py demo-a
```

---

## 15. Korisni saveti

- **Počni sa `--limit 5`** — LLM inferenca je spora; prvo proveri da sve radi.
- **TinyLlama** je brz ali slabiji — za prave eksperimente koristi `llama3.1` ili `mistral`.
- **Ollama mora biti pokrenut** pre `main.py`, `run_evaluation.py` i UI-ja.
- Za sumarizaciju na Amazon skupu, sarkazam se **automatski detektuje** modelom (nema ground truth labela za sarkazam u tom skupu).
- Za error analysis pogledaj `samples` niz u JSON rezultatima — tu su svi primeri sa predikcijama.

---

## 16. Povezana dokumentacija

| Fajl | Sadržaj |
|------|---------|
| `docs/PREGLED_PROJEKTA.md` | Ovaj dokument — kompletan pregled |
| `docs/ANALIZA.md` | Kratki vodič za analizu i eksperimente |
| `docs/README.md` | Detaljna tehnička dokumentacija (EN) |
| `docs/QUICKSTART.md` | Brzi start vodič |
| `docs/UI_GUIDE.md` | Uputstvo za Streamlit interfejs |
| `docs/tema_za_master_rad.txt` | Predlog teme master rada |
| `README.md` | Kratak pregled u korenu projekta |
