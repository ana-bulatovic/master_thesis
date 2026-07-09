# Nacrt master rada — struktura i prva poglavlja

**Radna verzija za pisanje master rada**

**Naslov (predlog):**  
Primena velikih jezičkih modela za detekciju sarkazma, klasifikaciju sentimenta i sumarizaciju korisničkih recenzija

---

# DEO A — Kako treba da izgleda dokument

## Formalni delovi (pre poglavlja)

Ovi delovi idu **pre uvoda** i obično se ne numerišu kao poglavlja (ili se numerišu rimskim brojevima, zavisno od zahteva fakulteta).

| Redosled | Deo | Šta sadrži |
|----------|-----|------------|
| 1 | **Naslovna strana** | Naziv rada, ime autora, mentor, fakultet, studijski program, godina |
| 2 | **Strana sa odobrenjem** | Potpis mentora i članova komisije (šablon sa fakulteta) |
| 3 | **Izjava o autentičnosti** | Da je rad originalan, da su izvori citirani |
| 4 | **Zahvalnica** | Kratka (opciono, ½ strane) |
| 5 | **Apstrakt na srpskom** | 150–300 reči: problem, metod, rezultati, zaključak |
| 6 | **Ključne reči** | 5–7 reči (npr. *veliki jezički modeli, sarkazam, sentiment, sumarizacija, prompting*) |
| 7 | **Abstract (engleski)** | Prevod apstrakta |
| 8 | **Keywords** | Engleski prevod ključnih reči |
| 9 | **Sadržaj** | Automatski generisan u Wordu/LaTeX-u |
| 10 | **Lista skraćenica** | NLP, LLM, ROUGE, F1... |
| 11 | **Lista slika i tabela** | Opciono, ali preporučljivo |

### Tehnički format (tipični zahtevi)

- **Font:** Times New Roman 12 pt (ili zahtev fakulteta)
- **Prored:** 1,5
- **Margine:** leva 3 cm (za povez), ostale 2,5 cm
- **Numeracija strana:** rimski (i, ii, iii...) za formalne delove; arapski od Uvoda
- **Naslovi poglavlja:** podebljani, centrirani ili levo, velika slova
- **Podnaslovi:** numerisani (1.1, 1.1.1)
- **Citiranje:** IEEE ili APA — proveriti šablon fakulteta
- **Slike i tabele:** numerisane po poglavlju (Slika 3.2, Tabela 4.1) + opis ispod/iznad
- **Literatura:** na kraju, numerisana ili po prezimenu autora

### Procena obima

| Deo | Preporučeni obim |
|-----|------------------|
| Uvod | 3–5 strana |
| Teorijski okvir | 15–25 strana |
| Metodologija | 8–12 strana |
| Implementacija | 8–15 strana |
| Eksperimentalni rezultati | 10–20 strana |
| Diskusija | 5–8 strana |
| Zaključak | 2–4 strane |
| **Ukupno (bez priloga)** | **~50–80 strana** |

---

## Glavna poglavlja (numerisana)

```
1. UVOD
2. TEORIJSKI OKVIR I PREGLED LITERATURE
   2.1. Obrađa prirodnog jezika
   2.2. Veliki jezički modeli
   2.3. Prompting tehnike
   2.4. Detekcija sarkazma
   2.5. Analiza sentimenta
   2.6. Sumarizacija teksta
   2.7. Višestepeni NLP pipeline-i
   2.8. Pregled srodnih radova
3. METODOLOGIJA ISTRAŽIVANJA
   3.1. Cilj i istraživačka pitanja
   3.2. Skupovi podataka
   3.3. Modeli i softverska platforma
   3.4. Dizajn pipeline sistema
   3.5. Varijante eksperimenata
   3.6. Metrike evaluacije
   3.7. Eksperimentalni protokol
4. IMPLEMENTACIJA SISTEMA
   4.1. Arhitektura sistema
   4.2. Priprema podataka
   4.3. Modul za komunikaciju sa LLM modelom
   4.4. Moduli za sarkazam, sentiment i sumarizaciju
   4.5. Evaluacioni modul
   4.6. Korisnički interfejs
5. EKSPERIMENTALNI REZULTATI
   5.1. Rezultati detekcije sarkazma
   5.2. Rezultati klasifikacije sentimenta
   5.3. Rezultati sumarizacije
   5.4. Poređenje pipeline varijanti
   5.5. Uticaj prompting tehnika
   5.6. Kvalitativna analiza grešaka
6. DISKUSIJA
   6.1. Interpretacija rezultata
   6.2. Uticaj sarkazma na sentiment i sumarizaciju
   6.3. Ograničenja istraživanja
   6.4. Mogućnosti za budući rad
7. ZAKLJUČAK

LITERATURA

PRILOZI
   Prilog A — Primeri promptova
   Prilog B — Primeri predikcija modela
   Prilog C — Konfiguracioni fajlovi
```

---

# DEO B — Apstrakt (nacrt)

## Apstrakt (srpski)

Rast količine korisničkih recenzija na internet platformama čini automatsku analizu teksta neophodnom za razumevanje mišljenja potrošača. Tradicionalni pristupi u oblasti obrade prirodnog jezika (NLP) često rešavaju zadatke detekcije sarkazma, klasifikacije sentimenta i sumarizacije teksta nezavisno jedan od drugog, iako su ovi zadaci u praksi semantički povezani. Sarkastične recenzije, na primer, mogu sadržati pozitivne reči uz negativno značenje, što dovodi do pogrešne klasifikacije sentimenta i netačnih sažetaka.

U ovom radu istražuje se primena lokalno pokrenutih velikih jezičkih modela (LLM), dostupnih preko platforme Ollama, na tri NLP zadatka: detekciju sarkazma, klasifikaciju sentimenta i apstraktivnu sumarizaciju. Implementiran je višestepeni pipeline sistem koji omogućava ispitivanje uticaja informacija o sarkazmu i sentimentu na kvalitet generisanih sažetaka. Eksperimenti su sprovedeni na skupovima podataka iSarcasmEval, Amazon Reviews 2023 i XSum, uz primenu tri prompting strategije: zero-shot, few-shot i chain-of-thought prompting.

Rezultati pokazuju [OVDE DOPUNI KADA IMAŠ EKSPERIMENTE — npr. da li sentiment-aware i sarcasm-aware sumarizacija daju bolje ROUGE skorove]. Rad doprinosi empirijskoj proceni lokalnih LLM modela u realnim NLP zadacima i pruža okvir za dalja istraživanja višezadatnih pipeline sistema.

**Ključne reči:** veliki jezički modeli, detekcija sarkazma, analiza sentimenta, sumarizacija teksta, prompting, Ollama

---

# DEO C — Poglavlje 1: UVOD (detaljno)

## 1. Uvod

### 1.1. Motivacija

U poslednjoj deceniji došlo je do eksponencijalnog rasta količine tekstualnih podataka generisanih na internetu. Platforme za elektronsku trgovinu, društvene mreže i servisi za recenziranje svakodnevno objavljuju milione korisničkih komentara i ocena. Ovi tekstovi predstavljaju vredan izvor informacija o zadovoljstvu kupaca, kvalitetu proizvoda i trendovima na tržištu, ali je njihova ručna analiza praktično nemoguća zbog obima.

Automatska obrada prirodnog jezika (Natural Language Processing — NLP) omogućava mašinsko razumevanje, klasifikaciju i generisanje teksta. Međutim, korisničke recenzije često sadrže složene jezičke fenomena koji otežavaju tačnu analizu. Jedan od najizazovnijih takvih fenomena jeste **sarkazam** — oblik indirektne komunikacije u kome govornik izražava značenje suprotno od doslovnog smisla izgovorenih reči. Recenzija poput *„Odličan proizvod, pukao je posle dva dana — baš kao što sam i očekivao"* doslovno sadrži pozitivne izraze, ali u kontekstu jasno komunicira zadovoljstvo negativno.

Pogrešna interpretacija sarkazma direktno utiče na zadatke koji se oslanjaju na emocionalni ton teksta. **Analiza sentimenta** (sentiment analysis) klasifikuje tekst kao pozitivan, negativan ili neutralan, dok **sumarizacija teksta** (text summarization) generiše kraći prikaz glavnih informacija iz dužeg ulaza. Ako model ne prepozna sarkazam, može pogrešno klasifikovati sarkastičnu negativnu recenziju kao pozitivnu i generisati sažetak koji ne odražava stvarno mišljenje autora.

Pojava **velikih jezičkih modela** (Large Language Models — LLM) otvorila je nove mogućnosti za rešavanje ovih zadataka bez potrebe za obimnim treniranjem specijalizovanih modela od nule. Modeli poput Llama, Mistral i Qwen, pokrenuti lokalno preko platforme Ollama, mogu se prilagoditi različitim zadacima isključivo promenom ulaznog **prompta** (uputa modelu). Ovo je posebno atraktivno u akademskom i istraživačkom kontekstu jer omogućava transparentnu eksperimentaciju sa različitim strategijama promptovanja.

Motivacija za ovaj rad leži u potrebi da se empirijski ispita koliko dobro lokalni LLM modeli rešavaju povezane NLP zadatke na realnim skupovima podataka, kao i da li **eksplicitno uključivanje informacija o sarkazmu i sentimentu** poboljšava kvalitet sumarizacije u odnosu na nezavisnu (baseline) sumarizaciju.

### 1.2. Opis problema

Istraživački problem koji ovaj rad adresira može se formuliasati na sledeći način:

> Kako lokalno pokrenuti veliki jezički model može da reši zadatke detekcije sarkazma, klasifikacije sentimenta i apstraktivne sumarizacije, i da li rezultati ranijih koraka u pipeline-u poboljšavaju kvalitet kasnijih?

U praksi se ova tri zadatka često tretiraju kao odvojeni problemi. Detekcija sarkazma se rešava binarnom klasifikacijom, sentiment se određuje nezavisno, a sumarizacija se primenjuje direktno na izvorni tekst bez kontekstualnih informacija iz prethodnih koraka. Takav **atomistički pristup** ignoriše činjenicu da su sarkazam, emocionalni ton i semantički sadržaj teksta međusobno povezani.

Konkretni izazovi uključuju:

1. **Sarkazam kao izvor grešaka u sentiment analizi.** Sarkastične recenzije često koriste pozitivan vokabular za izražavanje negativnog stava, što dovodi do lažno pozitivnih predikcija kod modela koji ne modeluju pragmatički kontekst.

2. **Sumarizacija bez konteksta.** Standardna sumarizacija ne uzima u obzir da je recenzija sarkastična ili da izražava specifičan sentiment, što može dovesti do sažetaka koji su gramatički ispravni ali semantički pogrešni.

3. **Zavisnost od prompting strategije.** Performanse LLM modela značajno variraju u zavisnosti od toga kako je zadatak formulisan u promptu (zero-shot, few-shot ili chain-of-thought), što zahteva sistematsko poređenje.

4. **Ograničenja lokalnih modela.** Za razliku od komercijalnih API-ja (GPT-4, Claude), lokalni modeli imaju manji broj parametara i ograničeniji kontekst, što može uticati na kvalitet rezultata — posebno na složenim primerima sa sarkazmom.

### 1.3. Cilj rada

**Glavni cilj** ovog rada jeste dizajn, implementacija i eksperimentalna evaluacija višestepenog NLP pipeline sistema zasnovanog na lokalnim LLM modelima, koji integriše detekciju sarkazma, klasifikaciju sentimenta i apstraktivnu sumarizaciju korisničkih recenzija.

Konkretni ciljevi:

1. Implementirati pipeline sistem koji podržava različite varijante obrade teksta (nezavisna sumarizacija, sentiment-aware sumarizacija, sarcasm-aware sumarizacija i kombinovana varijanta).

2. Pripremiti i standardizovati skupove podataka za tri NLP zadatka (iSarcasmEval, Amazon Reviews 2023, XSum).

3. Sprovesti eksperimente sa različitim prompting tehnikama (zero-shot, few-shot, chain-of-thought) i lokalnim LLM modelima.

4. Kvantitativno evaluirati performanse koristeći standardne metrike (Accuracy, F1, ROUGE) i uporediti pipeline varijante.

5. Sprovesti kvalitativnu analizu tipičnih grešaka modela, sa posebnim osvrtom na sarkastične primere.

### 1.4. Istraživačka pitanja i hipoteze

Na osnovu definisanog cilja, u radu se postavljaju sledeća istraživačka pitanja:

**I1.** Koliko su lokalni LLM modeli tačni u zadacima detekcije sarkazma i klasifikacije sentimenta na standardnim skupovima podataka?

**I2.** Koja prompting tehnika (zero-shot, few-shot, chain-of-thought) daje najbolje rezultate za svaki od tri zadatka?

**I3.** Da li uključivanje informacije o sentimentu u prompt za sumarizaciju poboljšava ROUGE metrike u odnosu na baseline sumarizaciju?

**I4.** Da li uključivanje informacije o sarkazmu u prompt za sumarizaciju poboljšava kvalitet sažetaka, posebno na sarkastičnim recenzijama?

**I5.** Da li kombinovana varijanta (sarkazam + sentiment) daje bolje rezultate od pojedinačnih context-aware varijanti?

Na osnovu ovih pitanja formulisu se hipoteze:

**H1.** Few-shot i chain-of-thought prompting daju bolje rezultate od zero-shot prompting-a za zadatke klasifikacije.

**H2.** Sentiment-aware sumarizacija postiže više ROUGE skorove od nezavisne sumarizacije na Amazon recenzijama.

**H3.** Sarkasm-aware sumarizacija smanjuje broj semantički netačnih sažetaka na sarkastičnim primerima.

**H4.** Kombinovana varijanta pipeline-a ne mora uvek biti najbolja po ROUGE metrikama, ali daje semantički konzistentnije sažetke na složenim primerima.

### 1.5. Metodologija rada (kratak pregled)

Istraživanje je **eksperimentalno** i obuhvata:

- **Priprema podataka** — preuzimanje, čišćenje i standardizacija skupova podataka u jedinstven JSONL format.
- **Implementacija pipeline sistema** — Python moduli za komunikaciju sa Ollama API-jem, detekciju sarkazma, klasifikaciju sentimenta i sumarizaciju.
- **Dizajn eksperimenata** — varijante pipeline-a × prompting tehnike × skupovi podataka.
- **Evaluacija** — automatsko računanje metrika (Accuracy, Precision, Recall, F1, ROUGE-1/2/L) na test skupovima.
- **Analiza** — kvantitativno poređenje varijanti i kvalitativna analiza reprezentativnih primera.

Detaljan opis metodologije dat je u Poglavlju 3.

### 1.6. Doprinos rada

Doprinos ovog master rada ogleda se u sledećem:

1. **Integrisani pipeline** — implementiran je modularni sistem koji omogućava ispitivanje uticaja sarkazma i sentimenta na sumarizaciju, sa mogućnošću uključivanja i isključivanja pojedinih koraka.

2. **Empirijska evaluacija lokalnih LLM modela** — rad pruža uporedne rezultate za više modela i prompting strategija na standardnim skupovima podataka, što je relevantno za istraživače koji koriste Ollama platformu.

3. **Četiri varijante sumarizacije** — eksplicitno su definisane i evaluirane varijante: baseline, sentiment-aware, sarcasm-aware i kombinovana, što omogućava sistematsko ispitivanje istraživačkih pitanja I3–I5.

4. **Otvorena implementacija** — kompletan kod, konfiguracija i pripremljeni skupovi podataka omogućavaju ponovljivost eksperimenata.

### 1.7. Struktura rada

Rad je organizovan u sedam poglavlja.

**Poglavlje 1 (Uvod)** predstavlja motivaciju, opis problema, ciljeve, istraživačka pitanja i doprinos rada.

**Poglavlje 2 (Teorijski okvir i pregled literature)** daje teorijsku osnovu za razumevanje LLM modela, prompting tehnika i tri NLP zadatka, uz pregled srodnih istraživanja.

**Poglavlje 3 (Metodologija istraživanja)** opisuje skupove podataka, modele, dizajn pipeline sistema, varijante eksperimenata i metrike evaluacije.

**Poglavlje 4 (Implementacija sistema)** detaljno predstavlja softversku arhitekturu i ključne module implementiranog sistema.

**Poglavlje 5 (Eksperimentalni rezultati)** prikazuje kvantitativne rezultate eksperimenata i kvalitativnu analizu primera.

**Poglavlje 6 (Diskusija)** interpretira rezultate, razmatra ograničenja i predlaže pravce budućih istraživanja.

**Poglavlje 7 (Zaključak)** sumira ključne nalaze i daje preporuke.

Na kraju rada nalazi se spisak literature i prilozi sa dodatnim materijalima.

---

# DEO D — Poglavlje 2: TEORIJSKI OKVIR (detaljno — prve sekcije)

## 2. Teorijski okvir i pregled literature

### 2.1. Obrađa prirodnog jezika

Obrađa prirodnog jezika (Natural Language Processing — NLP) predstavlja interdisciplinarnu oblast na preseku računarstva, lingvistike i veštačke inteligencije, koja se bavi automatskim razumevanjem, generisanjem i manipulacijom ljudskim jezikom. Cilj NLP sistema je omogućiti računarima da obrađuju tekst i govor na način koji je koristan za ljude — bilo da se radi o prevođenju, odgovaranju na pitanja, klasifikaciji dokumenata ili izvlačenju informacija.

Tradicionalni NLP pristupi zasnivali su se na **pravilima** (rule-based systems) i **statističkim modelima** (n-grami, skriveni Markovljevi modeli). Ovi pristupi zahtevaju značajan domen-specifičan inženjering i imaju ograničenu sposobnost generalizacije na nove jezičke konstrukcije.

Revolucija u oblasti došla je sa **neuronskim mrežama** i, posebno, sa **transformer arhitekturom** (Vaswani et al., 2017). Transformer modeli koriste mehanizam **pažnje** (attention) koji omogućava modelu da simultano obrađuje sve reči u ulazu i da uči složene zavisnosti između njih, bez rekurzivne obrade sekvence. Ova arhitektura je osnova za savremene jezičke modele.

U kontekstu ovog rada, NLP zadaci se dele na tri kategorije prema tipu izlaza:

- **Klasifikacija** — dodela diskretne labele tekstu (sarkazam: da/ne; sentiment: pozitivan/negativan/neutralan).
- **Generisanje** — kreiranje novog teksta na osnovu ulaza (sumarizacija).
- **Sekvenca-zadaci** — kombinacija gornjeg, gde model generiše labelu ili tekst kao odgovor na prompt.

### 2.2. Veliki jezički modeli

**Veliki jezički modeli** (Large Language Models — LLM) su neuronske mreže sa milijardama parametara, trenirane na ogromnim korpusima teksta iz interneta, knjiga i drugih izvora. Treniranje obuhvata zadatak **predviđanja sledeće reči** (next token prediction), čime model internalizuje gramatička pravila, činjenice o svetu i određene sposobnosti zaključivanja.

Ključne karakteristike savremenih LLM modela relevantne za ovaj rad:

**Generativna priroda.** Za razliku od klasičnih klasifikatora koji vraćaju verovatnoću po klasi, LLM generiše tekstualni odgovor. Zadatak klasifikacije se rešava formulisanjem pitanja u promptu i parsiranjem generisanog odgovora (npr. „Odgovori samo: positive, negative ili neutral").

**In-context učenje.** LLM modeli mogu da reše nove zadatke bez ažuriranja težina modela, isključivo na osnovu instrukcija i primera u promptu. Ovo je poznato kao **in-context learning** i čini osnovu zero-shot i few-shot pristupa.

**Ograničen kontekst.** Svaki model ima maksimalnu dužinu konteksta (broj tokena koje može obraditi odjednom). Za lokalne modele ovo je obično 2048–8192 tokena, što je dovoljno za recenzije ali može biti ograničavajuće za duge dokumente.

**Halucinacije.** LLM modeli mogu generisati sadržaj koji nije prisutan u ulazu — fenomen poznat kao halucinacija. U sumarizaciji je to posebno problematično, jer model može dodati informacije koje korisnik nije naveo u recenziji.

#### 2.2.1. Ollama platforma

Za potrebe ovog rada korišćena je **Ollama** — open-source platforma za lokalno pokretanje LLM modela. Ollama pruža REST API preko koga aplikacija šalje prompt i prima generisani odgovor, bez potrebe za slanjem podataka na eksterne servere.

Prednosti lokalnog pokretanja za istraživačke svrhe:

- **Privatnost podataka** — recenzije se ne dele sa trećim stranama.
- **Reproducibilnost** — fiksna verzija modela i konfiguracija.
- **Bez troškova API poziva** — pogodno za obimne eksperimente.
- **Offline rad** — ne zavisi od internet konekcije tokom inferencije.

U radu su testirani modeli iz porodica Llama 3.1, Mistral i Qwen 2.5, dostupni preko Ollama registra modela.

### 2.3. Prompting tehnike

**Prompting** označava proces formulisanja ulaznog teksta (prompta) koji se šalje LLM modelu. Kvalitet prompta direktno utiče na tačnost i konzistentnost odgovora. U ovom radu ispituju se tri prompting strategije.

#### 2.3.1. Zero-shot prompting

Kod **zero-shot** pristupa model dobija samo instrukciju o zadatku, bez primera. Prompt tipično sadrži opis zadatka i ulazni tekst:

```
Analiziraj sentiment sledeće recenzije.
Recenzija: [tekst]
Sentiment (positive/negative/neutral):
```

Prednost je jednostavnost i brzina. Mana je što model nema referentne primere željenog formata odgovora, što može dovesti do nekonzistentnih izlaza.

#### 2.3.2. Few-shot prompting

**Few-shot** prompting dodaje nekoliko primera (ulaz → izlaz) pre stvarnog zadatka. Primeri „kalibruju" model na željeni format i tip zadatka:

```
Primeri:
Recenzija: Odličan proizvod! → Sentiment: positive
Recenzija: Loš kvalitet. → Sentiment: negative

Sada klasifikuj:
Recenzija: [tekst] → Sentiment:
```

Few-shot pristup često daje bolje rezultate od zero-shot, posebno kod manjih lokalnih modela, ali povećava dužinu prompta i troši deo kontekstnog prozora.

#### 2.3.3. Chain-of-thought prompting

**Chain-of-thought** (CoT) prompting podstiče model da eksplicitno prikaže korake razmišljanja pre finalnog odgovora. Za detekciju sarkazma, prompt može uključivati korake:

```
Korak 1: Identifikuj doslovno značenje reči.
Korak 2: Uoči kontradikcije sa kontekstom.
Korak 3: Odredi da li postoji sarkazam.
Odgovor:
```

CoT je posebno koristan za složene zadatke koji zahtevaju zaključivanje, ali generiše duže odgovore i povećava vreme inferencije.

#### 2.3.4. Poređenje tehnika u kontekstu rada

U ovom radu ista prompting tehnika se primenjuje konzistentno na sva tri zadatka (sarkazam, sentiment, sumarizacija) kako bi se omogućilo fer poređenje. Podrazumevana tehnika u implementiranom sistemu je **few-shot**, uz mogućnost prebacivanja na zero-shot ili chain-of-thought kroz konfiguracioni fajl ili argument komandne linije.

### 2.4. Detekcija sarkazma

**Sarkazam** je figurativni oblik govora u kome izraženo značenje odstupa od doslovnog. U pisanom tekstu na društvenim mrežama i platformama za recenzije sarkazam je čest, posebno kod izražavanja nezadovoljstva na ironičan način.

Detekcija sarkazma formalno je zadatak **binarne klasifikacije**: za dati tekst, model treba da odredi da li je sarkastičan (`sarcastic`) ili nije (`non-sarcastic`).

#### 2.4.1. Izazovi detekcije sarkazma

Sarkazam je izazovan za automatsku detekciju iz nekoliko razloga:

1. **Kontekstualna zavisnost** — ista rečenica može biti sarkastična ili doslovna u zavisnosti od situacije.
2. **Odsustvo eksplicitnih markera** — za razliku od emocija izraženih kroz emodžije, sarkazam retko ima jednoznačne leksičke indikatore.
3. **Kulturna i jezička specifičnost** — obrasci sarkazma variraju između jezika i kultura.
4. **Pozitivan vokabular, negativna namera** — tipičan obrazac u recenzijama: „Sjajan proizvod, baš vredi svake pare" uz ocenu 1 zvezdice.

#### 2.4.2. Skup podataka iSarcasmEval

Za evaluaciju detekcije sarkazma u ovom radu korišćen je skup **iSarcasmEval** (Farha et al., 2022), kreiran u okviru SemEval-2022 takmičenja. Skup sadrži tweetove na engleskom jeziku sa ručnom anotacijom sarkazma i predstavlja referentni benchmark u oblasti.

U implementiranom sistemu zadatak se rešava LLM promptom koji traži od modela da analizira tekst i odgovori da li sadrži sarkazam, uz parsiranje binarnog odgovora iz generisanog teksta.

### 2.5. Analiza sentimenta

**Analiza sentimenta** (sentiment analysis) jeste zadatak određivanja emocionalne polariteta ili stava izraženog u tekstu. U okviru ovog rada sentiment se modelira kao **trosrazna klasifikacija**:

- **Pozitivan** — autor izražava zadovoljstvo ili preporuku.
- **Negativan** — autor izražava nezadovoljstvo ili kritiku.
- **Neutralan** — autor ne izražava jaku emocionalnu polarizaciju.

#### 2.5.1. Sentiment u korisničkim recenzijama

Kod recenzija proizvoda sentiment je često povezan sa numeričkom ocenom. U skupu Amazon Reviews 2023, ocene od 1 do 5 zvezdica mapiraju se na klase sentimenta:

| Ocena | Klasa sentimenta |
|-------|------------------|
| 1–2 | negativan |
| 3 | neutralan |
| 4–5 | pozitan |

Ova automatska anotacija omogućava kreiranje velikog skupa podataka bez ručnog označavanja, ali uvodi šum — korisnik može dati 3 zvezdice a ipak pisati pretežno pozitivan tekst.

#### 2.5.2. Veza između sarkazma i sentimenta

Sarkazam i sentiment nisu nezavisni. Sarkastična recenzija gotovo uvek ima negativan stvarni sentiment, bez obzira na pozitivne reči u tekstu. Zato je logično očekivati da informacija o sarkazmu može poboljšati klasifikaciju sentimenta — i obrnuto, pogrešna detekcija sarkazma vodi do pogrešnog sentimenta. U implementiranom pipeline-u ova veza se eksplicitno koristi pri sumarizaciji, gde se obe informacije mogu proslediti modelu.

---

## Napomena za nastavak Poglavlja 2

Sledeće sekcije koje treba napisati (kraći nacrt):

### 2.6. Sumarizacija teksta
- Ekstraktivna vs. apstraktivna sumarizacija
- ROUGE metrike
- Amazon (naslov kao referenca) i XSum

### 2.7. Višestepeni NLP pipeline-i
- Koncept pipeline varijanti (A, B, C, D)
- Context-aware sumarizacija

### 2.8. Pregled srodnih radova
- Survey radovi iz `Literatura/` foldera
- Uporedna tabela: rad × pristup × zadaci

---

# DEO E — Šta dalje pisati (po prioritetu)

| Redosled | Poglavlje | Šta treba | Odakle uzimaš materijal |
|----------|-----------|-----------|------------------------|
| 1 | **3. Metodologija** | Dataseti, modeli, eksperimenti | `docs/PREGLED_PROJEKTA.md`, `config/` |
| 2 | **4. Implementacija** | Arhitektura, moduli, dijagrami | `src/` kod |
| 3 | **5. Rezultati** | Tabele, grafikoni | `results/evaluation_*.json` |
| 4 | **6. Diskusija** | Interpretacija, greške | Logovi, error analysis |
| 5 | **7. Zaključak** | 1–2 strane, bez novih informacija | Sve poglavlje 5 i 6 |
| 6 | **2.6–2.8** | Dopuna teorije | PDF-ovi iz `Literatura/` |

### Saveti za pisanje

1. **Uvod piši poslednji** (ili ga doradi posle rezultata) — najlakše je precizirati doprinos kad znaš šta si dobila.
2. **Svako poglavlje počinje uvodnim pasusom** koji kaže šta će čitalac naučiti u tom poglavlju.
3. **Svako poglavlje završava kratkim rezimeom** prelaza na sledeće.
4. **Slike i tabele su obavezne** — bar: arhitektura sistema, pipeline dijagram, tabele rezultata.
5. **Citiraj literature iz foldera** `Literatura/` — imaš 4 survey PDF-a spremna.

---

*Fajl kreiran kao radni nacrt. Prepiši u Word/LaTeX šablon fakulteta i prilagodi formatiranje prema zvaničnim zahtevima.*
