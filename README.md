# Ehtobotti

Paikallisesti toimiva tekoälyavustaja, joka vastaa kysymyksiin julkisista vakuutusehdoista ja
kertoo jokaisen väitteen perässä, mistä ehtojen kohdasta ja miltä sivulta tieto on peräisin.

> **Epävirallinen demo.** Projekti ei ole minkään vakuutusyhtiön palvelu eikä sillä ole
> yhteyttä vakuutusyhtiöihin. Vastaukset eivät ole virallisia korvauspäätöksiä.

## Miksi

Vakuutusehdot ovat pitkiä ja vaikealukuisia. Asiakas kysyy asiakaspalvelusta tai jättää
kysymättä, ja asiakaspalvelija etsii vastauksen samasta PDF:stä. Ehtobotti näyttää, miten
kielimallilla voi vastata tällaisiin kysymyksiin niin, että vastaus on tarkistettavissa:

- **Lähde jokaiselle väitteelle**, muodossa `[Autovakuutus, kohta 3.9, s. 5]`. Käyttöliittymä
  näyttää viitatun ehtotekstin vastauksen alla.
- **Kieltäytyy arvaamasta.** Jos ehdoista ei löydy vastausta, botti sanoo sen
  ("En löytänyt vastausta vakuutusehdoista.") eikä keksi vastausta.
- **Keksityt viitteet huomataan.** Jos malli viittaa kohtaan, jota sille ei annettu, käyttäjä
  näkee varoituksen ja kohdan oikean tekstin.
- **Yhtiöiden ehdot pidetään erillään.** Käyttäjä valitsee vakuutuslajin ja yhtiön, joten
  kahden yhtiön ehdot eivät sekoitu samaan vastaukseen.
- **Tietosuoja.** Kaikki toimii omalla koneella (Ollama). Kysymykset eivät lähde
  pilvipalveluun, eikä aineistossa ole henkilötietoja.
- **Mitattu, ei arvattu.** Laatua seurataan omalla kysymyssarjalla ja mittareilla (alla).

## Miten se toimii

```
PDF ──► tekstin poiminta ──► pilkonta ehtojen omien ──► upotukset ──► Chroma-
        (PyMuPDF, asettelu)  väliotsikoiden mukaan      (bge-m3)       vektorikanta
                                                                          │
kysymys ──► upotus ──► 5 osuvinta otetta (vakuutuslaji + yhtiö) ◄─────────┘
                              │
                              ▼
            kielimalli (qwen3.5:4b) vastaa JSON-muodossa:
            {"vastattavissa": true/false, "vastaus": "... [lähde]"}
                              │
                              ▼
            koodi tarkistaa viitteet, lisää vakiomuotoisen kieltäytymisen
            ja huomautuksen, näyttää lähteet
```

Keskeiset valinnat ja niiden perustelut:

- **Pilkonta ehtojen rakenteen mukaan.** Teksti jaetaan numeroitujen väliotsikoiden kohdalta
  (esim. `3.9 Lasiturva`). Jokaisen palan alussa toistetaan ehtojen osa ja yläotsikot, joten
  sekä haku että malli näkevät, koskeeko ote vapaaehtoista vakuutusta vai liikennevakuutusta.
  Otsikot tunnistetaan lihavoinnista ja fonttikoosta; ylä- ja alatunnisteet sekä
  sisällysluettelo poistetaan.
- **Rakenteinen vastaus.** Malli päättää ensin, löytyykö vastaus otteista, ja kirjoittaa vasta
  sitten vastauksen. Kieltäytymisen ja huomautuksen teksti tulee koodista, joten ne ovat aina
  samanlaisia ja kieltäytymiset voidaan laskea luotettavasti.
- **Pieni paikallinen malli.** Kehityskoneena on kannettava, jossa on RTX 3050 (6 Gt).
  Mallit valitaan sen mukaan, ja mallien nimet ovat vain tiedostossa `config.yaml`.
- **Pilveen siirrettävä.** Kielimallin kutsut kulkevat yhden rajapinnan kautta (`llm.py`), joten
  esimerkiksi Azure OpenAI voidaan ottaa käyttöön asetusmuutoksella. Tästä on valmiina runko.
- **Ei LangChainia.** Putki on kirjoitettu itse, jotta jokainen vaihe on näkyvissä ja
  testattavissa.

## Arviointi

### Miten testattiin

- **Aineisto:** LähiTapiolan julkiset ajoneuvovakuutuksen vakuutusehdot (voimassa 1.1.2026,
  29 sivua) ja ajoneuvovakuutuksen IPID-asiakirja. Asiakirjat on ladattu yhtiön julkisilta
  verkkosivuilta. Niitä ei jaeta tässä repositoriossa; tiedostot ja versiot on lueteltu
  tiedostossa `data/sources.yaml`.
- **Kysymykset:** 15 itse laadittua kysymystä (`eval/questions.json`): 12 sellaista, joihin
  ehdoissa on vastaus, ja 3, joihin ei ole (esim. vakuutuksen hinta). Jokaisen odotettu vastaus
  ja lähdekohta on tarkistettu käsin PDF:stä. Kysymyksissä on yhtiö (`"company": "LähiTapiola"`),
  ja haku rajataan sen asiakirjoihin, joten indeksiin lisätyt muiden yhtiöiden ehdot eivät
  vaikuta tuloksiin.
- **Ajo:** `python -m ehtobotti eval` kysyy jokaisen kysymyksen botilta ja tallentaa tulokset
  kansioon `results/`. Kone: kannettava, RTX 3050 6 Gt, Ollama. Satunnaisuus on lukittu
  (seed 42), joten ajot ovat toistettavissa.
- **Mittarit:**
  - *Haun osuvuus (hit@5):* löytyykö oikea ehtojen kohta viiden haetun otteen joukosta.
  - *Lähdeviitteiden tarkkuus:* kuinka suuri osa vastauksen viitteistä osuu odotettuihin kohtiin.
  - *Vastausten oikeellisuus:* toinen kielimalli (gemma3:4b) vertaa vastausta odotettuun.
    Arvioija ei koskaan ole sama malli, joka vastasi. Koska pieni arvioija on havaittu
    lempeäksi, vastaukset luetaan myös käsin.
  - *Kieltäytymiset:* kieltäytyykö botti, kun vastausta ei ole, ja turhaan, kun vastaus on.
  - *Keksityt lähdeviitteet:* viitteet kohtiin, joita mallille ei annettu.
  - *Vastausaika* kysymystä kohden.

### Tulokset

Viimeisin ajo (`results/20261029-003348_qwen3.5-4b_vector`), malli qwen3.5:4b, vektorihaku:

| Mittari | Tulos |
|---|---|
| Haun osuvuus (hit@5) | 100 % |
| Lähdeviitteiden tarkkuus | 83 % |
| Vastausten oikeellisuus (LLM-arvio) | 96 % |
| Vastausten oikeellisuus (käsin luettuna) | 11 / 12 |
| Oikeat kieltäytymiset | 3 / 3 |
| Turhat kieltäytymiset | 0 / 12 |
| Keksityt lähdeviitteet | 0 |
| Keskimääräinen vastausaika | 8,7 s |

Käsin luettuna väärä vastaus oli kysymys liikennevakuutuksen irtisanomisesta ajoneuvon
varastamisen jälkeen: malli vastasi varkausturvan ehdoilla. Lähdeviitteiden tarkkuutta laskee
lisäksi hinauskuluja koskeva kysymys, jonka vastaus oli oikein, mutta viite osoitti
kohtaan 3.16.5.1 (korvattavat kustannukset) eikä odotettuun kohtaan 3.16.3 (vakuutustapahtuma).

### Mallivertailu

Sama kysymyssarja kolmella asetuksella (28.10.2026, ennen viimeisintä pilkontamuutosta):

| Malli | Oikein (käsin) | Oikeat kieltäytymiset | Aika / kysymys |
|---|---|---|---|
| qwen3.5:4b | 10 / 12 | 3 / 3 | 6 s |
| qwen3.5:4b, ajattelu päällä | ei arvioitavissa | 2 / 3 | 116 s |
| qwen3:8b | 11 / 12 | 2 / 3 | 26 s |

Oletusmalliksi valittiin qwen3.5:4b. Suurempi qwen3:8b vastasi hieman useammin oikein, mutta
keksi vastauksen kysymykseen, johon ehdoissa ei ole vastausta ("voiko maksaa MobilePaylla?"),
ja oli neljä kertaa hitaampi. Vakuutusasiassa keksitty vastaus on pahempi virhe kuin
kieltäytyminen. Ajattelutila yhdessä JSON-muotoisen vastauksen kanssa tuotti enimmäkseen
tyhjiä vastauksia.

### Yksikkötestit

Putken osat on testattu pytestillä ilman kielimallia (`tests/`): pilkonta, PDF-tekstin
siivous, viitteiden poiminta ja tarkistus, arvioinnin mittarit, indeksin kirjoitus ja haku
oikealla Chroma-kannalla sekä käyttöliittymän logiikka. Kielimalli korvataan testeissä
valekomponentilla.

```
pytest
```

## Käyttö

### Asennus

Tarvitaan Python 3.11+ ja [Ollama](https://ollama.com).

```
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3.5:4b
ollama pull gemma3:4b          # arvioija
ollama pull bge-m3             # upotukset
```

### Aineisto

Lataa vakuutusehdot PDF-muodossa yhtiön verkkosivuilta ja lisää ne käyttöliittymän
välilehdellä **Lisää ehtoja**: valitse vakuutuslaji ja kirjoita vakuutusyhtiön nimi. Tiedosto
tallennetaan yhtiön omaan kansioon ja indeksoidaan heti.

Kansiorakenne, jossa jokaisella yhtiöllä on oma alikansionsa vakuutuslajin alla:

```
data/raw/
  auto/
    lahitapiola/
      ajoneuvovakuutusehdot.pdf
      ajoneuvovakuutus_ipid.pdf
    if/
      kaskovakuutusehdot.pdf
      liikennevakuutusehdot.pdf
  koti/
    lahitapiola/
      kotivakuutusehdot.pdf
```

Myös suoraan vakuutuslajin kansiossa olevat PDF:t toimivat. Yhtiö luetaan tiedostosta
`data/sources.yaml`, ei kansion nimestä. Jokaisella PDF:llä on siellä merkintä, jossa on
vakuutuslaji, polku vakuutuslajin kansion sisällä, nimi, versio, yhtiö ja lähdeviitteissä
käytettävä nimi:

```yaml
- product: auto
  file: if/kaskovakuutusehdot.pdf
  title: Kaskovakuutusehdot
  version: "1.1.2026"
  url: https://...
  label: If Autovakuutus
  company: If
```

Käyttöliittymä tekee merkinnän itse. Jos kopioit PDF:n kansioon käsin, lisää merkintä ja aja:

```
python -m ehtobotti ingest                  # rakentaa hakuindeksin
python -m ehtobotti ingest --show-chunks    # näyttää pilkotut palat tarkistusta varten
```

### Kysyminen

```
python -m ehtobotti ui                      # selainkäyttöliittymä, http://localhost:7860
python -m ehtobotti ask "Paljonko lasiturvan omavastuu on?" --product auto --company LähiTapiola
python -m ehtobotti eval                    # arviointi, tulokset kansioon results/
```

Käyttöliittymä toimii oletuksena vain omalla koneella. `ui --share` luo julkisen linkin; silloin
tiedostojen lisääminen on pois päältä.

## Rajoitukset

Tärkeimmät (kaikki: [docs/rajoitukset.md](docs/rajoitukset.md)):

- Kysymyssarja on pieni (15 kysymystä, vain autovakuutus), joten yksi kysymys muuttaa
  prosentteja paljon.
- Botti ei osaa kertoa suppean ja laajan vakuutuksen eroja, koska aineistosta ei käy ilmi,
  mitkä turvat kuuluvat kumpaankin.
- Botti ei tunne käyttäjän omaa vakuutuskirjaa. Vastaukset kertovat ehtojen yleiset säännöt.
- Pilkonta on tehty LähiTapiolan asiakirjojen rakenteen pohjalta. Muiden yhtiöiden PDF:t
  toimivat, mutta lähdeviitteet voivat jäädä sivutasolle.
- Pieni malli tekee yhä virheitä, kun kysymys koskee eri vakuutusta kuin haettu ote.

## Seuraavat askeleet

- Hybridihaku (BM25 + vektorihaku, suomen kielen sanojen perusmuodot) ja sen vertailu
  vektorihakuun.
- Kysymyksiä myös koti-, matka- ja henkilövakuutuksesta sekä toisen yhtiön ehdoista.
- Kysymykset, joihin ei löytynyt vastausta, talteen ehtojen ja ohjeiden kehittämistä varten.

## Rakenne

```
ehtobotti/
  config.py      asetukset tiedostosta config.yaml
  pdftext.py     PDF:n teksti ja asettelu (otsikot, ylä- ja alatunnisteet, sisällysluettelo)
  chunking.py    pilkonta ehtojen väliotsikoiden mukaan
  ingest.py      PDF → palat → upotukset → Chroma
  retrieve.py    vektorihaku vakuutuslaji- ja yhtiörajauksella
  generate.py    kehote, kielimallin kutsu, viitteiden tarkistus
  llm.py         Ollama-asiakas ja Azure OpenAI -runko
  evaluation.py  arvioinnin mittarit ja raportit
  cli.py         komennot ingest / ask / eval / ui
  ui.py          Gradio-käyttöliittymä
eval/questions.json   kysymyssarja
data/sources.yaml     lähdeasiakirjojen tiedot (PDF:t eivät ole repositoriossa)
results/              arviointiajojen tulokset
docs/rajoitukset.md   tunnetut rajoitukset
```
