# Tunnetut rajoitukset

Rajoitukset, jotka botin käyttäjän ja arvioijan on hyvä tietää. Tiivistelmä on README:ssä.

## Aineisto

- Aineistona on LähiTapiolan julkisia asiakirjoja: ajoneuvovakuutuksen vakuutusehdot (1.1.2026)
  ja IPID-asiakirja sekä kotivakuutuksen vakuutusehdot ja IPID-asiakirja. Matka- ja
  henkilövakuutus puuttuvat. Arviointikysymykset koskevat toistaiseksi vain autovakuutusta.
- Pilkonta tunnistaa numeroidut väliotsikot lihavoinnin tai fonttikoon perusteella. Se on tehty
  ja testattu LähiTapiolan asiakirjoilla. Muiden yhtiöiden PDF:t toimivat, mutta jos otsikoita
  ei tunnisteta, teksti pilkotaan kappaleittain ja lähdeviitteessä on vain sivunumero.
- Lähde-PDF:ssä voi olla virheitä, jotka näkyvät viitteissä sellaisenaan. Esimerkiksi
  autovakuutusehtojen sivulla 15 kohta 3.23.2 on numeroitu "3.2.2".
- **Suppea ja laaja vakuutus:** IPID-asiakirjan tekstissä luetellaan kaikki vakuutusturvat,
  mutta siitä ei käy ilmi, mitkä turvat kuuluvat suppeaan ja mitkä laajaan vakuutukseen.
  Botti ei siksi osaa kertoa tasojen eroja. Tieto löytyy yleensä vakuutusyhtiön tuotesivulta
  tai vakuutuskirjasta, joita aineistossa ei ole.
- Botti ei tunne käyttäjän omaa vakuutuskirjaa (valitut turvat, omavastuut). Vastaukset
  kertovat ehtojen yleiset säännöt, eivät sitä, mitä yksittäinen asiakas on valinnut.

## Vastausten laatu

- Pieni kielimalli (qwen3.5:4b) ei aina erota, kumpaa vakuutusta ote koskee. Kysymykseen
  "Korvataanko tuulilasin rikkoutuminen liikennevakuutuksesta?" se vastaa "Kyllä" ja viittaa
  vapaaehtoisen vakuutuksen lasiturvaan, vaikka oikea vastaus on "Ei" (kysymys auto-12).
  Sama virhe toistui myös suuremmalla mallilla (qwen3:8b). Kun jokaisen palan alkuun lisättiin
  myös yläotsikot, vastaus oli oikein, mutta tämä perustuu yhteen ajoon.
- Suurempi malli qwen3:8b vastasi muuten hieman paremmin, mutta keksi vastauksen
  kysymykseen, johon ehdoissa ei ole vastausta, ja oli noin neljä kertaa hitaampi.
  Oletusmalliksi on siksi valittu qwen3.5:4b.
- Mallin "ajattelu"-tila (think) yhdessä JSON-muotoisen vastauksen kanssa tuotti testissä
  enimmäkseen tyhjiä vastauksia, joten se on oletuksena pois päältä.

## Arviointi

- Kysymyssarja on pieni (15 kysymystä, yksi tuote), joten prosenttiluvut vaihtelevat paljon
  yhden kysymyksen muuttuessa.
- Vastausten oikeellisuuden arvioi toinen pieni kielimalli (gemma3:4b). Se on havaittu
  liian lempeäksi: se on hyväksynyt vastauksen, jossa irtisanojaksi oli vaihtunut väärä
  osapuoli. Tulokset tarkistetaan siksi myös käsin.
- Odotetut lähteet on merkitty vakuutusehtojen kohtina tai asiakirjan sivuina (IPID). Jos
  sama asia löytyy myös muualta kuin merkitystä lähteestä, oikeaa viitettä ei lasketa oikeaksi.
