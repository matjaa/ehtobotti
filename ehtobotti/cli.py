"""Command line interface. User-facing output is in Finnish."""

from __future__ import annotations

from typing import Optional

import typer

from ehtobotti.config import load_settings

app = typer.Typer(help="Ehtobotti – vakuutusehtojen kysymys-vastausavustaja (epävirallinen demo).")


@app.command()
def ingest(
    product: Optional[str] = typer.Option(None, help="Käsittele vain yksi tuote (koti, auto, matka, henkilo)."),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Tulosta palat tarkistusta varten."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Pilko vain, älä laske upotuksia tai kirjoita indeksiä."),
) -> None:
    """Rakenna tai päivitä hakuindeksi kansiosta data/raw."""
    from ehtobotti.chunking import format_chunk_debug
    from ehtobotti.ingest import build_chunks, write_index

    settings = load_settings()
    if product and product not in settings.products:
        raise typer.BadParameter(f"Tuntematon tuote: {product}. Vaihtoehdot: {', '.join(settings.products)}")

    chunks = build_chunks(settings, product, log=typer.echo)
    if not chunks:
        typer.echo(f"PDF-tiedostoja ei löytynyt kansiosta {settings.raw_dir}/<tuote>/.")
        raise typer.Exit(code=1)

    if show_chunks:
        for chunk in chunks:
            typer.echo(format_chunk_debug(chunk))

    if no_embed:
        typer.echo(f"Yhteensä {len(chunks)} palaa. Indeksiä ei päivitetty (--no-embed).")
        return

    from ehtobotti.llm import get_client

    client = get_client(settings)
    write_index(settings, chunks, client.embed)
    typer.echo(f"Indeksoitu {len(chunks)} palaa kokoelmaan '{settings.collection}'.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Kysymys vakuutusehdoista."),
    product: Optional[str] = typer.Option(None, help="Rajaa haku yhteen tuotteeseen (koti, auto, matka, henkilo)."),
    model: Optional[str] = typer.Option(None, help="Kielimalli; oletus config.yaml-tiedostosta."),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Haettavien otteiden määrä."),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Näytä haettujen otteiden teksti."),
) -> None:
    """Kysy vakuutusehdoista. Vastaus sisältää lähdeviitteet."""
    from ehtobotti.generate import answer_question, citation_label
    from ehtobotti.llm import get_client
    from ehtobotti.retrieve import IndexNotFoundError, VectorRetriever

    settings = load_settings()
    if product and product not in settings.products:
        raise typer.BadParameter(f"Tuntematon tuote: {product}. Vaihtoehdot: {', '.join(settings.products)}")
    if model and model not in settings.chat_models:
        typer.echo(f"Huom: mallia {model} ei ole config.yaml-tiedostossa, käytetään oletusasetuksia.")

    client = get_client(settings)
    try:
        retriever = VectorRetriever(settings, client.embed)
    except IndexNotFoundError:
        typer.echo("Hakuindeksiä ei löytynyt. Aja ensin: python -m ehtobotti ingest")
        raise typer.Exit(code=1)

    try:
        answer = answer_question(
            question, retriever=retriever, client=client, settings=settings,
            product=product, model=model, top_k=top_k,
        )
    except ConnectionError:
        typer.echo(f"Ollamaan ei saatu yhteyttä osoitteessa {settings.ollama_host}. Onko `ollama serve` käynnissä?")
        raise typer.Exit(code=1)

    typer.echo(f"\n{answer.text}\n")
    typer.echo("Lähteet:")
    for chunk in answer.sources:
        typer.echo(f"  {citation_label(chunk, settings)} {chunk.section_title} ({chunk.document}, osuvuus {chunk.score:.2f})")
        if show_chunks:
            typer.echo(f"    {chunk.text.replace(chr(10), chr(10) + '    ')}\n")
    if answer.unsupported_citations:
        bad = ", ".join(f"{c.product} {c.section} s. {c.page}" for c in answer.unsupported_citations)
        typer.echo(f"Varoitus: vastauksessa on viitteitä, joita ei löydy haetuista otteista: {bad}")
    typer.echo(f"\nMalli: {answer.model} | aika {answer.latency_s:.1f} s")


@app.command(name="eval")
def eval_command() -> None:
    """Aja arviointi (tulossa vaiheessa 3)."""
    typer.echo("Ei vielä toteutettu (vaihe 3).")
    raise typer.Exit(code=1)


@app.command()
def ui() -> None:
    """Käynnistä Gradio-käyttöliittymä (tulossa vaiheessa 4)."""
    typer.echo("Ei vielä toteutettu (vaihe 4).")
    raise typer.Exit(code=1)
