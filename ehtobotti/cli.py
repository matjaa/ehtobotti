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
def ask(question: str) -> None:
    """Kysy vakuutusehdoista (tulossa vaiheessa 2)."""
    typer.echo("Ei vielä toteutettu (vaihe 2).")
    raise typer.Exit(code=1)


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
