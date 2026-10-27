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
    if not answer.structured:
        typer.echo("Huom: malli ei vastannut pyydetyssä JSON-muodossa, vastaus tulkittiin tekstinä.")
    typer.echo("Haetut otteet (eivät sisältäneet vastausta):" if answer.refused else "Lähteet:")
    for chunk in answer.sources:
        part = chunk.metadata.get("part") or chunk.document
        typer.echo(f"  {citation_label(chunk, settings)} {chunk.section_title} ({part}, osuvuus {chunk.score:.2f})")
        if show_chunks:
            typer.echo(f"    {chunk.text.replace(chr(10), chr(10) + '    ')}\n")
    if answer.unsupported_citations:
        bad = ", ".join(f"{c.product} {c.section} s. {c.page}" for c in answer.unsupported_citations)
        typer.echo(f"Varoitus: vastauksessa on viitteitä, joita ei löydy haetuista otteista: {bad}")
    typer.echo(f"\nMalli: {answer.model} | aika {answer.latency_s:.1f} s")


@app.command(name="eval")
def eval_command(
    model: Optional[str] = typer.Option(None, help="Arvioitava kielimalli; oletus config.yaml-tiedostosta."),
    retrieval: str = typer.Option("vector", help="Hakutapa: vector tai hybrid (vaihe 5)."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Arvioijamalli; oletus config.yaml-tiedostosta."),
    no_judge: bool = typer.Option(False, "--no-judge", help="Ohita vastausten LLM-arvio (nopeampi)."),
    product: Optional[str] = typer.Option(None, help="Aja vain yhden tuotteen kysymykset."),
    limit: Optional[int] = typer.Option(None, help="Aja vain N ensimmäistä kysymystä."),
    verified_only: bool = typer.Option(False, "--verified-only", help="Aja vain tarkistetut kysymykset."),
) -> None:
    """Aja arviointi kysymyssarjalla ja tallenna tulokset kansioon results/."""
    from ehtobotti import evaluation
    from ehtobotti.llm import get_client
    from ehtobotti.retrieve import IndexNotFoundError, VectorRetriever

    settings = load_settings()
    if retrieval != "vector":
        typer.echo("Vain vector-haku on toteutettu. Hybridihaku tulee vaiheessa 5.")
        raise typer.Exit(code=1)

    model = model or settings.default_chat_model
    judge = None if no_judge else (judge_model or settings.judge_for(model))
    if judge == model:
        typer.echo("Arvioijamallin on oltava eri kuin arvioitava malli.")
        raise typer.Exit(code=1)

    questions = evaluation.load_questions(settings.questions_path)
    if product:
        questions = [q for q in questions if q.product == product]
    if verified_only:
        questions = [q for q in questions if q.verified]
    questions = questions[:limit] if limit else questions
    if not questions:
        typer.echo("Ei kysymyksiä valituilla rajauksilla.")
        raise typer.Exit(code=1)
    unverified = sum(not q.verified for q in questions)
    if unverified:
        typer.echo(f"Huom: {unverified}/{len(questions)} kysymystä on tarkistamatta (verified: false).")

    client = get_client(settings)
    try:
        retriever = VectorRetriever(settings, client.embed)
    except IndexNotFoundError:
        typer.echo("Hakuindeksiä ei löytynyt. Aja ensin: python -m ehtobotti ingest")
        raise typer.Exit(code=1)

    try:
        results = evaluation.run(questions, retriever=retriever, client=client, settings=settings,
                                 model=model, judge_model=judge, log=typer.echo)
    except ConnectionError:
        typer.echo(f"Ollamaan ei saatu yhteyttä osoitteessa {settings.ollama_host}. Onko `ollama serve` käynnissä?")
        raise typer.Exit(code=1)

    # Thinking changes results a lot, so it is part of the run's name.
    label = f"{model}-think" if settings.chat_options(model).think else model
    summary_md = evaluation.summary_markdown(evaluation.summarize(results), model=label, retrieval=retrieval,
                                             judge_model=judge, top_k=settings.top_k)
    csv_path = evaluation.write_results(results, summary_md, settings.results_dir, label, retrieval)
    typer.echo(f"\n{summary_md}\nTulokset: {csv_path} (+ .md)")


@app.command()
def ui() -> None:
    """Käynnistä Gradio-käyttöliittymä (tulossa vaiheessa 4)."""
    typer.echo("Ei vielä toteutettu (vaihe 4).")
    raise typer.Exit(code=1)
