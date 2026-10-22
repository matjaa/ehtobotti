# Ehtobotti – insurance terms RAG assistant

## Purpose
Portfolio project for an AI trainee application (Finnish insurance/financial sector).
A local RAG assistant that answers questions about publicly available Finnish insurance
terms (vakuutusehdot) and cites the exact section and page for every claim.
The project must demonstrate: a working RAG pipeline, honest evaluation with metrics,
a clean CLI + simple UI, and business awareness (privacy, refusal when unsure, cloud-ready design).

This is an unofficial demo, not affiliated with any insurer. Never use company logos.

## Environment and hardware constraints
- OS: Windows. Python 3.11+, virtualenv in `.venv`. Use paths that work on Windows (`pathlib`).
- LLM runtime: Ollama running locally (http://localhost:11434). Python client: `ollama`.
- Laptop GPU: RTX 3050 6 GB VRAM, 16 GB RAM. This drives every model decision:
  - Chat models: `qwen3.5:4b` (primary), `gemma3:4b` (comparison), `qwen3:8b` (optional, partially offloads to CPU, slower).
  - Embedding model: `bge-m3`.
  - Keep context ≤ 4096 tokens unless measured otherwise. Retrieve ~5 chunks of 300–400 tokens.
  - Run only one chat model at a time (eval runs models sequentially, never in parallel).
  - Pass `think=False` only to models that support thinking (Qwen). Gemma does not; make this a per-model config flag.
- Never hardcode model names in code; read them from config.

## Language conventions
- Code, identifiers, comments, docstrings, commit messages, tests: English.
- README.md: Finnish (written for a Finnish recruiter; lead with business value, then tech).
- User-facing UI and CLI output text: Finnish.
- System prompt for the LLM: Finnish (gives better Finnish answers from small models).

## Data
- Products: kotivakuutus, autovakuutus, matkavakuutus, henkilö-/tapaturmavakuutus.
- Source PDFs go in `data/raw/<product>/`. The user downloads them manually.
- `data/raw/` and `data/index/` are gitignored (do not redistribute the PDFs).
  Instead keep `data/sources.yaml` listing product, document title, version/date and public URL.
- Only public documents and synthetic questions. No personal data anywhere.

## Architecture
```
ehtobotti/
  config.py        # settings from config.yaml (models, top_k, chunk sizes, provider)
  ingest.py        # PDF -> text (PyMuPDF) -> section-based chunks -> embeddings -> Chroma
  chunking.py      # split by insurance terms section numbering, keep metadata
  retrieve.py      # vector search; later hybrid (BM25 + vector, reciprocal rank fusion)
  generate.py      # prompt building, LLM call, citation formatting
  llm.py           # provider abstraction: OllamaClient now, AzureOpenAIClient stub later
  cli.py           # typer CLI: ingest / ask / eval
  ui.py            # Gradio app
eval/
  questions.json   # test set
  run_eval.py      # metrics, writes results/
tests/             # pytest unit tests
data/ results/ config.yaml requirements.txt README.md
```

### Chunking rules
- Split on the terms' own section headings (e.g. `4.2 Vuotovahingot`, regex on leading section numbers).
- If a section is longer than the max chunk size, split it further on paragraphs, keeping the section heading as a prefix in every sub-chunk.
- Metadata per chunk: `product`, `document`, `version`, `section` (e.g. "4.2"), `section_title`, `page`, `chunk_id`.
- Write a debug command/flag that prints chunks so splitting can be inspected by eye.

### Retrieval
- Phase 1: vector search with `bge-m3` embeddings in a persistent Chroma collection, optional `product` filter.
- Phase 2: add BM25 (`rank_bm25`) with Finnish-aware tokenization (lowercase + Snowball Finnish stemmer) and fuse with reciprocal rank fusion. Compare against vector-only in eval.

### Generation
System prompt (Finnish) must enforce:
1. Answer only from the provided excerpts.
2. Cite every claim as `[Tuote, kohta X.Y, s. N]`.
3. If the excerpts do not contain the answer, say so clearly ("En löytänyt vastausta vakuutusehdoista.") instead of guessing.
4. Short note that this is not an official decision on compensation.
Return the answer plus the list of retrieved sources so the CLI/UI can show them.

### Provider abstraction
`llm.py` defines a small interface (`chat(messages, **opts)`, `embed(texts)`).
Ollama is the only working implementation. An Azure OpenAI client may exist as a documented stub
so switching to a cloud model is a config change. Do not add cloud dependencies before phase 3.

## CLI
```
python -m ehtobotti ingest                        # build/refresh the index from data/raw
python -m ehtobotti ask "kysymys" [--product koti] [--model qwen3.5:4b] [--show-chunks]
python -m ehtobotti eval [--model ...] [--retrieval vector|hybrid]
python -m ehtobotti ui                            # launch Gradio
```

## Evaluation
- `eval/questions.json`: ~40 questions, ~10 per product, including ~8 that the terms cannot answer.
  Schema: `id, product, question, expected_answer, expected_sections[], answerable (bool)`.
- Claude may draft candidate questions, but every expected answer and section must be verified by the user against the PDF. Never edit expected answers to make results look better.
- Metrics:
  - retrieval hit@5 (expected section among retrieved chunks)
  - citation accuracy (cited sections match expected)
  - answer correctness (LLM-as-judge with a different model than the one answering, plus manual spot checks)
  - refusal accuracy on unanswerable questions
  - average latency per answer
- Output: `results/<timestamp>_<model>_<retrieval>.csv` and a markdown summary table to paste into README.

## UI (Gradio)
- Chat view, product dropdown (Kaikki / Koti / Auto / Matka / Henkilö), sources shown under each answer (expandable, with section, page and the excerpt text).
- Visible disclaimer: unofficial demo, not an official compensation decision.

## Roadmap (1–2 weeks)
1. Project skeleton, config, ingest + chunking with debug output, unit tests for chunking.
2. Vector retrieval + `ask` command end to end with citations and refusal.
3. Eval set + `eval` command, first results table.
4. Gradio UI.
5. Hybrid retrieval + model comparison (qwen3.5:4b vs gemma3:4b, optionally qwen3:8b), update results.
6. Finnish README (problem, demo GIF/screenshots, architecture, results, limitations, next steps), cleanup.
Optional if time: automatic product detection, conversation follow-ups, query logging for unanswered questions, Dockerfile.

## Working rules for Claude Code
- Work in small steps; after each change run the relevant command or tests and show the result.
- Confirm Ollama is reachable and models are pulled before debugging model output (`ollama list`, `ollama ps`).
- Keep `requirements.txt` up to date; justify every new dependency.
- Prefer simple, readable code over frameworks (no LangChain/LlamaIndex unless agreed; the point is to show understanding of the pipeline).
- Do not commit anything under `data/raw/` or `data/index/`.
- When something is a design trade-off, explain it briefly so it can go into the README.
