# issue-tools

Command line utilities for indexing GitHub issues and pull requests into a local
vector database, performing semantic search, clustering similar work, and
tracking model usage costs.

## Features

- **Daily incremental indexing** – fetches the latest GitHub issues/PRs and
  stores document embeddings on disk (initial run indexes 10 records by
  default).
- **Local vector database** – persists embeddings and metadata in SQLite so
  embeddings are loaded on demand rather than into memory.
- **GitHub filter previews** – exercise filter strings against the live GitHub
  search API to verify the results before running a full indexing job.
- **Semantic search with filters** – run natural language searches that also
  honour GitHub-style filters such as `label:bug`, `state:open`,
  `date:>=2024-01-01`, and `is:issue` before scoring results.
- **K-means clustering** – group similar issues or pull requests with the
  stored embeddings to uncover related work.
- **Streaming updates** – quickly check for issues or pull requests that have
  changed since the last indexing run.
- **Evaluation harness** – measure retrieval quality with precision@k and MRR
  metrics to baseline future experiments.
- **Token accounting** – keep a running token count and estimated costs for
  Gemini embedding and Gemini 2.5 Flash Lite LLM calls.

## Requirements

- Python 3.11+
- GitHub personal access token with `repo` scope (set in the environment
  variable listed in the config, default `GITHUB_TOKEN`).
- Gemini API key from Google AI Studio (environment variable from config,
  default `GEMINI_API_KEY`).

## Installation

### Quick start (recommended)

```bash
./scripts/install.sh --with-tests
```

This creates a managed virtual environment in `.venv`, installs the package,
and exposes the CLI through the convenience wrapper:

```bash
./issue-tools --help
```

Set `PYTHON_BIN` if you need a specific interpreter (defaults to `python3.11`).

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

## Configuration

The CLI reads options from `config.toml` in the project root (optional). Example:

```toml
repository = "my-org/my-repo"
index_interval_hours = 24
initial_index_limit = 10
llm_model = "models/gemini-2.5-flash-lite"

[embedding]
document_model = "gemini-embedding-001"
document_task_type = "RETRIEVAL_DOCUMENT"
query_model = "gemini-embedding-001"
query_task_type = "RETRIEVAL_QUERY"
max_chars = 6000
batch_size = 8

[cost.per_1k_tokens]
"models/gemini-embedding-001" = 0.00013
"models/gemini-2.5-flash-lite" = 0.10
```

Ensure the data directory defined by `data_dir` exists or let the CLI create it
on first run.

Environment variables such as the GitHub token and Gemini API key can be stored
in a `.env` file. Copy `.env.sample` to `.env` and fill in your credentials:

```bash
cp .env.sample .env
```

The CLI loads this file automatically before reading configuration so the
variables are available.

## Usage

Run `issue-tools --help` to view the available commands. Key workflows:

### Tab completion

Enable shell completions to see command and option suggestions when you press
<kbd>Tab</kbd> in your terminal:

```bash
issue-tools --install-completion
```

Follow the printed instructions to activate completions for your shell. You can
re-run the command with `--show-completion` to preview the script without
installing it.

### Index GitHub issues/pull requests

```bash
issue-tools index --repo my-org/my-repo
```

The first run fetches up to 10 issues/PRs (configurable). Subsequent runs only
refresh items updated since the last daily indexing window. A spinner is shown
while fetching and embedding, and token/cost totals are printed on completion.

Use `--filter` to index a subset of documents that match GitHub search syntax
without advancing the scheduled full index window:

```bash
issue-tools index --repo my-org/my-repo --filter "label:bug state:open" --limit 20
```

Filtered runs hydrate each matching issue or pull request before embedding and
skip documents whose `updated_at` timestamp matches what is already stored.

### Semantic search

```bash
issue-tools search "crash on startup label:bug state:open" --repo my-org/my-repo
```

Results honour the supplied filters before ranking semantically and include
clickable GitHub URLs for convenience.

### Preview GitHub filters

```bash
issue-tools filter "label:bug state:open" --repo my-org/my-repo --limit 5
```

This uses GitHub's REST search API to fetch matching issues or pull requests so
you can confirm filters before indexing.

### Cluster related work

```bash
issue-tools cluster --repo my-org/my-repo --query "crash on startup label:bug" --limit 40
```

The query is used to perform a semantic search (while honouring any GitHub-style
filters). The top results are then clustered, with the optimal number of
clusters chosen automatically unless `--k` is supplied. Pass `--limit 0` to
cluster every matching document instead of truncating the sample. Include one or
more `number:<id>` filters to anchor the search around specific issues or pull
requests before clustering the most similar matches.

### Stream recent updates

```bash
issue-tools stream --repo my-org/my-repo
```

### Evaluate retrieval quality

Provide a JSON file with queries and expected issue numbers, e.g.:

```json
[
  {"query": "crash in payment flow", "expected_numbers": [42, 105]},
  {"query": "improve docs", "expected_numbers": [12]}
]
```

Then run:

```bash
issue-tools evaluate evaluations/sample.json --repo my-org/my-repo
```

### Inspect token usage

```bash
issue-tools tokens --show-lifetime
```

### HTTP API

The command implementations are also exposed through a FastAPI application. Launch the server with Uvicorn:

```bash
uvicorn issue_tools.api:app --host 0.0.0.0 --port 8000
```

Endpoints mirror the CLI commands. For example, trigger an indexing run with:

```bash
curl -X POST http://localhost:8000/index \
     -H "Content-Type: application/json" \
     -d '{"repo": "my-org/my-repo"}'
```

All responses include the same structured data returned by the CLI, including token usage summaries when available.

### MCP server

Agents can call the same commands over the Model Context Protocol. Run the FastMCP server with:

```bash
python -m issue_tools.mcp_server
```

The MCP tools (`index`, `search`, `cluster`, `stream`, `inspect_db`, `evaluate`, and `tokens`) return JSON payloads that match the API and CLI results, so agents and humans share a common specification.

## Development

Run the test suite with:

```bash
pytest
```

## Notes

- Document embeddings are generated with Gemini retrieval embeddings for
  indexing and Gemini query embeddings for search.
- `models/gemini-2.5-flash-lite` is the default LLM model name tracked for downstream usage.
- Embeddings are truncated to the configured character limit before indexing to
  stay within model context windows.
- Each stored document includes metadata about the embedding model and indexing
  timestamp to ease future re-indexing or model upgrades.
