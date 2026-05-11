# Willow

Willow is a pure-Python coding agent.

## Install

From a local checkout:

```bash
scripts/install.sh
```

From a hosted repository:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/willow/main/scripts/install.sh \
  | WILLOW_REPO_URL=https://github.com/<owner>/willow.git bash
```

The installer uses the project script declared in `pyproject.toml`:

```toml
[project.scripts]
willow = "willow.cli:main"
```

## Usage

Open the TUI:

```bash
willow
```

Run one prompt headlessly:

```bash
willow -p "summarize this repository"
```
