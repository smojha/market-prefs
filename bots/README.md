# LLM Trading Agents Experiment Framework

This directory contains a framework for running controlled trading experiments with LLM-based agents.  
It was designed for behavioral experiments where bots (LLM traders) and human participants interact in a market (e.g. via [oTree](https://otree.readthedocs.io)).

---

## Features

- Run multiple LLM trading agents in parallel via Selenium‐controlled browser sessions.
- Support for **different LLM providers** (OpenAI, Anthropic, Mistral, Google, etc.).
- Configurable run metadata and storage of experiment data (`bot-data/`).
- CLI interface with arguments for:
  - Experiment link (`url`)
  - Model choice
  - Production vs debug runs
  - Notes / run comments
  - Timeout duration
  - **Number of bot traders** (use only the last N links from the config file).
- Metadata logging for each run (`runs.metadata`).
- Modular design:
  - `services/cli.py` – CLI argument parsing
  - `services/runner.py` – entry point for runs
  - `domain/experiment.py` – experiment orchestration
  - `agents/trading_agents.py` – bot logic
  - `adapters/web_driver.py` – browser management

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/smojha/market-prefs.git
cd llm-trading-experiments
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment variables
In the `bots/config/` directory, please create the following `.env` file, filling in the relevant **private** information.
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
GOOGLE_API_KEY=...
```

## Running an Experiment
### Command format
```bash
python -m bots.main URL MODEL_NAME [options]
```

### Required arguments
- **`URL`** 
The oTree SessionStartLinks page (e.g. http://localhost:8000/SessionStartLinks/abc123).
- **`MODEL_NAME`** 
Which LLM to use (gpt-3.5, gpt-4o, etc.).

### Optional arguments
- **`-p, --production`** 
Use bot-data/production-runs instead of debug runs.
- **`-m, --message`** 
Required when using --production; notes about this run.
- **`--timeout-minutes`** 
How long to let the experiment run before auto-exit (default 105).
- **`--num-bots N`** 
Only the last `N` links in the config file become bots.
If omitted, all subjects are bots by default.

## Examples
Run a debug experiment with all bots:
```bash
python -m bots.main http://localhost:8000/SessionStartLinks/test123 gpt-4o
```
Run a production experiment with notes and only 4 bots:
```bash
python -m bots.main http://localhost:8000/SessionStartLinks/jo2g7edd gpt-4o \
  -p -m "gpt-4o 2/3" --num-bots 4
```

## Output & Data
- **Run folders:**
Each run creates a folder inside `bot-data/debug-runs/` or `bot-data/production-runs/`, named `run-N`.
- **Metadata log:**
A line is appended to `runs.metadata` in the data folder, containing timestamp, run number, model, subject count, run comments, and experiment link.