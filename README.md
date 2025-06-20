# RAG‑powered Gemini Math Agent

Chat with your Math textbook. Upload files to `data/`, it will ingest as vector store. Agent uses Gemini + RAG + WolframAlpha math solver.

## Setup

```bash
git clone ...
cd repo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
edit keys
