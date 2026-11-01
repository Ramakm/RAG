#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

uvicorn backend.main:app --port 8000 &
API_PID=$!
trap "kill $API_PID" EXIT

streamlit run app.py --server.port 8501
