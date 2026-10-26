PY ?= python
.PHONY: demo seed ingest ask eval test api ui
demo: seed ingest          ## build demo DB + docs and index them
seed:
	$(PY) -m agentic_rag.cli seed
ingest:
	$(PY) -m agentic_rag.cli ingest
ask:
	$(PY) -m agentic_rag.cli ask "$(Q)"
eval:
	$(PY) -m agentic_rag.cli eval
test:
	$(PY) -m pytest -q
api:
	$(PY) -m uvicorn agentic_rag.api:app --reload --port 8000
ui:
	$(PY) -m streamlit run app/streamlit_app.py
