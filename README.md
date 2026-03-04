# HR Intelligence — Multi-Agent RAG

## Stack
- LangGraph 1.0.7 StateGraph  (real multi-agent routing)
- LCEL chains per specialist node
- ChromaDB 1.5.2 + BM25 hybrid search + CrossEncoder rerank
- FastAPI async backend + Streamlit chat UI

## Setup
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
# drop HR PDFs/docx into data/docs/
python -m app.ingestion.ingest
uvicorn api.main:app --reload   # terminal 1
streamlit run ui/streamlit_app.py  # terminal 2
