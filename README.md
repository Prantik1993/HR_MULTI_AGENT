# HR Intelligence Multi-Agent RAG

## Stack
- LangGraph 1.0.7 StateGraph
- ChromaDB + BM25 hybrid retrieval + CrossEncoder reranking
- FastAPI async backend + Streamlit UI
- Rotating log files (logs/app.log, logs/errors.log)

## Setup
pip install -r requirements.txt
cp .env.example .env
python -m app.ingestion.ingest
uvicorn api.main:app --reload
streamlit run ui/streamlit_app.py
tail -f logs/app.log
