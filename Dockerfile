FROM python:3.12-slim
WORKDIR /srv
COPY rag_cag_project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home assistant \
    && mkdir -p /home/assistant/data \
    && chown assistant:assistant /home/assistant/data
COPY --chown=assistant:assistant rag_cag_project /srv
USER assistant
ENV RAG_REQUIRE_API_KEY=true
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
ENV RAG_DB_PATH=/home/assistant/data/assistant.sqlite3
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
