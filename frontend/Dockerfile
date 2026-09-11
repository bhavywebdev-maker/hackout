FROM python:3.11-slim
WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import streamlit, os; p = os.path.join(os.path.dirname(streamlit.__file__), 'static', 'index.html'); open(p, 'w').write(open(p).read().replace('<title>Streamlit</title>', '<title>Bharat AI Banking</title>'))"
COPY frontend /app
COPY data /app/data
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
