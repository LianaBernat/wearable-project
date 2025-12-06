# Usa uma imagem oficial do Python
FROM python:3.11-slim

# Evita buffering de logs
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia o requirements e instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do app
COPY app.py .

# Streamlit roda por padrão na porta 8501
EXPOSE 8501

# Comando para iniciar o Streamlit quando o container subir
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
