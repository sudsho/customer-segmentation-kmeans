FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT 8050
EXPOSE 8050

CMD ["gunicorn", "dashboard.app:server", "--bind", "0.0.0.0:8050"]
