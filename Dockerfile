FROM python:3.9-slim

WORKDIR /app
# Copy your dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app/
# Expose the port FastAPI runs on
EXPOSE 8000
# Command to run the server when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#