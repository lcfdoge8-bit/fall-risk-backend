FROM python:3.12-slim

# Set work directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all backend files into the container
COPY . /code

# Give the container full read/write permissions for the SQLite database
RUN chmod -R 777 /code

# Run FastAPI on Port 7860 (Hugging Face's mandatory default port)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
