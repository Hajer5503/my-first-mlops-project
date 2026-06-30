# Quick Demo

## Phase 2: Prediction API

```bash
# Start the API
poetry run python app.py

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Response: {"prediction": 0, "class_name": "setosa"}
```

## Phase 3: RAG Chatbot

```bash
# Start Ollama (if not running)
ollama serve

# In another terminal, start RAG API
poetry run python rag_api.py

# Ask a question
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the measurements of a virginica iris?"}'

# Response: Detailed answer from Mistral, grounded in iris_guide.txt
```

## View Experiments

```bash
# MLflow UI
poetry run mlflow ui
# Visit http://localhost:5000

# LangSmith Traces
# Visit https://smith.langchain.com (requires free account)
```