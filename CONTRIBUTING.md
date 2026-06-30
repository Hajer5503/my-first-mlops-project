# Contributing to My First MLOps Project

This project is an educational implementation of a complete MLOps pipeline. Contributions are welcome!

## Setup for Development

```bash
# Clone repo
git clone https://github.com/Hajer5503/my-first-mlops-project.git
cd my-first-mlops-project

# Install dev dependencies
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your LangSmith API key (optional for Phase 3)
```

## Running Tests

```bash
poetry run pytest tests/ -v
poetry run black --check .
poetry run flake8 .
```

## Project Structure

- **Phase 1:** MLOps fundamentals (MLflow, DVC)
- **Phase 2:** Production API (FastAPI, logging)
- **Phase 3:** LLMOps (RAG, LangChain, LangSmith)

## Common Issues

### Port Already in Use
```bash
Get-Process python | Stop-Process -Force  # Windows
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9  # Mac/Linux
```

### Ollama Connection Error
- Make sure Ollama is running: `ollama serve`
- Pull models: `ollama pull mistral` and `ollama pull nomic-embed-text`

### MLflow Registry Empty
- Run training: `poetry run dvc repro`
- Register model via UI: `poetry run mlflow ui`

## Future Improvements

- [ ] Model drift detection
- [ ] Advanced RAG (multi-modal, re-ranking)
- [ ] Kubernetes deployment
- [ ] Prometheus monitoring

---

Made with ❤️ by Hajer Abdelkefi