# My First MLOps Project

A complete **end-to-end MLOps pipeline** covering model training, deployment, monitoring, and LLM integration.

Built as a structured learning progression through 4 phases of MLOps mastery.

## Quick Start

```bash
# Install dependencies
poetry install

# Train models
poetry run dvc repro

# Start the prediction API (Phase 2)
poetry run python app.py

# Start the RAG chatbot API (Phase 3)
poetry run python rag_api.py
```

APIs available at:
- Prediction API: `http://localhost:8000/docs`
- RAG Chatbot: `http://localhost:8001/docs`

---

## Phase Overview

### Phase 0: Engineering Foundations ✅
**Goals:** Python, Git, Docker, Linux fundamentals

**Key Learnings:**
- Poetry for dependency management
- Docker containerization
- GitHub workflow and best practices
- CLI/shell scripting

**Tools:** Python 3.11, Poetry, Docker, Git

---

### Phase 1: Model Training & Experiment Tracking ✅
**Goals:** MLflow, DVC, reproducible pipelines

**Architecture:**
data/iris.csv (versioned with Git)
↓
save_dataset.py (DVC stage)
↓
data/iris.csv (prepared)
↓
train.py (DVC stage + MLflow tracking)
↓
mlruns/ (MLflow experiment artifacts)
↓
models:/iris-classifier@production (Model Registry)
**Key Learnings:**
- MLflow 2.22.5 for experiment tracking (not 3.x — has Windows FastAPI UI bugs)
- DVC for data versioning and reproducible pipelines
- Model Registry with aliases (`@production`, `@staging`)
- Never commit `mlruns/` to Git (contains Windows absolute paths → breaks CI)
- GitHub Actions for CI/CD validation

**Tools:** MLflow, DVC, scikit-learn, GitHub Actions

**Metrics:**
- Model: Random Forest Classifier
- Dataset: Iris (150 samples, 4 features)
- Best accuracy: 100% on test set

---

### Phase 2: Production API & Monitoring ✅
**Goals:** FastAPI deployment, prediction logging, observability

**Architecture:**
Model Registry (MLflow)
↓
FastAPI Server (app.py)
↓
POST /predict → Pydantic validation
↓
Model inference
↓
Prediction logging (logs/predictions.log)
↓
Rotating file handler (10MB max, 5 backups)
**Endpoints:**
- `GET /health` — health check
- `POST /predict` — iris classification
- `GET /metrics` — prediction statistics

**Key Learnings:**
- FastAPI with Pydantic for request validation
- Uvicorn ASGI server
- Structured logging with rotating file handlers
- Metrics endpoint for monitoring prediction distribution
- Port binding and process management

**Tools:** FastAPI, Uvicorn, Pydantic, logging

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

---

### Phase 3: LLMOps & RAG ✅
**Goals:** Retrieval-Augmented Generation, LangChain orchestration, LangSmith observability

**Architecture:**
Question
↓
Vector Embeddings (nomic-embed-text)
↓
FAISS Vector Store (retrieval)
↓
Retrieved Documents (context augmentation)
↓
Ollama LLM (Mistral 7B)
↓
LangSmith Tracing (observability)
↓
Answer
**RAG Flow:**
1. **Retrieve:** FAISS searches vector DB for relevant iris knowledge
2. **Augment:** Retrieved docs added to prompt
3. **Generate:** Mistral LLM answers with context

**Endpoints:**
- `GET /health` — health check
- `POST /ask` — RAG question answering

**Key Learnings:**
- LangChain orchestration (`langchain-ollama`, `langchain-text-splitters`)
- FAISS vector database for semantic search
- Embedding models (nomic-embed-text for embeddings, Mistral for generation)
- LangSmith for LLM observability (tracing, latency monitoring)
- RAG architecture benefits (grounded, verifiable answers)

**Tools:** LangChain, FAISS, Ollama (Mistral), LangSmith

**Example Request:**
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the measurements of a setosa iris?"}'
```

**LangSmith Monitoring:**
- Full trace of retrieval + generation
- Latency breakdown per component
- Prompt/response inspection
- Token usage tracking

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.11 |
| **Package Mgmt** | Poetry | Latest |
| **ML Training** | scikit-learn | Latest |
| **Experiment Tracking** | MLflow | 2.22.5 |
| **Data Versioning** | DVC | Latest |
| **Prediction API** | FastAPI | 0.138.0 |
| **ASGI Server** | Uvicorn | 0.49.0 |
| **LLM Framework** | LangChain | Latest |
| **Vector DB** | FAISS | Latest |
| **Local LLM** | Ollama (Mistral 7B) | Latest |
| **LLM Observability** | LangSmith | Cloud |
| **Containerization** | Docker | Latest |
| **CI/CD** | GitHub Actions | Latest |

---

## Project Structure
```

my-first-mlops-project/
├── app.py                    # Phase 2: Prediction API
├── rag_api.py                # Phase 3: RAG chatbot API
├── train.py                  # Model training + MLflow logging
├── save_dataset.py           # Data loading for DVC pipeline
├── predict.py                # Inference utility
├── Dockerfile                # Container image
├── docker-compose.yml        # Multi-service orchestration
├── pyproject.toml            # Poetry dependencies + config
├── poetry.lock               # Locked dependency versions
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # DVC lock file
├── .github/workflows/
│   └── test.yml              # GitHub Actions CI pipeline
├── data/
│   └── iris.csv              # Iris dataset (Git tracked)
├── docs/
│   └── iris_guide.txt        # Knowledge base for RAG
├── logs/
│   └── predictions.log       # API prediction logs
├── results/
│   └── metrics.json          # Training metrics
├── tests/
│   └── test_train.py         # Unit tests
└── .gitignore                # Git exclusions
---

## Key Lessons Learned

### MLOps Principles
1. **Reproducibility:** DVC pipelines ensure exact same results
2. **Versioning:** Track data, models, and code together
3. **Monitoring:** Predict in production with observability
4. **Modularity:** Separate concerns (train, serve, monitor, observe)

### Common Pitfalls Avoided
- ❌ Never commit `mlruns/` → use `.gitignore`
- ❌ Never commit secrets (API keys) → use `.env` + `.gitignore`
- ❌ Don't use MLflow 3.x on Windows (FastAPI UI broken)
- ❌ Don't track small data with DVC remote → commit to Git instead
- ❌ Don't load models at startup without error handling

### Windows-Specific Issues Resolved
- PowerShell PATH management for `poetry` and `docker`
- WSL integration for Linux CLI tools
- Port binding conflicts (`Ctrl+C`, `Get-Process`)
- Docker Desktop integration with local file volumes

---

## Running Locally

### Prerequisites
- Python 3.11+
- Poetry
- Docker & Docker Desktop
- Ollama (for Phase 3 RAG)

### Setup

```bash
# 1. Clone repo
git clone https://github.com/Hajer5503/my-first-mlops-project.git
cd my-first-mlops-project

# 2. Install dependencies
poetry install

# 3. Download LLM (Phase 3 only)
ollama pull mistral
ollama pull nomic-embed-text

# 4. Create .env for LangSmith (Phase 3)
echo "LANGCHAIN_TRACING_V2=true" > .env
echo "LANGCHAIN_API_KEY=<your-key>" >> .env
echo "LANGCHAIN_PROJECT=iris-rag" >> .env
```

### Train Models (Phase 1)
```bash
poetry run dvc repro
poetry run mlflow ui  # View experiments at http://localhost:5000
```

### Run Prediction API (Phase 2)
```bash
poetry run python app.py
# Visit http://localhost:8000/docs
```

### Run RAG Chatbot (Phase 3)
```bash
# Start Ollama server first
ollama serve

# In another terminal
poetry run python rag_api.py
# Visit http://localhost:8001/docs
```

### Run Tests
```bash
poetry run pytest tests/ -v
poetry run black .      # Code formatting
poetry run flake8 .     # Linting
```

---

## Future Improvements

- [ ] **Phase 4:** Model drift detection + automated retraining
- [ ] **Cloud Deployment:** AWS SageMaker or GCP Vertex AI
- [ ] **Advanced RAG:** Multi-modal embeddings, hybrid search, re-ranking
- [ ] **Monitoring:** Prometheus + Grafana dashboards
- [ ] **A/B Testing:** Compare model versions in production
- [ ] **LLMOps:** Prompt versioning, few-shot evaluation
- [ ] **Kubernetes:** Containerize and orchestrate services

---

## Learning Outcomes

After completing this project, you'll understand:

✅ **Model Training & Experiment Tracking** (MLflow, DVC)  
✅ **Production API Design** (FastAPI, validation, logging)  
✅ **Prediction Monitoring** (metrics, logging, observability)  
✅ **LLM Orchestration** (LangChain, RAG, embeddings)  
✅ **LLM Observability** (LangSmith tracing, debugging)  
✅ **CI/CD Automation** (GitHub Actions, automated testing)  
✅ **Containerization** (Docker, multi-service composition) 

## Quick Links

- 📖 **[Contributing Guide](CONTRIBUTING.md)** — How to extend this project
- 🎬 **[Demo Script](DEMO.md)** — Quick start examples
- 🔗 **[GitHub Repo](https://github.com/Hajer5503/my-first-mlops-project)** — Full source code
- 📊 **[LinkedIn](https://linkedin.com/in/hajer-abdelkefi-ab31452ab)** — My professional profile

---

---

## Author

**Hajer Abdelkefi** | Data Science Engineering Student @ ESPRIT  
🔗 LinkedIn: [linkedin.com/in/hajer-abdelkefi](https://linkedin.com/in/hajer-abdelkefi-ab31452ab)  
🔗 GitHub: [github.com/Hajer5503](https://github.com/Hajer5503)

---

## License

MIT
