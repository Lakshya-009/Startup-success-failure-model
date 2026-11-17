# Startup Success Model

A machine learning model that evaluates how closely a startup resembles **failed** versus **successful** startups using two public datasets. It also retrieves the most similar failed and successful startups for a given startup description.

The model is designed to be accessed through a backend service. It does not contain any frontend or web application code of its own.

---

## 1. Features

- Predicts whether a startup resembles historically failed or successful startups.
- Uses funding, country, and industry as prediction features.
- Retrieves similar failed and successful startups using TF-IDF similarity.
- Supports individual and batch predictions.
- Provides confidence levels, missing-field information, and evaluation caveats.
- Exposes model functionality through a FastAPI REST API.
- Can be integrated with a Java Spring Boot backend.
- Loads the trained model once when the API server starts.

## 2. Project Structure

```text
startup-success-model/
│
├── startup_model/
│   ├── model.py
│   ├── cleaning.py
│   ├── similar.py
│   ├── train.py
│   ├── config.py
│   │
│   ├── data/
│   │   └── source CSV files
│   │
│   └── artifacts/
│       ├── model.joblib
│       └── model_card.json
│
├── reports/
│   ├── report.md
│   ├── metrics.json
│   └── evaluation.png
│
├── tests/
│   └── unit tests
│
├── examples/
│   └── sample input
│
├── main.py
├── requirements.txt
└── README.md
```

## 3. Requirements

- Python 3.10 or compatible version supported by the project dependencies.
- pip
- Virtual environment
- FastAPI
- Uvicorn
- Dependencies listed in `requirements.txt`

## 4. Installation and Quick Start

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If FastAPI and Uvicorn are not already included in `requirements.txt`, install them:

```bash
pip install fastapi uvicorn
```

### Step 4: Run unit tests

```bash
python -m unittest discover -s tests -v
```

The project currently includes 11 tests.

### Step 5: Load and use the model directly

```python
from startup_model import StartupRiskModel

# Load the model once
model = StartupRiskModel.load()

# Predict startup success similarity
result = model.predict({
    "funding_usd_millions": 2.5,
    "country": "India",
    "industry": "edu"
})

print(result)

# Batch predictions
results = model.predict_many([
    {
        "funding_usd_millions": 2.5,
        "country": "India",
        "industry": "edu"
    }
])

# Find similar startups
similar = model.similar_startups(
    "Quick grocery delivery application",
    industry="commerce_logistics",
    k_each=4
)

print(similar)
```

---

## 5. FastAPI Integration

The model can be exposed as a REST API using FastAPI. This allows other applications, including a Java Spring Boot backend, to communicate with the ML model over HTTP.

### API Architecture

```text
          Spring Boot Backend
                  |
                  | HTTP POST
                  |
                  v
            FastAPI Server
            localhost:8000
                  |
          StartupRiskModel
                  |
          -------------------
          |                 |
          v                 v
      Prediction       Similarity Search
          |                 |
          v                 v
       JSON Response     JSON Response
          |                 |
          -------------------
                  |
                  v
          Spring Boot Backend
```

### Step 1: Create `main.py`

Create a file named `main.py` in the root directory of the ML project.

```python
from fastapi import FastAPI, HTTPException
from startup_model import StartupRiskModel

# Initialize FastAPI
app = FastAPI(
    title="Startup Success Model API",
    description="REST API for startup success prediction and similarity search.",
    version="1.0.0"
)

# Load the model once when the server starts
model = StartupRiskModel.load()


# Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Startup Success Model API"
    }


# Startup Prediction Endpoint
@app.post("/predict")
def predict(features: dict):

    try:
        result = model.predict(features)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )


# Similar Startup Search Endpoint
@app.post("/similar-startups")
def similar_startups(payload: dict):

    try:
        text = payload.get("text")
        industry = payload.get("industry")
        k_each = payload.get("k_each", 4)

        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=422,
                detail="A non-empty startup description is required."
            )

        if not isinstance(k_each, int) or isinstance(k_each, bool) or k_each < 1:
            raise HTTPException(
                status_code=422,
                detail="k_each must be a positive integer."
            )

        result = model.similar_startups(
            text,
            industry=industry,
            k_each=k_each
        )

        return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Similar startup search failed."
        )
```

The model is loaded once when the application module is initialized instead of loading it for every API request.

---

## 6. Running FastAPI on Localhost

After creating `main.py`, start the FastAPI server from the project root directory.

### Step 1: Activate the virtual environment

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Explanation:

| Parameter | Purpose |
|---|---|
| `main` | Python file containing the FastAPI application |
| `app` | FastAPI application instance |
| `--host 0.0.0.0` | Allows the server to accept connections through available network interfaces |
| `--port 8000` | Runs the server on port 8000 |
| `--reload` | Automatically reloads the server when source files change; intended for development |

### Step 3: Verify the server

Once the server starts successfully, the API will be accessible at:

**Base URL:**

```text
http://localhost:8000
```

**Health Check:**

```text
http://localhost:8000/health
```

**Interactive API Documentation:**

```text
http://localhost:8000/docs
```

**Alternative API Documentation:**

```text
http://localhost:8000/redoc
```

The Swagger UI at `/docs` allows you to test the API endpoints directly from your browser.

To stop the server, press:

```text
CTRL + C
```

---

## 7. API Endpoints

The FastAPI application exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks whether the ML service is running |
| POST | `/predict` | Predicts startup success similarity |
| POST | `/similar-startups` | Retrieves similar failed and successful startups |

### 7.1 Health Check

**Request:**

```http
GET http://localhost:8000/health
```

**Example Response:**

```json
{
    "status": "healthy",
    "service": "Startup Success Model API"
}
```

### 7.2 Startup Prediction

**Request:**

```http
POST http://localhost:8000/predict
```

**Request Body:**

```json
{
    "funding_usd_millions": 2.5,
    "country": "India",
    "industry": "edu"
}
```

**Example Response:**

```json
{
    "success_score": 0.013,
    "band": "failure_like",
    "band_label": "closer to failed startups",
    "confidence": "low",
    "dataset_success_rate": 0.038,
    "missing_fields": [],
    "out_of_range": true,
    "funding_percentile": 0.016,
    "normalized_input": {
        "funding_usd_millions": 0.2,
        "country": "India",
        "industry": "edu"
    },
    "notes": [],
    "model_version": "0.1.0"
}
```

The above is an illustrative response showing the output structure.

### 7.3 Similar Startup Search

**Request:**

```http
POST http://localhost:8000/similar-startups
```

**Request Body:**

```json
{
    "text": "An online grocery delivery startup offering quick delivery services.",
    "industry": "commerce_logistics",
    "k_each": 4
}
```

**Response Structure:**

```json
{
    "failed": [
        {
            "name": "Example Failed Startup",
            "industry": "commerce_logistics",
            "country": "India",
            "funding": 2.5,
            "failure_reason": "Example reason",
            "similarity": 0.82
        }
    ],
    "succeeded": [
        {
            "name": "Example Successful Startup",
            "industry": "commerce_logistics",
            "country": "USA",
            "funding": 10.0,
            "exit_reason": "Example exit",
            "valuation": 100.0,
            "similarity": 0.76
        }
    ]
}
```

The example values are illustrative. Actual results depend on the bundled datasets and similarity search.

---

## 8. Testing the API

You can test the API using Swagger UI, Postman, or cURL.

### Using cURL

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"funding_usd_millions\":2.5,\"country\":\"India\",\"industry\":\"edu\"}"
```

**Similar Startup Search:**

```bash
curl -X POST "http://localhost:8000/similar-startups" \
-H "Content-Type: application/json" \
-d "{\"text\":\"Quick grocery delivery startup\",\"industry\":\"commerce_logistics\",\"k_each\":4}"
```

For Windows PowerShell, you can also use `Invoke-RestMethod`:

```powershell
$body = @{
    funding_usd_millions = 2.5
    country = "India"
    industry = "edu"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

---

## 9. Integrating with a Spring Boot Backend

The FastAPI service can be consumed by a Java Spring Boot backend using HTTP requests.

### Communication Flow

```text
User
 |
 v
Frontend Application
 |
 v
Spring Boot REST API
 |
 | HTTP POST
 v
FastAPI ML Service
 |
 v
Startup Success Model
 |
 v
Prediction / Similar Startup Results
 |
 | JSON Response
 v
Spring Boot Backend
 |
 v
Frontend Application
```

### FastAPI Base URL

For local development on the same machine:

```text
http://localhost:8000
```

### Spring Boot Configuration

In `application.properties`:

```properties
app.ml.base-url=${ML_SERVICE_BASE_URL:http://localhost:8000}
```

### Example Spring Boot HTTP Request

Using Spring's `RestClient`:

```java
RestClient restClient = RestClient.create(
    "http://localhost:8000"
);

Map<String, Object> features = Map.of(
    "funding_usd_millions", 2.5,
    "country", "India",
    "industry", "edu"
);

Map response = restClient.post()
    .uri("/predict")
    .body(features)
    .retrieve()
    .body(Map.class);
```

The Spring Boot backend should handle the response and pass the relevant information to the application layer.

**Important:** If Spring Boot and FastAPI run on the same computer, `localhost:8000` works. If they run in separate Docker containers, use the ML service's container or network hostname instead of `localhost`.

For production deployment, configure the ML service URL through an environment variable and avoid exposing the development server directly to the public internet.

---

## 10. Prediction Input Contract

The `predict()` method accepts the following fields:

| Field | Type | Description |
|---|---|---|
| `funding_usd_millions` | Number or null | Total funding raised, in millions of USD |
| `country` | String or null | Country of startup operation |
| `industry` | String or null | Startup industry category |
| `industry_bucket` | String or null | Accepted alias for industry |

### Supported Industry Categories

- `fintech`
- `health`
- `edu`
- `mobility_hw`
- `commerce_logistics`
- `software_ai`
- `media_social`
- `other`

Free-text industry descriptions are also accepted and mapped through keywords.

### Input Handling

- Missing values are allowed and represented using `null`.
- Unknown additional fields are ignored.
- Invalid values are ignored and included in `notes`.
- Funding currency conversion must be handled by the LLM extractor before calling the model.
- Funding must be provided in millions of USD.

---

## 11. Prediction Output Contract

The prediction endpoint returns:

| Field | Description |
|---|---|
| `success_score` | Comparative model score between 0 and 1 |
| `band` | `failure_like`, `mixed`, `success_like`, or `insufficient_data` |
| `band_label` | Human-readable interpretation |
| `confidence` | `none`, `low`, `medium`, or `moderate` |
| `dataset_success_rate` | Observed success share in the comparable training-data score group |
| `missing_fields` | Fields missing from the input |
| `out_of_range` | Indicates whether funding is below the training data's 10th percentile |
| `funding_percentile` | Funding percentile in the training data |
| `normalized_input` | Normalized input features |
| `notes` | Human-readable caveats |
| `model_version` | Model version |

**Important:** Ask founders for missing fields before scoring whenever possible. Display the model's confidence and notes alongside the result.

---

## 12. Model Evaluation

The model was evaluated using repeated 5-fold cross-validation on 1,134 startups:

- 298 successful startups
- 836 failed startups

| Metric | Result |
|---|---:|
| AUC | **0.83 ± 0.03** |
| Balanced Accuracy | **0.75** |
| Accuracy | 0.74 |
| Recall of Successes | ~0.79 |
| Precision of Successes | ~0.50 |

### Feature Contribution

| Feature | CV AUC |
|---|---:|
| Funding | 0.69 |
| Industry | 0.69 |
| Country | 0.60 |
| All Features Together | **0.83** |

Additional findings:

- Scores below 0.35 corresponded to actual failures 93% of the time.
- Scores above 0.65 corresponded to actual successes 61% of the time.
- The dataset base success rate was 26%.
- Training AUC was 0.85 compared with cross-validation AUC of 0.83.

Detailed evaluation is available in:

```text
reports/report.md
reports/metrics.json
reports/evaluation.png
```

---

## 13. Important Model Limitations

Read these limitations before relying on model predictions.

1. **Not a real-world probability:** `success_score` is a similarity score from a class-balanced model, not the actual probability that a startup will succeed.

2. **Dataset limitations:** The model uses notable companies, including unicorns and publicly documented failures. It does not represent the complete startup ecosystem.

3. **Funding bias:** Early-stage startups with little funding may fall outside the training data range and can be classified as closer to failed startups simply because of their size.

4. **Definition of success:** Success refers to becoming a unicorn or achieving an exit, not necessarily running a profitable small business.

5. **Country and industry bias:** Some country and industry signals reflect the composition of the source datasets.

6. **Missing-data bias:** Only rows with all required inputs observed are used for training. Using incomplete rows can produce misleadingly high accuracy by allowing the model to identify dataset-specific missing-data patterns.

7. **Model improvement:** Adding a dataset of ordinary startups, including those that survived, stalled, or shut down, would improve real-world relevance.

The model intentionally never reports a `high` confidence level.

---

## 14. Retraining the Model

To retrain the model using the bundled datasets:

```bash
python -m startup_model.train
```

This regenerates the model artifacts and evaluation reports.

Data cleaning is handled by:

```text
startup_model/cleaning.py
```

The cleaning pipeline:

- Re-parses funding values in the failures dataset because of unit inconsistencies.
- Removes duplicate startup names.
- Treats the 57 companies appearing in both datasets as failed.
- Relabels defunct or bankrupt unicorns as failed.
- Drops ambiguous devalued companies.
- Harmonizes country and industry vocabularies.

If the installed scikit-learn version differs from the version used to save the model, `load()` rebuilds it in memory from the bundled data.

---

## 15. Installing the Model as a Python Package

Instead of copying the model source files into another project, it can be installed directly from Git:

```bash
pip install git+https://github.com/<org>/<repo>.git
```

Then import it:

```python
from startup_model import StartupRiskModel

model = StartupRiskModel.load()
```

---

## 16. Recommended Development Workflow

For local development:

1. Activate the Python virtual environment.
2. Install dependencies.
3. Run unit tests.
4. Start the FastAPI server on port `8000`.
5. Verify `/health`.
6. Test `/predict` and `/similar-startups` through Swagger UI.
7. Start the Spring Boot backend on port `8080`.
8. Configure the backend to access the FastAPI base URL.
9. Test the complete frontend-to-backend-to-ML integration.

### Local Development URLs

| Component | URL |
|---|---|
| FastAPI ML Service | `http://localhost:8000` |
| FastAPI Swagger UI | `http://localhost:8000/docs` |
| FastAPI Health Check | `http://localhost:8000/health` |
| Spring Boot Backend | `http://localhost:8080` |

---

## 17. License and Data

The model uses two public startup datasets. Refer to their original sources and applicable licenses before redistributing the data or trained artifacts.

See the project reports and bundled dataset documentation for further details.
