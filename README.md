# Fake News Detection Using Deep Learning

### Word Embedding + Bidirectional Long Short-Term Memory (BiLSTM)

A deep learning-powered web application developed to classify news headlines as **Real or Fake** using Natural Language Processing (NLP), Word Embeddings, and a Bidirectional Long Short-Term Memory (BiLSTM) neural network.

The system uses the **Indian Fake News Dataset (IFND)** and provides an interactive dashboard for real-time headline classification, batch prediction, model analysis, and prediction history.

## Project Overview

The rapid growth of digital news platforms has increased the spread of misinformation. This project implements a deep learning-based solution that learns sequential patterns in news headlines to classify them as Real or Fake.

Unlike traditional machine learning methods, the BiLSTM architecture processes text sequences in both forward and backward directions, enabling the model to capture contextual information from surrounding words.

The application combines a trained neural network with a FastAPI backend, SQLite database, and responsive web interface.

## Key Features

- **Fake News Detection:** Classify news headlines into Real or Fake categories.
- **BiLSTM Neural Network:** Uses bidirectional recurrent neural networks for text classification.
- **Word Embeddings:** Converts textual tokens into dense numerical representations.
- **Text Preprocessing:** Normalization, tokenization, and sequence encoding.
- **Single Headline Prediction:** Analyze individual news headlines.
- **Batch Classification:** Process up to 200 headlines simultaneously.
- **Prediction Scores:** Display the model's estimated Fake-class probability.
- **Word Effect Analysis:** Estimate the influence of individual words on predictions.
- **Prediction History:** Store previous classifications using SQLite.
- **Random Test Samples:** Retrieve test headlines for evaluation.
- **Model Performance Dashboard:** Display evaluation metrics and confusion matrix results.
- **REST API:** FastAPI endpoints for integration with other applications.
- **Interactive Interface:** Responsive dashboard built with HTML, CSS, and JavaScript.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | Bidirectional LSTM |
| Text Representation | Word Embedding |
| NLP Processing | Tokenization, Sequence Padding |
| Data Processing | Pandas, NumPy |
| Model Format | Keras (.keras) |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| Web Server | Uvicorn |
| Testing | Pytest, HTTPX |

## Project Structure

The following is a proposed GitHub-friendly repository structure:

```text
fake-news-detection-bilstm/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── database.py
│
├── static/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── models/
│   └── bilstm.keras
│
├── data/
│   ├── history.db
│   ├── test_headlines.csv
│   ├── word_index.json
│   ├── model_metrics.json
│   └── test_predictions.csv
│
├── dataset/
│   └── IFND.csv
│
├── notebooks/
│   └── WordEmbedding_BiLSTM.ipynb
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── run.py
├── .gitignore
└── README.md
```

**Note:** This is a cleaned structure for publishing. The uploaded ZIP uses different directory and evaluation-file names. Rename the files and update references in `app/model.py` before adopting this structure.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection-bilstm.git
cd fake-news-detection-bilstm
```

### 2. Create a Virtual Environment

Python 3.10 or newer is required. Use a Python version supported by your installed TensorFlow release.

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

TensorFlow is a large dependency and installation may take several minutes.

### 4. Run the Application

```bash
python run.py
```

The application starts its FastAPI server and opens the dashboard automatically.

### 5. Access the Dashboard

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

A dedicated GPU is not required for inference.

## Deep Learning Methodology

The project implements a supervised deep learning pipeline.

```text
          News Headline
                |
                v
        Text Normalization
                |
                v
           Tokenization
                |
                v
        Integer Encoding
                |
                v
       Sequence Padding
          (40 Tokens)
                |
                v
         Word Embedding
                |
                v
       Bidirectional LSTM
                |
                v
       Binary Classification
                |
                v
      Fake-Class Probability
                |
                v
       Real / Fake Result
                |
                v
         Web Dashboard
                |
                v
      SQLite Prediction History
```

### 1. Text Preprocessing

Input headlines undergo text repair, lowercasing, and tokenization.

Each token is converted into a numerical index using a stored vocabulary.

### 2. Word Embedding

The embedding layer transforms numerical token indices into dense vectors.

The model uses:

- Vocabulary size: 20,000
- Embedding dimension: 100
- Maximum sequence length: 40 tokens

### 3. Bidirectional LSTM

The neural network processes input sequences in forward and backward directions to capture contextual relationships.

The saved model configuration uses 64 LSTM units.

### 4. Binary Classification

The model outputs a Fake-class score between 0 and 1.

- **Score ≥ 0.5:** Fake
- **Score below 0.5:** Real

If the input contains no recognized vocabulary tokens, the application returns no classification.

The score is a model output and should not be interpreted as independently verified factual certainty.

### 5. Word Effect Analysis

The application estimates word influence by removing individual words and comparing the resulting model scores.

Positive effects indicate that a word pushes the model toward the Fake category, while negative effects indicate influence toward Real.

These effects are approximations and do not establish causal or factual explanations.

## Model Performance

The trained BiLSTM model was evaluated on 7,185 test headlines.

| Evaluation Metric | Score |
|---|---|
| Accuracy | 94.99% |
| Precision | 90.38% |
| Recall | 94.77% |
| F1 Score | 92.52% |

### Robustness Evaluation

| Evaluation Subset | Accuracy | F1 Score |
|---|---|---|
| All test headlines | 94.99% | 92.52% |
| Excluding synthetic data | 95.12% | 87.22% |
| Without giveaway words | 93.97% | 69.16% |
| Neither synthetic nor giveaway words | 95.06% | 55.25% |

The lower F1 scores on certain subsets highlight the importance of evaluating model performance beyond overall accuracy.

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Classify a single headline |
| POST | `/api/predict/batch` | Classify up to 200 headlines |
| GET | `/api/samples/random` | Retrieve a test headline |
| GET | `/api/history` | Retrieve prediction history |
| DELETE | `/api/history` | Clear prediction history |
| GET | `/api/model` | Retrieve model information |
| GET | `/api/health` | Check server status |

## API Example

**Request:**

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "headline": "Government announces new education policy"
}
```

**Illustrative Response:**

```json
{
  "headline": "Government announces new education policy",
  "label": "real",
  "prob_fake": 0.12,
  "terms": [],
  "has_giveaway_words": false,
  "truncated": false
}
```

The example response is illustrative; actual predictions depend on the trained model.

## Testing

Run the automated API tests using:

```bash
python -m pytest
```

## Applications

- Fake news classification research
- Natural Language Processing
- Deep learning-based text classification
- Academic machine learning projects
- Educational misinformation analysis
- Neural network experimentation
- REST API development

## Limitations

- The model learns linguistic patterns from the IFND dataset rather than verifying factual claims.
- Predictions may be influenced by words frequently appearing in fact-checking articles.
- Headlines outside the training distribution may be classified incorrectly.
- The model processes only the first 40 tokens.
- Prediction scores are not guarantees of factual correctness.
- Model performance varies across evaluation subsets.

## Future Enhancements

- Transformer-based NLP architectures
- Multilingual news classification
- Integration with fact-checking services
- Explainable AI dashboards
- Real-time news monitoring
- Cloud deployment
- Advanced model comparison
- Improved generalization across news domains

## Dataset Reference

Sharma, D. K., & Garg, S. (2021). *IFND: A benchmark dataset for fake news detection*. Complex & Intelligent Systems.

The IFND dataset is intended for research use. Review its licensing and redistribution requirements before publishing the dataset.

## License

No software license is currently specified. Add an appropriate LICENSE file if you intend to grant redistribution and reuse permissions.

---

**Project Category:** Deep Learning | Natural Language Processing | BiLSTM | Fake News Detection | Web Development

**Developed for academic and research purposes.**
