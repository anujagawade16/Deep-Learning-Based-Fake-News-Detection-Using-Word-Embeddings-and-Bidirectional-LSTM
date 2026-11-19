# IFND Headline Checker

A web app for testing Student 3's Word Embedding + BiLSTM fake news model.

## Run it
You need Python 3.10 or newer. In this folder, run:

```
pip install -r requirements.txt
python run.py
```

The app opens at http://127.0.0.1:8000. Press Ctrl+C in the terminal to stop it.

TensorFlow is a large download (over 500 MB), and the app can take several seconds to start. A GPU is not needed.

## How the word effects work
A BiLSTM has no simple per-word weights. To explain a result, the app removes each word in turn and
measures how the model's score changes. A positive value means the word pushes the headline towards
Fake. These are estimates of the model's behaviour, and words can also interact with each other.

The notebook did not save its vocabulary, so `data/word_index.json` was rebuilt from the same training
partition. Predictions using it matched the notebook's saved results on all 7,185 test headlines.

## What's inside
- `app/main.py`: FastAPI routes
- `app/model.py`: model loading, prediction and word effects
- `app/database.py`: SQLite check history, stored in `data/history.db`
- `static/`: web interface (HTML, CSS, JavaScript)
- `tests/`: API tests; run them with `python -m pytest`

## API
Interactive API documentation is at http://127.0.0.1:8000/docs.

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/predict` | Check one headline and save it to the history |
| POST | `/api/predict/batch` | Check up to 200 headlines (without word effects) |
| GET | `/api/samples/random` | Get an unseen test headline with its label |
| GET | `/api/history` | List saved checks |
| DELETE | `/api/history` | Delete saved checks |
| GET | `/api/model` | Get model details and test results |
| GET | `/api/health` | Confirm the server is running |

## Limits of the model
The model learned the wording of the IFND dataset. It does not check facts. Words common on
fact-checking sites, such as "fact check" and "viral", push a headline towards Fake, so its
results should not be used to decide whether a claim is true.

IFND is for research use only. Cite: Sharma, D. K., & Garg, S. (2021). IFND: a benchmark dataset
for fake news detection. Complex & Intelligent Systems.
