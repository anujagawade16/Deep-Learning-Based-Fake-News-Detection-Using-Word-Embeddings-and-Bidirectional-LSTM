"""Loads Student 3's trained BiLSTM and runs predictions on headlines."""
import json
import os
import re
import threading
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import keras
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / 'models' / 'bilstm.keras'
DATA_DIR = BASE / 'data'
MAX_LEN = 40
UNKNOWN = 1

APOSTROPHES = ['\x92', '\xd5', '\x89\xdb\xaa', '\xe2\x80\x99', '\xef\xbf\xbd']
GIVEAWAY = re.compile(
    r'\b(?:fact check|factcheck|fake|false|falsely|viral|claim|claims|claimed|misleading|hoax|'
    r'rumou?r|debunk|no this|not a|shared as|shared with|morphed|doctored|old video|old photo|'
    r'video|photo|image|picture)\b')


def repair_text(text):
    """Same text repair as the training notebook."""
    for broken in APOSTROPHES:
        text = text.replace(broken, "'")
    text = re.sub(r'[^\x00-\x7f]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text):
    """Same tokenization as the training notebook."""
    return re.findall(r"[a-z0-9']+", repair_text(text).lower())


def to_logit(prob):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return np.log(prob / (1 - prob))


class HeadlineClassifier:
    def __init__(self):
        self.model = keras.saving.load_model(MODEL_PATH)
        self.lock = threading.Lock()
        # The notebook did not save its vocabulary, so this file was rebuilt from the same
        # training partition and checked against the notebook's saved test predictions.
        self.word_index = json.loads((DATA_DIR / 'word_index.json').read_text(encoding='utf-8'))
        self.samples = pd.read_csv(DATA_DIR / 'test_headlines.csv')
        self.report = self._build_report()

    def encode(self, tokens):
        return [self.word_index.get(t, UNKNOWN) for t in tokens[:MAX_LEN]]

    def _probabilities(self, sequences, batch_size=256):
        padded = keras.utils.pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        outputs = []
        with self.lock:
            for start in range(0, len(padded), batch_size):
                batch = padded[start:start + batch_size]
                outputs.append(np.asarray(self.model(batch, training=False)).ravel())
        return np.concatenate(outputs).astype(float)

    def _empty_result(self, headline):
        key = re.sub(r'[^a-z0-9]+', ' ', repair_text(headline).lower()).strip()
        tokens = tokenize(headline)
        return {'headline': headline, 'label': None, 'prob_fake': None, 'terms': [],
                'has_giveaway_words': bool(GIVEAWAY.search(key)),
                'truncated': len(tokens) > MAX_LEN}

    def _is_known(self, sequence):
        return any(index != UNKNOWN for index in sequence)

    def _word_effects(self, tokens, sequence, prob_fake, top_terms):
        """Remove each word in turn and measure how the score changes."""
        without_word = [sequence[:i] + sequence[i + 1:] for i in range(len(sequence))]
        if len(sequence) == 1:
            return [{'term': tokens[0], 'effect': round(float(to_logit(prob_fake)), 4)}]
        changed = self._probabilities(without_word)
        base = to_logit(prob_fake)
        effects = [{'term': tokens[i], 'effect': round(float(base - to_logit(changed[i])), 4)}
                   for i in range(len(sequence))]
        effects.sort(key=lambda e: abs(e['effect']), reverse=True)
        return effects[:top_terms]

    def predict(self, headline, top_terms=8):
        result = self._empty_result(headline)
        tokens = tokenize(headline)
        sequence = self.encode(tokens)
        if not self._is_known(sequence):
            return result  # the model knows none of these words
        prob_fake = float(self._probabilities([sequence])[0])
        result.update(label='fake' if prob_fake >= 0.5 else 'real', prob_fake=round(prob_fake, 6),
                      terms=self._word_effects(tokens, sequence, prob_fake, top_terms))
        return result

    def predict_many(self, headlines):
        """Faster than predict for many headlines; word effects are skipped."""
        results = [self._empty_result(h) for h in headlines]
        sequences = [self.encode(tokenize(h)) for h in headlines]
        known = [i for i, s in enumerate(sequences) if self._is_known(s)]
        if known:
            probs = self._probabilities([sequences[i] for i in known])
            for i, prob_fake in zip(known, probs):
                results[i].update(label='fake' if prob_fake >= 0.5 else 'real',
                                  prob_fake=round(float(prob_fake), 6))
        return results

    def random_sample(self):
        row = self.samples.sample(1).iloc[0]
        return {'headline': row['headline'], 'label': 'fake' if row['label'] == 1 else 'real'}

    def _build_report(self):
        saved = json.loads((DATA_DIR / 'student_3_metrics.json').read_text(encoding='utf-8'))
        results = pd.read_csv(DATA_DIR / 'student_3_test_predictions.csv')
        actual, predicted = results['label'], results['prediction']
        return {
            'algorithm': saved['algorithm'],
            'test_metrics': saved['test_metrics'],
            'robustness': saved['robustness'],
            'details': saved['details'],
            'parameters': self.model.count_params(),
            'confusion_matrix': {
                'real_as_real': int(((actual == 0) & (predicted == 0)).sum()),
                'real_as_fake': int(((actual == 0) & (predicted == 1)).sum()),
                'fake_as_real': int(((actual == 1) & (predicted == 0)).sum()),
                'fake_as_fake': int(((actual == 1) & (predicted == 1)).sum()),
            },
        }
