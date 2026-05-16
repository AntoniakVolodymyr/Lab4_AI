import os
import re
import tarfile
import random
import requests
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = "aclImdb"

VOCAB_SIZE = 12000
MAX_LEN = 250
OOV_TOKEN = "<OOV>"
PAD_TOKEN = "<PAD>"

LIMIT_PER_CLASS = None


def setup_nltk():
    resources = ["punkt", "punkt_tab", "stopwords"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"Error downloading {res}: {e}")


def download_dataset():
    if os.path.exists(DATA_DIR):
        print("Dataset already exists.")
        return

    print("Downloading dataset...")
    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    archive_name = "imdb.tar.gz"
    with open(archive_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print("Extracting dataset...")
    with tarfile.open(archive_name, "r:gz") as tar:
        tar.extractall()

    print("Dataset downloaded and extracted.")

def load_imdb_data(base_path, subset="train", limit_per_class=None):
    texts, labels = [], []

    for label_type in ["pos", "neg"]:
        dir_name = os.path.join(base_path, subset, label_type)
        label = 1 if label_type == "pos" else 0

        filenames = os.listdir(dir_name)
        random.shuffle(filenames)

        if limit_per_class is not None:
            filenames = filenames[:limit_per_class]

        for fname in filenames:
            with open(os.path.join(dir_name, fname), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), np.array(labels, dtype=np.float32)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯіІїЇєЄґҐ' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_texts(texts):
    return [word_tokenize(clean_text(text)) for text in texts]


def build_vocab(tokenized_texts, vocab_size):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    word_index = {PAD_TOKEN: 0, OOV_TOKEN: 1}

    most_common = counter.most_common(vocab_size - 2)
    for i, (word, _) in enumerate(most_common, start=2):
        word_index[word] = i

    return word_index


def texts_to_sequences(tokenized_texts, word_index):
    oov_id = word_index[OOV_TOKEN]
    return [[word_index.get(token, oov_id) for token in tokens] for tokens in tokenized_texts]


def prepare_data():
    setup_nltk()
    download_dataset()

    print("Loading text data...")
    train_texts, train_labels = load_imdb_data(DATA_DIR, "train", LIMIT_PER_CLASS)
    test_texts, test_labels = load_imdb_data(DATA_DIR, "test", LIMIT_PER_CLASS)

    print("Tokenizing...")
    train_tokens = tokenize_texts(train_texts)
    test_tokens = tokenize_texts(test_texts)

    print("Building vocabulary...")
    word_index = build_vocab(train_tokens, VOCAB_SIZE)
    print(f"Vocabulary size: {len(word_index)}")

    train_seq = texts_to_sequences(train_tokens, word_index)
    test_seq = texts_to_sequences(test_tokens, word_index)

    x_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    return x_train, train_labels, x_test, test_labels, word_index


def build_model(config):
    model = models.Sequential()
    model.add(layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=config["embedding_dim"],
        input_length=MAX_LEN
    ))

    for layer_cfg in config["layers"]:
        if layer_cfg["type"] == "lstm":
            model.add(layers.LSTM(
                layer_cfg["units"],
                return_sequences=layer_cfg.get("return_sequences", False),
                dropout=layer_cfg.get("dropout", 0.0),
                recurrent_dropout=layer_cfg.get("recurrent_dropout", 0.0)
            ))
        elif layer_cfg["type"] == "dropout":
            model.add(layers.Dropout(layer_cfg["rate"]))
        elif layer_cfg["type"] == "dense":
            model.add(layers.Dense(
                layer_cfg["units"],
                activation=layer_cfg.get("activation", "relu")
            ))

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer_name = config["optimizer"]
    lr = config["learning_rate"]

    if optimizer_name == "adam":
        optimizer = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        optimizer = optimizers.RMSprop(learning_rate=lr)
    elif optimizer_name == "nadam":
        optimizer = optimizers.Nadam(learning_rate=lr)
    else:
        raise ValueError("Unknown optimizer")

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


EXPERIMENTS = [
    {
        "name": "Exp 1",
        "epochs": 15,
        "embedding_dim": 128,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "layers": [
            {"type": "lstm", "units": 64, "dropout": 0.3, "recurrent_dropout": 0.2},
            {"type": "dropout", "rate": 0.3}
        ]
    },
    {
        "name": "Exp 2",
        "epochs": 20,
        "embedding_dim": 128,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "layers": [
            {"type": "lstm", "units": 128, "dropout": 0.3, "recurrent_dropout": 0.2},
            {"type": "dropout", "rate": 0.4}
        ]
    },
    {
        "name": "Exp 3",
        "epochs": 25,
        "embedding_dim": 128,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "layers": [
            {"type": "lstm", "units": 64, "dropout": 0.3, "recurrent_dropout": 0.3},
            {"type": "dropout", "rate": 0.4}
        ]
    },
    {
        "name": "Exp 4",
        "epochs": 30,
        "embedding_dim": 128,
        "optimizer": "rmsprop",
        "learning_rate": 0.001,
        "layers": [
            {"type": "lstm", "units": 128, "return_sequences": True, "dropout": 0.3, "recurrent_dropout": 0.2},
            {"type": "lstm", "units": 64, "dropout": 0.3, "recurrent_dropout": 0.2},
            {"type": "dropout", "rate": 0.4}
        ]
    },
    {
        "name": "Exp 5",
        "epochs": 35,
        "embedding_dim": 160,
        "optimizer": "nadam",
        "learning_rate": 0.001,
        "layers": [
            {"type": "lstm", "units": 96, "dropout": 0.35, "recurrent_dropout": 0.2},
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "dropout", "rate": 0.4}
        ]
    }
]


def plot_history(history, exp_name):
    safe_name = exp_name.replace(":", "").replace(" ", "_")

    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="test/val loss")
    plt.title(f"Loss — {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{safe_name}_loss.png", dpi=150)
    plt.close()

    # Графік accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="test/val accuracy")
    plt.title(f"Accuracy — {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{safe_name}_accuracy.png", dpi=150)
    plt.close()


def predict_reviews(model, word_index, reviews):
    tokens = tokenize_texts(reviews)
    seq = texts_to_sequences(tokens, word_index)
    x = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(x).flatten()

    for text, prob in zip(reviews, probs):
        label = "positive" if prob >= 0.5 else "negative"
        print(f"\nReview: {text}")
        print(f"Model output: {prob:.4f}")
        print(f"Predicted class: {label}")


def main():
    x_train, y_train, x_test, y_test, word_index = prepare_data()

    results = []
    best_model = None
    best_history = None
    best_config = None
    best_acc = -1

    for i, config in enumerate(EXPERIMENTS, start=1):
        print("\n" + "=" * 70)
        print(f"Training {config['name']}")
        print("=" * 70)

        model = build_model(config)
        model.summary()

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=config["epochs"],
            batch_size=64,
            validation_data=(x_test, y_test),
            callbacks=[reduce_lr, early_stop],
            verbose=1
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results.append({
            "№": i,
            "experiment": config["name"],
            "epochs": len(history.history["loss"]),
            "optimizer": config["optimizer"],
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "config": config
        })

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_history = history
            best_config = config

    print("\n\nRESULTS TABLE")
    print("-" * 120)
    for r in results:
        print(
            f"{r['№']}. {r['experiment']} | epochs={r['epochs']} | "
            f"optimizer={r['optimizer']} | loss={r['test_loss']:.4f} | "
            f"accuracy={r['test_accuracy']:.4f}"
        )

    print("\nBest experiment:", best_config["name"])
    plot_history(best_history, best_config["name"])

    best_model.save("best_lstm_imdb_model.keras")
    print("Best model saved as best_lstm_imdb_model.keras")

    custom_reviews = [
        "This movie was surprisingly emotional and beautifully acted.",
        "The film was boring, predictable and painfully slow.",
        "A wonderful film with strong characters and a satisfying ending."
    ]

    required_reviews = [
        "I expected to hate it, but it was actually the best movie of the year.",
        "The plot was as deep as a puddle.",
        "It was not a bad film, but certainly not a great one either."
    ]

    print("\nCUSTOM REVIEWS")
    predict_reviews(best_model, word_index, custom_reviews)

    print("\nREQUIRED REVIEWS")
    predict_reviews(best_model, word_index, required_reviews)

if __name__ == "__main__":
    main()
