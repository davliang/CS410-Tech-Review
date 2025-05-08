import random
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import joblib
from rich.progress import track

from cs410_han.config import settings
from cs410_han.console import console
from cs410_han.data import (
    build_vocab,
    create_data_loader,
    create_embedding_matrix,
    get_max_lengths,
    load_data,
    nlp,
    PAD_TOKEN,
    split_data,
    train_word_embeddings,
    TokenizedData,
    YelpReviewDataset,
)
from cs410_han.evaluate import evaluate_model
from cs410_han.logger import logger
from cs410_han.model import HAN
from cs410_han.train import train_model


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Seed set to {seed}")


def process_texts_with_pipe(
    raw_data: List[Tuple[str, int]], description: str
) -> TokenizedData:
    """Process raw text using nlp.pipe."""
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    texts, labels = zip(*raw_data) if raw_data else ([], [])
    disable_pipes = ["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]
    pipe_kwargs = {"batch_size": 100, "n_process": -1}
    tokenized_data: TokenizedData = []

    logger.info(f"Processing {len(texts)} texts: {description}")

    with nlp.select_pipes(disable=disable_pipes):
        pipeline = zip(nlp.pipe(texts, **pipe_kwargs), labels)
        for i, (doc, label) in track(
            enumerate(pipeline),
            console=console,
            total=len(texts),
            description=description,
        ):
            try:
                sentences: List[List[str]] = []
                for sent in doc.sents:
                    toks = [tok.text.lower() for tok in sent if tok.is_alpha]
                    if toks:
                        sentences.append(toks)
                tokenized_data.append((sentences, label))
            except Exception as e:
                logger.error(f"Error at index {i} in {description}: {e}")
                tokenized_data.append(([], label))

    return tokenized_data


def main() -> None:
    """Run the HAN experiment."""
    set_seed(settings.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")
    logger.info(
        f"Config: {settings.model_dump(exclude={'data_path', 'model_save_dir', 'cache_dir'})}"
    )
    logger.info(f"Data path: {settings.data_path}")
    logger.info(f"Model dir: {settings.model_save_dir}")
    logger.info(f"Cache dir: {settings.cache_dir}")
    logger.info(f"Use cache: {settings.use_cache}")

    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    all_data_raw = load_data(settings.data_path)
    if not all_data_raw:
        logger.critical("No data loaded")
        sys.exit(1)

    train_raw, val_raw, test_raw = split_data(all_data_raw, settings.seed)
    if not (train_raw and val_raw and test_raw):
        logger.critical("Empty data split")
        sys.exit(1)

    # Tokenization
    cache_exists = (
        settings.cache_train_path.is_file()
        and settings.cache_val_path.is_file()
        and settings.cache_test_path.is_file()
    )
    if settings.use_cache and cache_exists:
        with console.status("[bold green]Loading tokenized data from cache…[/]"):
            try:
                train_tok = joblib.load(settings.cache_train_path)
                val_tok = joblib.load(settings.cache_val_path)
                test_tok = joblib.load(settings.cache_test_path)

                logger.info("Loaded tokenized data from cache")
            except Exception as e:
                logger.error(f"Cache load failed: {e}")
                cache_exists = False

    if not settings.use_cache or not cache_exists:
        console.print("[bold green]Tokenizing text data…[/]")
        train_tok = process_texts_with_pipe(train_raw, "Train")
        val_tok = process_texts_with_pipe(val_raw, "Val")
        test_tok = process_texts_with_pipe(test_raw, "Test")

        if settings.use_cache:
            try:
                joblib.dump(train_tok, settings.cache_train_path)
                joblib.dump(val_tok, settings.cache_val_path)
                joblib.dump(test_tok, settings.cache_test_path)

                logger.info("Cached tokenized data")
            except Exception as e:
                logger.error(f"Cache save failed: {e}")

    if not (train_tok and val_tok and test_tok):
        logger.critical("Tokenized data empty")
        sys.exit(1)

    logger.info(f"{len(train_tok)} training examples")

    # Vocabulary
    word_to_idx, _ = build_vocab(train_tok, min_freq=settings.min_freq)
    pad_idx = word_to_idx[PAD_TOKEN]
    logger.info(f"Vocab size: {len(word_to_idx):,}")

    # Word2Vec
    w2v_cache = settings.cache_w2v_model_path.is_file()
    if settings.use_cache and w2v_cache:
        with console.status("[bold green]Loading Word2Vec model from cache…[/]"):
            try:
                w2v = Word2Vec.load(str(settings.cache_w2v_model_path))
                logger.info("Loaded Word2Vec model")
            except Exception:
                w2v_cache = False

    if not settings.use_cache or not w2v_cache:
        with console.status("[bold green]Training Word2Vec embeddings…[/]"):
            w2v = train_word_embeddings(
                train_tok,
                val_tok,
                embed_dim=settings.embed_dim,
                min_freq=settings.min_freq,
            )
        if settings.use_cache:
            try:
                w2v.save(str(settings.cache_w2v_model_path))
                logger.info("Cached Word2Vec model")
            except Exception as e:
                logger.error(f"Cache save failed: {e}")

    if w2v is None:
        logger.critical("Word2Vec unavailable")
        sys.exit(1)

    embedding_matrix = create_embedding_matrix(w2v, word_to_idx, settings.embed_dim)

    # Data loaders
    max_sent, max_doc = get_max_lengths(train_tok, val_tok, test_tok)

    train_ds = YelpReviewDataset(train_tok, word_to_idx, max_sent, max_doc)
    val_ds = YelpReviewDataset(val_tok, word_to_idx, max_sent, max_doc)
    test_ds = YelpReviewDataset(test_tok, word_to_idx, max_sent, max_doc)

    train_loader = create_data_loader(train_ds, settings.batch_size, shuffle=True)
    val_loader = create_data_loader(val_ds, settings.batch_size, shuffle=False)
    test_loader = create_data_loader(test_ds, settings.batch_size, shuffle=False)

    logger.info(f"Batch size: {settings.batch_size}")

    # Model initialization
    model = HAN(
        vocab_size=len(word_to_idx),
        embed_dim=settings.embed_dim,
        word_hidden_dim=settings.word_hidden_dim,
        sent_hidden_dim=settings.sent_hidden_dim,
        num_classes=settings.num_classes,
        embedding_matrix=embedding_matrix,
        pad_idx=pad_idx,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Parameters: {params}")

    # Training
    best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=settings.learning_rate,
        momentum=settings.momentum,
        num_epochs=settings.num_epochs,
        patience=settings.early_stopping_patience,
        model_save_dir=settings.model_save_dir,
        device=device,
    )

    # Evaluation
    if best_state:
        model.load_state_dict(best_state)
    else:
        logger.warning("No best model state found")

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        description="Final Test Set",
    )

    logger.info(f"Test loss: {loss:.4f}")
    logger.info(f"Test accuracy: {acc:.2f}%")
    logger.info("Experiment complete")


if __name__ == "__main__":
    main()
