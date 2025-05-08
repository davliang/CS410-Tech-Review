from collections import Counter
from datetime import date
from pathlib import Path
import math
import random
import sys
from typing import Dict, List, Optional, Tuple

from gensim.models import Word2Vec
import numpy as np
from pydantic import BaseModel, Field, ValidationError, model_validator
from rich.progress import track
import spacy
import spacy.cli
import torch
from torch.utils.data import DataLoader, Dataset

from cs410_han.console import console
from cs410_han.logger import logger

SPACY_MODEL = "en_core_web_sm"
try:
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "tagger", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", before="parser")
    logger.info(f"Loaded spaCy model {SPACY_MODEL}.")
except OSError:
    logger.warning(f"Model {SPACY_MODEL} not found. Downloading...")
    try:
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
        logger.success(f"Downloaded and loaded spaCy model {SPACY_MODEL}.")
    except Exception as e:
        logger.critical(f"Failed to download model: {e}")
        sys.exit(1)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Votes(BaseModel):
    """Counts votes."""

    funny: int
    useful: int
    cool: int


class YelpReviewEntry(BaseModel):
    """Yelp review model."""

    votes: Votes
    user_id: str
    review_id: str
    stars: int = Field(..., ge=1, le=5)
    date: date
    text: str
    type: str
    business_id: str
    label: Optional[int] = None

    @model_validator(mode="after")
    def calculate_label(self) -> "YelpReviewEntry":
        # adjust label from stars
        self.label = self.stars - 1
        return self


def load_data(file_path: Path) -> List[Tuple[str, int]]:
    """Load reviews from JSON file."""
    data: List[Tuple[str, int]] = []
    logger.info(f"Loading data from {file_path}")
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        return data

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = YelpReviewEntry.model_validate_json(line)
                    if record.label is not None:
                        data.append((record.text, record.label))
                    else:
                        logger.warning(f"Missing label at line {line_num}")
                except ValidationError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error at line {line_num}: {e}")
        logger.info(f"Loaded {len(data):,} reviews")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")

    return data


def split_data(
    all_data: List[Tuple[str, int]], seed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split data into train, validation, and test sets."""
    if not all_data:
        logger.error("Empty dataset")
        return [], [], []

    random.seed(seed)
    shuffled = random.sample(all_data, len(all_data))
    n = len(shuffled)
    n_train = math.ceil(n * 0.8)
    n_val = math.ceil(n * 0.1)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    if len(train) + len(val) + len(test) != n:
        logger.warning("Split sizes mismatch")

    return train, val, test


def preprocess_text(text: str) -> List[List[str]]:
    """Tokenize text into sentences and words."""
    if not text or text.isspace():
        return []
    try:
        doc = nlp(text)
        sentences: List[List[str]] = []
        for sent in doc.sents:
            tokens = [tok.text.lower() for tok in sent if tok.is_alpha]
            if tokens:
                sentences.append(tokens)
        return sentences
    except Exception as e:
        logger.error(f"spaCy error: {e}")
        return []


TokenizedData = List[Tuple[List[List[str]], int]]


def build_vocab(
    tokenized_docs_train: TokenizedData,
    min_freq: int,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from tokenized documents."""
    if not tokenized_docs_train:
        logger.warning("Empty training data")
        vocab = {PAD_TOKEN, UNK_TOKEN}
        sorted_vocab = sorted(vocab)
        word_to_idx = {w: i for i, w in enumerate(sorted_vocab)}
        idx_to_word = {i: w for i, w in enumerate(sorted_vocab)}
        return word_to_idx, idx_to_word

    counts: Counter = Counter()
    for doc, _ in track(tokenized_docs_train, console=console, description="Counting"):
        for sent in doc:
            counts.update(sent)

    vocab = {w for w, c in counts.items() if c >= min_freq}
    vocab.update({PAD_TOKEN, UNK_TOKEN})
    sorted_vocab = sorted(vocab)
    word_to_idx = {w: i for i, w in enumerate(sorted_vocab)}
    idx_to_word = {i: w for i, w in enumerate(sorted_vocab)}

    # logger.info(f"Vocab size: {len(vocab):,}")
    return word_to_idx, idx_to_word


def train_word_embeddings(
    tokenized_docs_train: TokenizedData,
    tokenized_docs_val: TokenizedData,
    embed_dim: int,
    min_freq: int,
    workers: int = 4,
) -> Word2Vec:
    """Train Word2Vec embeddings."""
    if not tokenized_docs_train and not tokenized_docs_val:
        logger.error("No data for Word2Vec")
        return Word2Vec(vector_size=embed_dim, min_count=min_freq)

    sentences: List[List[str]] = []
    for doc, _ in tokenized_docs_train:
        sentences.extend(doc)
    for doc, _ in tokenized_docs_val:
        sentences.extend(doc)

    if not sentences:
        logger.error("No sentences for Word2Vec")
        return Word2Vec(vector_size=embed_dim, min_count=min_freq)

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=embed_dim,
        window=5,
        min_count=min_freq,
        workers=workers,
        sg=1,
        hs=0,
        negative=5,
        epochs=5,
    )
    logger.success(f"Trained Word2Vec vocab size: {len(w2v.wv.key_to_index):,}")
    return w2v


def create_embedding_matrix(
    w2v_model: Word2Vec, word_to_idx: Dict[str, int], embed_dim: int
) -> torch.Tensor:
    """Create embedding matrix from Word2Vec model."""
    vocab_size = len(word_to_idx)
    matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim)).astype(
        np.float32
    )
    count = 0
    pad_idx = word_to_idx.get(PAD_TOKEN)

    for word, idx in word_to_idx.items():
        if word in w2v_model.wv:
            matrix[idx] = w2v_model.wv[word]
            count += 1

    if pad_idx is not None:
        matrix[pad_idx] = np.zeros(embed_dim, dtype=np.float32)

    logger.info(f"Initialized {count:,} of {vocab_size:,} embeddings")
    return torch.tensor(matrix)


class YelpReviewDataset(Dataset):
    """Dataset for Yelp reviews."""

    def __init__(
        self,
        tokenized_data: TokenizedData,
        word_to_idx: Dict[str, int],
        max_sent_length: int,
        max_doc_length: int,
    ):
        self.data = tokenized_data
        self.word_to_idx = word_to_idx
        self.max_sent_length = max(1, max_sent_length)
        self.max_doc_length = max(1, max_doc_length)
        self.pad_idx = word_to_idx[PAD_TOKEN]
        self.unk_idx = word_to_idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        doc, label = self.data[index]
        doc_idx = torch.full(
            (self.max_doc_length, self.max_sent_length),
            self.pad_idx,
            dtype=torch.long,
        )
        sent_lens = torch.zeros(self.max_doc_length, dtype=torch.long)
        count = 0
        for sent in doc[: self.max_doc_length]:
            length = min(len(sent), self.max_sent_length)
            if length == 0:
                continue
            sent_lens[count] = length
            for j in range(length):
                doc_idx[count, j] = self.word_to_idx.get(sent[j], self.unk_idx)
            count += 1
        return (
            doc_idx,
            torch.tensor(label, dtype=torch.long),
            sent_lens,
            torch.tensor(count, dtype=torch.long),
        )


def create_data_loader(
    dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0
) -> DataLoader:
    """Create DataLoader for dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_max_lengths(
    *tokenized_splits: TokenizedData,
) -> Tuple[int, int]:
    """Get max sentence and document lengths."""
    max_sent = 0
    max_doc = 0
    for split in tokenized_splits:
        for doc, _ in split:
            max_doc = max(max_doc, len(doc))
            for sent in doc:
                max_sent = max(max_sent, len(sent))
    max_sent = max(1, max_sent)
    max_doc = max(1, max_doc)
    logger.info(f"Max sent: {max_sent}, Max doc: {max_doc}")
    return max_sent, max_doc
