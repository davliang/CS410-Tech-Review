import torch
from torch.utils.data import Dataset
import spacy
from transformers import AutoTokenizer
from tqdm import tqdm


class HANDataset(Dataset):
    """
    Dataset for HAN.
    Documents are tokenized into sentences and words using spacy.
    """

    def __init__(
        self,
        documents,
        labels,
        max_sentences=10,
        max_sentence_length=20,
        embedding_type="word2vec",
        pretrained_tokenizer_name="bert-base-uncased",
        vocab=None,
        batch_size=None,
        n_process=4,
    ):
        self.documents = documents
        self.labels = labels
        self.embedding_type = embedding_type

        # For defining the shape of the input tensor
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length

        # Disabling increases speed
        print("Loading spacy model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.nlp.add_pipe("sentencizer")

        # Pretokenize documents in parallel
        print("Tokenizing documents...")
        # Note: converting to list to allow indexing later
        self.tokenized_docs = list(
            tqdm(
                self.nlp.pipe(documents, n_process=n_process, batch_size=batch_size),
                total=len(documents),
                desc="Tokenizing",
            )
        )

        # Build vocab for embedding layer
        print("Building vocab...")
        if embedding_type == "word2vec":
            if vocab is None:
                self.vocab = self.build_vocab(self.tokenized_docs)
            else:
                self.vocab = vocab
        elif embedding_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
        else:
            raise ValueError("embedding_type must be either 'word2vec' or 'bert'")

    def build_vocab(self, tokenized_docs, min_freq=5):
        # Using the same minimum frequency as from the original paper
        from collections import Counter

        counter = Counter()
        for doc in tokenized_docs:
            for sent in doc.sents:
                tokens = [token.text for token in sent]
                counter.update(tokens)

        # Reserve 0 for padding and 1 for unknown tokens
        vocab = {"<PADDING>": 0, "<UNKNOWN>": 1}
        for token, freq in counter.items():
            if freq >= min_freq:
                vocab[token] = len(vocab)
        return vocab

    def encode_document(self, doc):
        sentences = list(doc.sents)
        sentences = sentences[: self.max_sentences]
        encoded = []
        for sent in sentences:
            if self.embedding_type == "word2vec":
                tokens = [token.text for token in sent]
                tokens = tokens[: self.max_sentence_length]
                token_ids = [
                    self.vocab.get(token, self.vocab["<UNKNOWN>"]) for token in tokens
                ]
            elif self.embedding_type == "bert":
                sent_text = sent.text
                encoded_sent = self.tokenizer.encode(
                    sent_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_sentence_length,
                )
                token_ids = encoded_sent

            if len(token_ids) < self.max_sentence_length:
                token_ids += [0] * (self.max_sentence_length - len(token_ids))

            encoded.append(token_ids)

        if len(encoded) < self.max_sentences:
            pad_sentence = [0] * self.max_sentence_length
            for _ in range(self.max_sentences - len(encoded)):
                encoded.append(pad_sentence)

        return encoded

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.tokenized_docs[idx]
        label = self.labels[idx]

        encoded_doc = self.encode_document(doc)
        encoded_doc = torch.tensor(encoded_doc, dtype=torch.long)

        # shape: (max_sentences, max_sentence_length)
        return encoded_doc, label
