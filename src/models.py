import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class EmbeddingLayer(nn.Module):
    """
    Provides word embeddings. Supports either a standard nn.Embedding
    (e.g., word2vec) or a transformer-based embedding (e.g., BERT).
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embedding_type="word2vec",
        pretrained_model_name="bert-base-uncased",
        freeze=True,
        pretrained_weights=None,
    ):
        super(EmbeddingLayer, self).__init__()
        self.embedding_type = embedding_type
        if embedding_type == "word2vec":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for word2vec embedding")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            if pretrained_weights is not None:
                self.embedding.weight.data.copy_(pretrained_weights)
        elif embedding_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
            self.embedding_dim = self.bert.config.hidden_size
            if freeze:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("embedding_type must be either 'word2vec' or 'bert'")

    def forward(self, input_ids, attention_mask=None):
        if self.embedding_type == "word2vec":
            return self.embedding(input_ids)
        elif self.embedding_type == "bert":
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state


class WordEncoder(nn.Module):
    """
    Encodes a sentence (sequence of words) using a bidirectional GRU and
    applies an attention mechanism to produce a sentence vector.
    """

    def __init__(self, embedding_dim, word_hidden_size, word_attention_size):
        super(WordEncoder, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.gru = nn.GRU(
            embedding_dim, word_hidden_size, bidirectional=True, batch_first=True
        )
        self.attention_fc = nn.Linear(2 * word_hidden_size, word_attention_size)
        self.context_vector = nn.Parameter(torch.randn(word_attention_size, 1))

    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: Tensor of shape (batch, max_words, embedding_dim)
            lengths: Tensor of shape (batch,) containing actual word counts per sentence.
        Returns:
            sentence_vector: Tensor of shape (batch, 2 * word_hidden_size)
            alpha: Word-level attention weights.
        """
        # Sort sequences by length (for pack_padded_sequence)
        sorted_lengths, perm_idx = torch.sort(lengths, descending=True)
        embeddings_sorted = embeddings[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings_sorted, sorted_lengths.cpu(), batch_first=True
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=embeddings.size(1)
        )
        # Unsort to restore original order
        _, unperm_idx = torch.sort(perm_idx)
        out = out[unperm_idx]  # (batch, max_words, 2 * word_hidden_size)

        # Attention mechanism
        uit = torch.tanh(
            self.attention_fc(out)
        )  # (batch, max_words, word_attention_size)
        sim = torch.matmul(uit, self.context_vector).squeeze(2)  # (batch, max_words)
        # Create mask to zero-out padded positions
        max_words = embeddings.size(1)
        mask = torch.arange(max_words, device=embeddings.device).expand(
            embeddings.size(0), max_words
        ) < lengths.unsqueeze(1)
        mask = mask.float()
        sim = sim * mask + (1 - mask) * (-1e20)
        alpha = F.softmax(sim, dim=1).unsqueeze(2)  # (batch, max_words, 1)
        sentence_vector = torch.sum(out * alpha, dim=1)  # (batch, 2 * word_hidden_size)
        return sentence_vector, alpha


class SentenceEncoder(nn.Module):
    """
    Encodes a document (sequence of sentence vectors) using a bidirectional GRU and
    applies an attention mechanism to produce a document vector.
    """

    def __init__(self, input_dim, sentence_hidden_size, sentence_attention_size):
        super(SentenceEncoder, self).__init__()
        self.gru = nn.GRU(
            input_dim, sentence_hidden_size, bidirectional=True, batch_first=True
        )
        self.attention_fc = nn.Linear(2 * sentence_hidden_size, sentence_attention_size)
        self.context_vector = nn.Parameter(torch.randn(sentence_attention_size, 1))

    def forward(self, sentence_vectors, doc_lengths):
        """
        Args:
            sentence_vectors: Tensor of shape (batch, max_sents, input_dim)
            doc_lengths: Tensor of shape (batch,) containing the number of valid sentences per document.
        Returns:
            doc_vector: Tensor of shape (batch, 2 * sentence_hidden_size)
            alpha: Sentence-level attention weights.
        """
        sorted_lengths, perm_idx = torch.sort(doc_lengths, descending=True)
        sentence_vectors_sorted = sentence_vectors[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(
            sentence_vectors_sorted, sorted_lengths.cpu(), batch_first=True
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=sentence_vectors.size(1)
        )
        _, unperm_idx = torch.sort(perm_idx)
        out = out[unperm_idx]  # (batch, max_sents, 2 * sentence_hidden_size)

        # Attention mechanism
        uit = torch.tanh(
            self.attention_fc(out)
        )  # (batch, max_sents, sentence_attention_size)
        sim = torch.matmul(uit, self.context_vector).squeeze(2)  # (batch, max_sents)
        max_sents = sentence_vectors.size(1)
        mask = torch.arange(max_sents, device=sentence_vectors.device).expand(
            sentence_vectors.size(0), max_sents
        ) < doc_lengths.unsqueeze(1)
        mask = mask.float()
        sim = sim * mask + (1 - mask) * (-1e20)
        alpha = F.softmax(sim, dim=1).unsqueeze(2)  # (batch, max_sents, 1)
        doc_vector = torch.sum(out * alpha, dim=1)  # (batch, 2 * sentence_hidden_size)
        return doc_vector, alpha


class HAN(nn.Module):
    """
    Hierarchical Attention Network for document classification.
    Expects input documents as a tensor of shape (batch, max_sents, max_words),
    where 0 is the padding token.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        word_hidden_size,
        sentence_hidden_size,
        word_attention_size,
        sentence_attention_size,
        num_classes,
        embedding_type="word2vec",
        pretrained_model_name="bert-base-uncased",
        freeze=True,
        pretrained_weights=None,
    ):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(
            vocab_size,
            embedding_dim,
            embedding_type,
            pretrained_model_name,
            freeze,
            pretrained_weights,
        )
        self.word_encoder = WordEncoder(
            embedding_dim, word_hidden_size, word_attention_size
        )
        # Input dimension to SentenceEncoder is 2 * word_hidden_size.
        self.sentence_encoder = SentenceEncoder(
            2 * word_hidden_size, sentence_hidden_size, sentence_attention_size
        )
        self.classifier = nn.Linear(2 * sentence_hidden_size, num_classes)

    def forward(self, documents):
        """
        Args:
            documents: Tensor of shape (batch, max_sents, max_words)
        Returns:
            logits: Tensor of shape (batch, num_classes)
            word_attn: Word-level attention weights (flattened per sentence).
            sent_attn: Sentence-level attention weights.
        """
        batch_size, max_sents, max_words = documents.size()
        # Compute word lengths (number of non-zero tokens per sentence)
        word_lengths = (documents != 0).sum(dim=2)  # shape: (batch, max_sents)
        # Compute document lengths (number of valid sentences)
        doc_lengths = (word_lengths != 0).sum(dim=1)  # shape: (batch)

        # Flatten documents to process all sentences with the word encoder.
        documents_flat = documents.view(-1, max_words)  # (batch*max_sents, max_words)
        word_lengths_flat = word_lengths.view(-1).clamp(min=1)  # avoid zeros

        # Embed words.
        embedded = self.embedding_layer(
            documents_flat
        )  # (batch*max_sents, max_words, embedding_dim)
        # Encode words into sentence vectors.
        sentence_vectors, word_attn = self.word_encoder(
            embedded, word_lengths_flat
        )  # (batch*max_sents, 2*word_hidden_size)
        # Reshape back to (batch, max_sents, 2*word_hidden_size)
        sentence_vectors = sentence_vectors.view(batch_size, max_sents, -1)
        # Encode sentences into a document vector.
        doc_vector, sent_attn = self.sentence_encoder(sentence_vectors, doc_lengths)
        logits = self.classifier(doc_vector)
        return logits, word_attn, sent_attn
