from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Apply attention over GRU outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_transform = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, gru_output: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return weighted sum and attention weights."""
        uit = torch.tanh(self.attention_transform(gru_output))
        scores = self.context_vector(uit).squeeze(2)

        max_len = scores.size(1)
        mask = (
            torch.arange(max_len, device=scores.device)
            .unsqueeze(0)
            .ge(lengths.to(scores.device).unsqueeze(1))
        )
        scores.masked_fill_(mask, -float("inf"))

        alpha = F.softmax(scores, dim=1)
        weighted = (gru_output * alpha.unsqueeze(2)).sum(dim=1)
        return weighted, alpha


class WordEncoder(nn.Module):
    """Encode words in sentences."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        embedding_matrix: torch.Tensor,
        pad_idx: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=pad_idx
        )
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)

    def forward(
        self, sentences: torch.Tensor, sent_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sentence vectors and word attention weights."""
        lengths = sent_lengths.cpu().clamp(min=1)
        embedded = self.embedding(sentences)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        return self.attention(gru_output, lengths)


class SentenceEncoder(nn.Module):
    """Encode sentences in documents."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)

    def forward(
        self, sentence_vectors: torch.Tensor, doc_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return document vectors and sentence attention weights."""
        lengths = doc_lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sentence_vectors, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        return self.attention(gru_output, lengths)


class HAN(nn.Module):
    """Hierarchical Attention Network for classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        word_hidden_dim: int,
        sent_hidden_dim: int,
        num_classes: int,
        embedding_matrix: torch.Tensor,
        pad_idx: int,
    ):
        super().__init__()
        self.word_encoder = WordEncoder(
            vocab_size, embed_dim, word_hidden_dim, embedding_matrix, pad_idx
        )
        self.sentence_encoder = SentenceEncoder(word_hidden_dim * 2, sent_hidden_dim)
        self.fc = nn.Linear(sent_hidden_dim * 2, num_classes)
        self.word_hidden_dim = word_hidden_dim

    def forward(
        self,
        docs: torch.Tensor,
        sent_lengths: torch.Tensor,
        doc_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return logits and placeholder for attention weights."""
        batch_size, max_doc, max_sent = docs.size()
        flat_docs = docs.view(-1, max_sent)
        flat_lens = sent_lengths.view(-1)

        mask = flat_lens > 0
        filtered_docs = flat_docs[mask]
        filtered_lens = flat_lens[mask]

        if filtered_docs.size(0) == 0:
            zeros = torch.zeros(batch_size, self.fc.out_features, device=docs.device)
            return zeros, None, None

        sent_vecs, _ = self.word_encoder(filtered_docs, filtered_lens)
        all_vecs = torch.zeros(
            batch_size * max_doc, sent_vecs.size(1), device=docs.device
        )
        all_vecs[mask] = sent_vecs
        sentence_vectors = all_vecs.view(batch_size, max_doc, -1)

        doc_vecs, _ = self.sentence_encoder(sentence_vectors, doc_lengths)
        output = self.fc(doc_vecs)
        return output, None, None
