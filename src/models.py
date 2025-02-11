import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class EmbeddingLayer(nn.Module):
    """
    Wrapper module to provide embeddings. Either provide pre-trained word2vec or BERT.
    """

    def __init__(
        self,
        vocab_size=None,
        embedding_dim=200,
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
        if self.embedding_type == "random":
            return self.embedding(input_ids)
        elif self.embedding_type == "bert":
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state


class WordAttentionLayer(nn.Module):
    """
    Word-level encoder with attention. Bidirectional GRU.
    Computes attention scores to build sentence vector.
    """

    def __init__(self, embedding_dim, word_hidden_dim, bidirectional=True):
        super(WordAttentionLayer, self).__init__()

        self.gru = nn.GRU(
            embedding_dim,
            word_hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )
        gru_out_dim = word_hidden_dim * 2 if bidirectional else word_hidden_dim
        self.attention_fc = nn.Linear(gru_out_dim, gru_out_dim)
        self.context_vector = nn.Parameter(torch.rand(gru_out_dim))

    def forward(self, x):
        gru_output, _ = self.gru(x)
        u_it = torch.tanh(self.attention_fc(gru_output))
        attn_scores = torch.matmul(u_it, self.context_vector)
        attn_weights = F.softmax(attn_scores, dim=1)
        sentence_vector = torch.sum(gru_output * attn_weights.unsqueeze(-1), dim=1)
        return sentence_vector, attn_weights


class SentenceAttentionLayer(nn.Module):
    """
    Sentence-level encoder with attention. Bidirectional GRU.
    Computes attention scores to build document vector.
    """

    def __init__(self, sent_input_dim, sent_hidden_dim, bidirectional=True):
        super(SentenceAttentionLayer, self).__init__()

        self.gru = nn.GRU(
            sent_input_dim,
            sent_hidden_dim,
            bidirectional=bidirectional,
            batch_first=True,
        )
        gru_out_dim = sent_hidden_dim * 2 if bidirectional else sent_hidden_dim
        self.attention_fc = nn.Linear(gru_out_dim, gru_out_dim)
        self.context_vector = nn.Parameter(torch.rand(gru_out_dim))

    def forward(self, x):
        gru_output, _ = self.gru(x)
        u_i = torch.tanh(self.attention_fc(gru_output))
        attn_scores = torch.matmul(u_i, self.context_vector)
        attn_weights = F.softmax(attn_scores, dim=1)
        document_vector = torch.sum(gru_output * attn_weights.unsqueeze(-1), dim=1)
        return document_vector, attn_weights


class HAN(nn.Module):
    """
    Hierarchical Attention Network.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        word_hidden_dim,
        sent_hidden_dim,
        num_classes,
        embedding_type="word2vec",
        pretrained_model_name="bert-base-uncased",
        freeze=True,
        pretrained_weights=None,
    ):
        super(HAN, self).__init__()

        self.embedding_layer = EmbeddingLayer(
            vocab_size,
            embedding_dim,
            embedding_type,
            pretrained_model_name,
            freeze,
            pretrained_weights,
        )
        self.word_attn = WordAttentionLayer(embedding_dim, word_hidden_dim)
        self.sent_attn = SentenceAttentionLayer(word_hidden_dim * 2, sent_hidden_dim)
        self.classifier = nn.Linear(sent_hidden_dim * 2, num_classes)

    def forward(self, documents, attention_mask=None):
        batch_size, num_sentences, max_sentence_length = documents.size()

        documents = documents.view(-1, max_sentence_length)
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, max_sentence_length)
            embeddings = self.embedding_layer(documents, attention_mask)
        else:
            embeddings = self.embedding_layer(documents)

        sentence_vectors, word_attn_weights = self.word_attn(embeddings)
        sentence_vectors = sentence_vectors.view(batch_size, num_sentences, -1)

        document_vector, sent_attn_weights = self.sent_attn(sentence_vectors)
        logits = self.classifier(document_vector)

        return logits, word_attn_weights, sent_attn_weights
