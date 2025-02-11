import pandas as pd
import csv
import json
from tqdm import tqdm
import pathlib
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import nltk
from nltk.tokenize import sent_tokenize
import spacy

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

nlp = spacy.load(
    "en_core_web_sm",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
)
nlp.enable_pipe("senter")

# Load Hugging Face's fast tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
tokenizer.pre_tokenizer = Whitespace()  # Ensures word tokenization

DATASET_PATH = "data/yelp-2015.json"


def tokenize_texts(texts):
    tokenized_results = []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=10000, n_process=8),
        total=len(texts),
        desc="Tokenizing (SpaCy Word Splitter with nlp.pipe)",
        unit="text",
    ):
        text_tokenized_sentences = []
        for sent in doc.sents:
            sentence_tokens = [token.text for token in sent]
            text_tokenized_sentences.append(sentence_tokens)
        tokenized_results.append(text_tokenized_sentences)

    return tokenized_results


def main():
    stub_filename = pathlib.Path(DATASET_PATH)
    # Check if parquet file exists
    if stub_filename.with_suffix(".parquet").exists():
        print("Loading data from Parquet...")
        df = pd.read_parquet(stub_filename.with_suffix(".parquet"))
    else:
        print("Loading data from JSON...")
        df = pd.read_json(DATASET_PATH, lines=True)
        print("Saving data to Parquet...")
        # Save to parquet for faster loading next time
        df.to_parquet(stub_filename.with_suffix(".parquet"))

    print("Cleaning data...")
    # Drop unnecessary columns
    df = df[["stars", "text"]]
    # Standardize all whitespace to single spaces with " ".join
    df.loc[:, "text"] = df["text"].apply(lambda x: " ".join(x.split()))

    print("Tokenizing text...")
    df["tokenized_text"] = tokenize_texts(df["text"].to_list())

    # print("Converting to StringArray...")
    # new_tokenized_text = []
    # for x in tqdm(df["tokenized_text"], desc="Applying String Conversion", unit="row"):
    #     new_tokenized_text.append(pd.Series(x).astype("string"))
    # df["tokenized_text"] = new_tokenized_text

    print("Saving data to CSV...")
    csv_filename = stub_filename.with_suffix("-tokenized.csv")
    # index flag to False to avoid writing row indices
    df.to_csv(csv_filename, header=False, index=False, quoting=csv.QUOTE_ALL)

    print("Saving data to Parquet...")
    df.to_parquet(stub_filename.with_suffix("-tokenized.parquet"))


if __name__ == "__main__":
    main()
