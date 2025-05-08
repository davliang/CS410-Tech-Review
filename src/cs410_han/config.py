__all__ = ["settings"]

from pathlib import Path
from typing import Literal
from pydantic import Field, FilePath, DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Paths
    data_path: FilePath = Field(
        default=FilePath("data/yelp-2015.json"),
        description="Path to the Yelp dataset",
    )
    model_save_dir: DirectoryPath = Field(
        default=DirectoryPath("models"),
        description="Directory to save the trained model",
    )
    cache_dir: DirectoryPath = Field(
        default=DirectoryPath("cache"),
        description="Directory to save cached files",
    )

    # Cache Settings
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached files for data processing",
    )

    @property
    def cache_train_path(self) -> Path:
        return self.cache_dir / "train_tokenized.joblib"

    @property
    def cache_val_path(self) -> Path:
        return self.cache_dir / "val_tokenized.joblib"

    @property
    def cache_test_path(self) -> Path:
        return self.cache_dir / "test_tokenized.joblib"

    @property
    def cache_w2v_model_path(self) -> Path:
        return self.cache_dir / "word2vec_model.gensim"

    # Model Hyperparameters
    embed_dim: int = Field(
        default=200,
        description="Dimensionality of the word embeddings",
    )
    word_hidden_dim: int = Field(
        default=50,
        description="Dimensionality of the word hidden layer",
    )
    sent_hidden_dim: int = Field(
        default=50,
        description="Dimensionality of the sentence hidden layer",
    )
    num_classes: int = Field(
        default=5,  # From the Yelp 2015 dataset
        description="Number of classes for classification",
    )

    # Training Hyperparameters
    learning_rate: float = Field(
        default=0.01,
        description="Learning rate for the optimizer",
    )
    momentum: float = Field(
        default=0.9,
        description="Momentum for the optimizer",
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for training",
    )
    num_epochs: int = Field(
        default=20,
        description="Number of epochs for training",
    )
    early_stopping_patience: int = Field(
        default=3,
        description="Number of epochs with no improvement after which training will be stopped",
    )

    # Preprocessing
    min_freq: int = Field(
        default=5,
        description="Minimum frequency for words to be included in the vocabulary",
    )

    # Other settings
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility",
    )
    log_level: Literal[
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(
        default="INFO",
        description="Logging level to pass to loguru",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
