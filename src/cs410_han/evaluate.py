from typing import Tuple

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cs410_han.logger import logger


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Evaluating",
) -> Tuple[float, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batches_processed = 0

    if len(data_loader) == 0:
        logger.warning(f"Cannot evaluate: {description} data loader is empty.")
        return 0.0, 0.0

    total_batches = len(data_loader)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        TextColumn("Acc: {task.fields[acc]:.2f}%"),
        expand=True,
    )
    task = progress.add_task(
        f"[cyan]{description}...", total=total_batches, loss=0.0, acc=0.0
    )

    with torch.no_grad():
        with progress:
            for batch_idx, batch in enumerate(data_loader):
                try:
                    docs, labels, sent_lengths, doc_lengths = [
                        b.to(device) for b in batch
                    ]

                    # validate inputs
                    if torch.any(doc_lengths <= 0) or docs.nelement() == 0:
                        progress.update(task, advance=1)
                        continue

                    # forward pass
                    outputs, _, _ = model(docs, sent_lengths, doc_lengths)
                    if outputs is None:
                        progress.update(task, advance=1)
                        continue

                    # compute loss
                    loss = criterion(outputs, labels)
                    if torch.isnan(loss):
                        logger.error(f"NaN loss in batch {batch_idx + 1}.")
                        progress.update(
                            task,
                            advance=1,
                            loss=float("nan"),
                            acc=(
                                100 * correct_predictions / total_samples
                                if total_samples
                                else 0.0
                            ),
                        )
                        continue

                    total_loss += loss.item()
                    _, predicted_labels = torch.max(outputs, dim=1)
                    correct_predictions += (predicted_labels == labels).sum().item()
                    total_samples += labels.size(0)
                    batches_processed += 1

                    # update progress
                    current_acc = (
                        100 * correct_predictions / total_samples
                        if total_samples
                        else 0.0
                    )
                    current_avg_loss = (
                        total_loss / batches_processed if batches_processed else 0.0
                    )
                    progress.update(
                        task, advance=1, loss=current_avg_loss, acc=current_acc
                    )

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx + 1}: {e}")
                    progress.update(task, advance=1, loss=float("inf"))
                    continue

    avg_loss = total_loss / batches_processed if batches_processed else 0.0
    accuracy = 100 * correct_predictions / total_samples if total_samples else 0.0

    if batches_processed < total_batches:
        logger.warning(
            f"Processed {batches_processed}/{total_batches} batches during evaluation."
        )

    return avg_loss, accuracy
