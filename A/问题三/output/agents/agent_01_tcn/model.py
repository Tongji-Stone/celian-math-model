from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return values
        return values[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(values) + self.residual(values))


class SmallTCNForecaster(nn.Module):
    """Causal TCN encoder with optional static encoder and direct decoder."""

    def __init__(
        self,
        dynamic_channels: int,
        static_features: int,
        horizon: int = 50,
        hidden_channels: int = 12,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        static_hidden: int = 8,
        decoder_hidden: int = 32,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = dynamic_channels
        for dilation in dilations:
            blocks.append(
                TemporalBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
            channels = hidden_channels
        self.temporal = nn.Sequential(*blocks)
        self.static_features = static_features
        if static_features:
            self.static_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(static_features, static_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            decoder_input = 2 * hidden_channels + static_hidden
        else:
            self.static_encoder = None
            decoder_input = 2 * hidden_channels
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, horizon),
        )

    def forward(
        self,
        dynamic_values: torch.Tensor,
        static_values: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # Input is [batch, time, channels]; Conv1d expects [batch, channels, time].
        encoded = self.temporal(dynamic_values.transpose(1, 2))
        sequence_length = encoded.shape[-1]
        positions = torch.arange(sequence_length, device=encoded.device).unsqueeze(0)
        starts = sequence_length - valid_lengths.unsqueeze(1)
        mask = (positions >= starts).unsqueeze(1).to(encoded.dtype)
        pooled_mean = (encoded * mask).sum(dim=-1) / valid_lengths.unsqueeze(1).to(
            encoded.dtype
        )
        pooled_last = encoded[:, :, -1]
        parts = [pooled_last, pooled_mean]
        if self.static_encoder is not None:
            parts.append(self.static_encoder(static_values))
        return self.decoder(torch.cat(parts, dim=1))


@dataclass(frozen=True)
class ArrayBundle:
    dynamic: np.ndarray
    static: np.ndarray
    lengths: np.ndarray
    target: np.ndarray


class ForecastDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, bundle: ArrayBundle) -> None:
        self.dynamic = torch.as_tensor(bundle.dynamic, dtype=torch.float32)
        self.static = torch.as_tensor(bundle.static, dtype=torch.float32)
        self.lengths = torch.as_tensor(bundle.lengths, dtype=torch.long)
        self.target = torch.as_tensor(bundle.target, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.dynamic)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.dynamic[index],
            self.static[index],
            self.lengths[index],
            self.target[index],
        )


@dataclass(frozen=True)
class TrainingResult:
    model: SmallTCNForecaster
    best_epoch: int
    best_validation_loss: float
    epochs_run: int
    history: tuple[dict[str, float], ...]


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _make_loader(
    bundle: ArrayBundle,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ForecastDataset(bundle),
        batch_size=min(batch_size, len(bundle.dynamic)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def train_with_early_stopping(
    model: SmallTCNForecaster,
    train_bundle: ArrayBundle,
    validation_bundle: ArrayBundle,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingResult:
    set_torch_seed(seed)
    train_loader = _make_loader(train_bundle, batch_size, True, seed)
    validation_loader = _make_loader(validation_bundle, batch_size, False, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for dynamic, static, lengths, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(dynamic, static, lengths)
            loss = loss_function(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(dynamic)
            train_count += len(dynamic)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for dynamic, static, lengths, target in validation_loader:
                loss = loss_function(model(dynamic, static, lengths), target)
                validation_loss_sum += float(loss.item()) * len(dynamic)
                validation_count += len(dynamic)
        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Early stopping did not save a model state")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_run=len(history),
        history=tuple(history),
    )


def train_fixed_epochs(
    model: SmallTCNForecaster,
    train_bundle: ArrayBundle,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> SmallTCNForecaster:
    set_torch_seed(seed)
    loader = _make_loader(train_bundle, batch_size, True, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    for _ in range(max(1, epochs)):
        model.train()
        for dynamic, static, lengths, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(dynamic, static, lengths), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def predict(model: SmallTCNForecaster, bundle: ArrayBundle, batch_size: int = 256) -> np.ndarray:
    loader = _make_loader(bundle, batch_size, False, seed=0)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for dynamic, static, lengths, _ in loader:
            predictions.append(model(dynamic, static, lengths).cpu().numpy())
    return np.vstack(predictions)
