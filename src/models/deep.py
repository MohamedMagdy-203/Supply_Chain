"""Neural-network quantile forecasting model (Deep Learning stage)."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import QUANTILE_LEVELS, SEED
from src.data import load_xy, save_predictions


@dataclass(frozen=True)
class Candidate:
    hidden_layers: tuple[int, ...]
    dropout: float
    learning_rate: float


CANDIDATES = (
    Candidate(hidden_layers=(64, 32), dropout=0.10, learning_rate=0.001),
    Candidate(hidden_layers=(128, 64), dropout=0.10, learning_rate=0.001),
    Candidate(hidden_layers=(128, 64, 32), dropout=0.20, learning_rate=0.0005),
)
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE = 10


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted on training rows only."""
    categorical = list(X_train.select_dtypes(include=["category", "object"]).columns)
    numeric = [column for column in X_train.columns if column not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def build_network(input_size: int, candidate: Candidate) -> nn.Sequential:
    """Create a feed-forward network with one output per forecast quantile."""
    layers: list[nn.Module] = []
    previous_size = input_size
    for hidden_size in candidate.hidden_layers:
        layers.extend([
            nn.Linear(previous_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(candidate.dropout),
        ])
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, len(QUANTILE_LEVELS)))
    return nn.Sequential(*layers)


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                 quantiles: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss across rows and requested quantiles."""
    error = y_true - y_pred
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def predict(model: nn.Module, X: np.ndarray, y_mean: float,
            y_scale: float, device: torch.device) -> np.ndarray:
    """Return non-negative, ordered quantile predictions in original units."""
    model.eval()
    with torch.no_grad():
        values = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    values = values * y_scale + y_mean
    values = np.sort(values, axis=1)
    return np.maximum(values, 0.0)


def train_candidate(candidate: Candidate, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, y_mean: float,
                    y_scale: float, device: torch.device):
    """Train one candidate and keep the checkpoint with lowest validation loss."""
    torch.manual_seed(SEED)
    model = build_network(X_train.shape[1], candidate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate)
    quantiles = torch.tensor(
        list(QUANTILE_LEVELS.values()), dtype=torch.float32, device=device
    ).view(1, -1)

    train_dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = pinball_loss(batch_y, outputs, quantiles)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = float(pinball_loss(y_val_tensor, val_outputs, quantiles).item())

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    val_predictions = predict(model, X_val, y_mean, y_scale, device)
    y_val_original = y_val * y_scale + y_mean
    quantile_values = np.asarray(list(QUANTILE_LEVELS.values()))
    errors = y_val_original[:, None] - val_predictions
    score = float(np.maximum(quantile_values * errors,
                             (quantile_values - 1.0) * errors).mean())
    return model, score, best_epoch


def main() -> None:
    """Tune on validation data, then save validation and test forecasts."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deep learning device: {device}", flush=True)

    X_train, y_train, _ = load_xy("train")
    X_val, y_val, keys_val = load_xy("val")
    X_test, _, keys_test = load_xy("test")

    preprocessor = build_preprocessor(X_train)
    X_train_ready = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_ready = preprocessor.transform(X_val).astype(np.float32)
    X_test_ready = preprocessor.transform(X_test).astype(np.float32)

    y_train_values = y_train.to_numpy(dtype=np.float32)
    y_val_values = y_val.to_numpy(dtype=np.float32)
    y_mean = float(y_train_values.mean())
    y_scale = float(y_train_values.std()) or 1.0
    y_train_scaled = (y_train_values - y_mean) / y_scale
    y_val_scaled = (y_val_values - y_mean) / y_scale

    best_model = None
    best_candidate = None
    best_score = float("inf")
    best_epoch = 0

    for index, candidate in enumerate(CANDIDATES, start=1):
        model, score, epoch = train_candidate(
            candidate, X_train_ready, y_train_scaled,
            X_val_ready, y_val_scaled, y_mean, y_scale, device
        )
        print(
            f"Candidate {index}/{len(CANDIDATES)} {candidate}: "
            f"validation pinball={score:.6f} (best epoch={epoch})",
            flush=True,
        )
        if score < best_score:
            best_model = model
            best_candidate = candidate
            best_score = score
            best_epoch = epoch

    if best_model is None:
        raise RuntimeError("No deep-learning candidate completed successfully.")

    print(f"Selected parameters: {best_candidate}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation pinball loss: {best_score:.6f}")

    val_predictions = predict(best_model, X_val_ready, y_mean, y_scale, device)
    test_predictions = predict(best_model, X_test_ready, y_mean, y_scale, device)

    val_frame = keys_val.copy()
    test_frame = keys_test.copy()
    for index, name in enumerate(QUANTILE_LEVELS):
        val_frame[name] = val_predictions[:, index]
        test_frame[name] = test_predictions[:, index]

    predictions = pd.concat([val_frame, test_frame], ignore_index=True)
    save_predictions(predictions, "deep")


if __name__ == "__main__":
    main()