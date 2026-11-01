"""
Powerwise model ported from JS to Python

Usage:
 - In Colab/Anaconda: `pip install numpy pandas`
 - Run this script or import `train_model_from_df` and `predict_values`.

This reproduces the closed-form multi-output linear regression used in the project's
`js/model.js` and demonstrates training on the included sample dataset.
"""
from __future__ import annotations
import os
import io
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

DEFAULT_FEATURE_COLUMNS = ['hour', 'temperature', 'previous_consumption']
DEFAULT_LABEL_COLUMNS = ['consumption', 'wastage']


class ModelState:
    def __init__(self):
        self.model = None
        self.feature_means: List[float] = []
        self.feature_stds: List[float] = []
        self.target_means: List[float] = []
        self.target_stds: List[float] = []
        self.trained = False
        self.dataset: pd.DataFrame = pd.DataFrame()
        self.feature_columns = DEFAULT_FEATURE_COLUMNS.copy()
        self.label_columns = DEFAULT_LABEL_COLUMNS.copy()


model_state = ModelState()


def vector_mean_std(df: pd.DataFrame, columns: List[str]) -> Tuple[List[float], List[float]]:
    means = [df[col].astype(float).mean() for col in columns]
    stds = [float(df[col].astype(float).std(ddof=0)) for col in columns]
    # fallback to 1 to avoid division by zero (mirrors JS behavior)
    stds = [s if s and not np.isnan(s) else 1.0 for s in stds]
    return means, stds


def normalize_array(arr: np.ndarray, means: List[float], stds: List[float]) -> np.ndarray:
    means = np.array(means)
    stds = np.array(stds)
    return (arr - means) / stds


def preprocess_df(df: pd.DataFrame, feature_columns: List[str], target_columns: List[str]) -> pd.DataFrame:
    # Keep only rows where all required columns are present and numeric
    cols = list(dict.fromkeys(feature_columns + target_columns))
    sub = df[cols].copy()
    sub = sub.dropna()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sub = sub.dropna()
    if sub.shape[0] == 0:
        raise ValueError('No valid rows found after filtering. Check your dataset format.')
    return sub


def train_model_from_df(df: pd.DataFrame, feature_columns: List[str] = None, target_columns: List[str] = None,
                        min_rows: int = 8) -> None:
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS.copy()
    if target_columns is None:
        target_columns = DEFAULT_LABEL_COLUMNS.copy()

    processed = preprocess_df(df, feature_columns, target_columns)
    if processed.shape[0] < min_rows:
        raise ValueError(f'Insufficient data: {processed.shape[0]} valid rows found. Need at least {min_rows} valid rows for training.')

    feature_means, feature_stds = vector_mean_std(processed, feature_columns)
    target_means, target_stds = vector_mean_std(processed, target_columns)

    X_raw = processed[feature_columns].astype(float).to_numpy()
    Y_raw = processed[target_columns].astype(float).to_numpy()

    X_norm = normalize_array(X_raw, feature_means, feature_stds)
    Y_norm = normalize_array(Y_raw, target_means, target_stds)

    # Add bias column
    ones = np.ones((X_norm.shape[0], 1))
    X = np.hstack([ones, X_norm])  # shape (n, p)

    # Normal equation for multi-output: W = (X^T X)^{-1} X^T Y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # add tiny ridge for numerical stability
        XtX_inv = np.linalg.inv(XtX + np.eye(XtX.shape[0]) * 1e-8)

    W = XtX_inv @ X.T @ Y_norm  # shape (p, k)

    bias = W[0, :]
    weights = W[1:, :]

    model_state.model = {'weights': weights, 'bias': bias}
    model_state.feature_means = feature_means
    model_state.feature_stds = feature_stds
    model_state.target_means = target_means
    model_state.target_stds = target_stds
    model_state.feature_columns = feature_columns
    model_state.label_columns = target_columns
    model_state.trained = True
    model_state.dataset = processed


def predict_values(feature_values: List[float] or Dict[str, Any]) -> Dict[str, float]:
    if not model_state.trained or model_state.model is None:
        raise RuntimeError('Model not trained')

    if isinstance(feature_values, dict):
        features = [float(feature_values[c]) for c in model_state.feature_columns]
    else:
        features = list(map(float, feature_values))

    if len(features) != len(model_state.feature_columns):
        raise ValueError(f'Expected {len(model_state.feature_columns)} features, got {len(features)}')

    x_norm = normalize_array(np.array(features), model_state.feature_means, model_state.feature_stds)

    # compute prediction: bias + weights^T * x_norm
    weighted = model_state.model['bias'] + model_state.model['weights'].T @ x_norm

    # un-normalize
    preds = (weighted * np.array(model_state.target_stds)) + np.array(model_state.target_means)

    return {col: float(preds[i]) for i, col in enumerate(model_state.label_columns)}


def get_dataset_info() -> Dict[str, Any]:
    if model_state.dataset is None or model_state.dataset.empty:
        return {'totalRows': 0, 'featureColumns': model_state.feature_columns, 'targetColumns': model_state.label_columns}

    info = {
        'totalRows': int(model_state.dataset.shape[0]),
        'featureColumns': model_state.feature_columns,
        'targetColumns': model_state.label_columns,
        'featureStats': {},
        'targetStats': {},
    }

    for col in model_state.feature_columns:
        vals = model_state.dataset[col].astype(float)
        info['featureStats'][col] = {'min': float(vals.min()), 'max': float(vals.max()), 'avg': float(vals.mean())}

    for col in model_state.label_columns:
        vals = model_state.dataset[col].astype(float)
        info['targetStats'][col] = {'min': float(vals.min()), 'max': float(vals.max()), 'avg': float(vals.mean())}

    return info


def make_forecast_advice(predictions: Dict[str, float]) -> str:
    advice = 'Prediction Analysis:\n'
    for k, v in predictions.items():
        advice += f'• {k}: {v:.2f}\n'
    high = [k for k, v in predictions.items() if v > 100]
    if high:
        advice += '\n⚠️ High consumption detected. Consider:\n'
        advice += '• Checking for equipment inefficiencies\n'
        advice += '• Reviewing usage patterns\n'
        advice += '• Implementing energy-saving measures\n'
    return advice


SAMPLE_CSV = """
hour,temperature,previous_consumption,consumption,wastage
0,18.5,0,2.3,0.5
1,17.8,2.3,2.1,0.4
2,17.2,2.1,1.8,0.3
3,16.9,1.8,1.6,0.2
4,16.5,1.6,1.9,0.4
5,17.1,1.9,2.8,0.7
6,18.2,2.8,4.2,1.1
7,19.8,4.2,5.8,1.5
8,21.5,5.8,6.9,1.8
9,23.1,6.9,7.2,1.9
10,24.8,7.2,7.8,2.1
11,26.2,7.8,8.1,2.2
12,27.5,8.1,8.5,2.3
13,28.1,8.5,8.2,2.2
14,27.8,8.2,7.9,2.1
15,26.9,7.9,7.1,1.9
16,25.5,7.1,6.8,1.8
17,24.2,6.8,7.2,1.9
18,22.8,7.2,8.1,2.2
19,21.2,8.1,9.2,2.4
20,19.8,9.2,8.8,2.3
21,18.9,8.8,6.5,1.7
22,18.1,6.5,4.2,1.1
23,17.5,4.2,3.1,0.8
"""


def demo(use_embedded: bool = True, csv_path: str = 'easy_sample.csv') -> None:
    # load data
    if use_embedded or not os.path.exists(csv_path):
        df = pd.read_csv(io.StringIO(SAMPLE_CSV.strip()))
    else:
        df = pd.read_csv(csv_path)

    print('Loaded rows:', df.shape[0])
    train_model_from_df(df)
    print('Model trained on', model_state.dataset.shape[0], 'rows')
    print('Dataset info:', get_dataset_info())

    # example prediction
    sample_input = {'hour': 14, 'temperature': 27.0, 'previous_consumption': 8.2}
    preds = predict_values(sample_input)
    print('\nSample input:', sample_input)
    print('Predictions:')
    for k, v in preds.items():
        print(f' - {k}: {v:.3f}')
    print('\nAdvice:\n', make_forecast_advice(preds))


if __name__ == '__main__':
    # Run demo when executed directly
    try:
        demo()
    except Exception as e:
        print('Error during demo:', e)
