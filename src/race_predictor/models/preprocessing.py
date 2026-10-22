"""Input preprocessing shared by the neural models (padding, sequence and static inputs)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from race_predictor.features import STATIC_BINARY, STATIC_NUMERIC

PAD_VALUE = 999.0  # fills every channel of a padded time step
MAX_LEN = 256  # longest sequence kept (the most recent runs); older runs are truncated


def pad_scaled_sequences(
    seqs: Sequence[np.ndarray], scaler: StandardScaler, max_len: int
) -> np.ndarray:
    """Standardise each run sequence and post-pad it to ``(len(seqs), max_len, n_features)``.

    NaNs become 0 before scaling; sequences longer than ``max_len`` keep their most recent runs.
    """
    from keras.utils import pad_sequences

    scaled = [scaler.transform(np.nan_to_num(seq)) for seq in seqs]
    return np.asarray(
        pad_sequences(
            scaled,
            maxlen=max_len,
            padding="post",
            truncating="pre",
            value=PAD_VALUE,
            dtype="float32",
        )
    )


def numeric_static(static: pd.DataFrame) -> np.ndarray:
    """Numeric static columns as a float array with NaNs replaced by 0 (input to the scaler)."""
    return np.nan_to_num(static[STATIC_NUMERIC].to_numpy(float))


def scaled_static(static: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Standardised numeric static features followed by the binary ones, as float32."""
    features = np.column_stack(
        [scaler.transform(numeric_static(static)), static[STATIC_BINARY].to_numpy(float)]
    )
    return np.nan_to_num(features).astype("float32")
