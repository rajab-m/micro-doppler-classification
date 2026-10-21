import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import torch



def normalize_data(X_train, X_val, normalization="global", eps=1e-8):
    """
    Normalize training and validation data.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape (n_samples, ...).
    X_val : np.ndarray
        Validation data, shape (n_samples, ...).
    normalization : str
        "global": Fit normalization on the entire training set.
        "local": Normalize each sample independently.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    X_train_norm, X_val_norm : np.ndarray
        Normalized arrays with their original shapes.
    """

    if normalization == "global":
        # Preserve the original sample shape
        original_shape = X_train.shape[1:]

        # Flatten each sample
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)

        # Fit only on training data
        scaler = StandardScaler()

        X_train_norm = scaler.fit_transform(X_train_flat)
        X_val_norm = scaler.transform(X_val_flat)

        # Restore original shapes
        X_train_norm = X_train_norm.reshape(len(X_train), *original_shape)
        X_val_norm = X_val_norm.reshape(len(X_val), *original_shape)

    elif normalization == "local":
        # Normalize each sample independently
        # Calculate statistics across all axes except the sample axis
        axes = tuple(range(1, X_train.ndim))

        train_mean = X_train.mean(axis=axes, keepdims=True)
        train_std = X_train.std(axis=axes, keepdims=True)

        val_axes = tuple(range(1, X_val.ndim))
        val_mean = X_val.mean(axis=val_axes, keepdims=True)
        val_std = X_val.std(axis=val_axes, keepdims=True)

        X_train_norm = (X_train - train_mean) / np.maximum(train_std, eps)

        X_val_norm = (X_val - val_mean) / np.maximum(val_std, eps)

    else:
        raise ValueError("normalization must be 'global' or 'local'")

    return X_train_norm, X_val_norm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)