import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].copy()
