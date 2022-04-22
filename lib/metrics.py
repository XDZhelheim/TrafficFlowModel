# This module is adapted from https://github.com/deepkashiwa20/DL-Traff-Graph

import numpy as np


def evaluate(y_pred, y_true):
    """
    (batch_size, N, timesteps)
    """
    dim = y_true.shape[2]
    if dim == 1:
        # single step case
        return (
            MSE(y_pred, y_true),
            RMSE(y_pred, y_true),
            MAE(y_pred, y_true),
            MAPE(y_pred, y_true),
        )
    else:
        # multi step case
        mse, rmse, mae, mape = (
            np.zeros(dim),
            np.zeros(dim),
            np.zeros(dim),
            np.zeros(dim),
        )

        for t in range(dim):
            mse[t] = MSE(y_pred[:, :, t], y_true[:, :, t])
            rmse[t] = RMSE(y_pred[:, :, t], y_true[:, :, t])
            mae[t] = MAE(y_pred[:, :, t], y_true[:, :, t])
            mape[t] = MAPE(y_pred[:, :, t], y_true[:, :, t])
        return mse, rmse, mae, mape


def MSE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_pred, y_true, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


# def MAPE(y_pred_, y_test_):
#     mask = (y_test_ != 0).astype(np.int)
#     return np.sum(np.nan_to_num(np.abs((y_pred_ - y_test_).astype(np.float32) / y_test_)) * mask) / np.sum(mask)

# def MSE(y_true, y_pred):
#     return np.mean(np.square(y_pred - y_true))

# def RMSE(y_true, y_pred):
#     return np.sqrt(MSE(y_pred, y_true))

# def MAE(y_true, y_pred):
#     return np.mean(np.abs(y_pred - y_true))

# def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-3):       # avoid zero division
#     return np.mean(np.abs(y_pred - y_true) / np.clip((np.abs(y_pred) + np.abs(y_true)) * 0.5, epsilon, None))

# def PCC(y_pred:np.array, y_true:np.array):      # Pearson Correlation Coefficient
#     return np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1]
