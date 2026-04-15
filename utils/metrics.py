from typing import Any

import numpy as np
from numpy import floating


def RSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Relative Squared Error（相对平方误差）

    计算预测值与真实值之间的残差平方和，并与真实值相对其均值的离散程度做归一化。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            相对平方误差，值越小通常表示预测效果越好
    """


    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred: np.ndarray, true: np.ndarray):
    """
    Correlation（相关系数）

    按列计算预测值与真实值之间的相关性，再对所有列的相关系数取平均。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            平均相关系数，值越接近 1 表示预测与真实越一致
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred: np.ndarray, true: np.ndarray):
    """
    Mean Absolute Error（平均绝对误差）

    计算预测值与真实值之间绝对误差的平均值。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            平均绝对误差，值越小越好
    """
    return np.mean(np.abs(true - pred))


def MSE(pred: np.ndarray, true: np.ndarray):
    """
    Mean Squared Error（均方误差）

    计算预测值与真实值之间平方误差的平均值。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            均方误差，值越小越好
    """
    return np.mean((true - pred) ** 2)


def RMSE(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Root Mean Squared Error（均方根误差）

    对均方误差开平方，更直观地反映预测误差的尺度。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            均方根误差，值越小越好
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred: np.ndarray, true: np.ndarray) -> floating[Any]:
    """
    Mean Absolute Percentage Error（平均绝对百分比误差）

    计算预测值相对于真实值的绝对百分比误差平均值。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            平均绝对百分比误差，值越小越好
    """
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred: np.ndarray, true: np.ndarray) -> floating[Any]:
    """
    Mean Squared Percentage Error（平均平方百分比误差）

    计算预测值相对于真实值的平方百分比误差平均值。

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        float
            平均平方百分比误差，值越小越好

    注意:
        当 true 中存在 0 时，可能会出现除零问题
    """
    return np.mean(np.square((true - pred) / true))


def metric(
    pred: np.ndarray, true: np.ndarray
) -> tuple[floating[Any], floating[Any], float, floating[Any], floating[Any]]:
    """
    汇总多个常用评估指标

    依次计算:
        - MAE: 平均绝对误差
        - MSE: 均方误差
        - RMSE: 均方根误差
        - MAPE: 平均绝对百分比误差
        - MSPE: 平均平方百分比误差

    参数:
        pred: ndarray
            预测值
        true: ndarray
            真实值

    返回:
        tuple
            (mae, mse, rmse, mape, mspe)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
