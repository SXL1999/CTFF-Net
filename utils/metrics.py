import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape

# 评价指标维度分解
def calculate_metrics(pred, true):
    metrics = []
    for i in range(pred.shape[2]):
        pred_column = pred[:, :, i]
        true_column = true[:, :, i]
        mae = np.mean(np.abs(pred_column - true_column))
        mse = np.mean((pred_column - true_column) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((pred_column - true_column) / true_column))
        mspe = np.mean(np.square((pred_column - true_column) / true_column))
        metrics.append((mae, mse, rmse, mape, mspe))
    return metrics

def save_metrics(metrics, file_name):
    with open(file_name, 'a') as f:  # 使用 'w' 模式，覆盖之前的内容；使用 'a' 模式，添加新的的内容
        f.write("评价指标\n")
        for idx, (mae, mse, rmse, mape, mspe) in enumerate(metrics):
            f.write(f'Column {idx+1} - MSE: {mse}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}\n')  # 添加换行符