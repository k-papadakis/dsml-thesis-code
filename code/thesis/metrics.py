import torch


def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def mape(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def smape(y_true, y_pred):
    return 2 * torch.mean(
        torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred))
    )


def mdape(y_true, y_pred):
    return torch.median(torch.abs((y_true - y_pred) / y_true))


def mase(y_true, y_pred):
    return torch.mean(
        torch.abs(y_true - y_pred) / torch.mean(torch.abs(y_true[1:] - y_true[:-1]))
    )


METRICS = (mse, rmse, mae, mape, mdape, smape, mase)
