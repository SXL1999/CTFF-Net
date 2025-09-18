import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from matplotlib import ticker
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'CTFF':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def visual(path, true, preds=None, name='./pic/test.png'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name)

def visual_Loss(train_losses, test_losses, name='./pic/loss.png'):
    plt.figure()
    plt.plot(train_losses, label='train_loss', linewidth=2)
    plt.plot(test_losses, label='test_loss', linewidth=2)
    plt.legend()
    plt.savefig(name)
    plt.show()


def visual_MeMetrics(mae, rmse):
    pass


def writer_Metrics(mae, mse, rmse, mape, model):
    folder = 'saved_Metrics'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, model + "_" + 'metrics.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'MAE', 'MSE', 'RMSE', 'MAPE'])
        for epoch in range(len(rmse)):
            writer.writerow([epoch + 1, mae[epoch], mse[epoch], rmse[epoch], mape[epoch]])


def writer_Loss(train_loss, test_loss, model):
    folder = 'saved_loss'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, model + "_" + 'loss.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])
        for epoch in range(len(train_loss)):
            writer.writerow([epoch + 1, train_loss[epoch], test_loss[epoch]])


def visual_TrajectoryPrediction(predictions, targets):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    predictions = predictions[::12, :, :].reshape(-1, 3)
    targets = targets[::12, :, :].reshape(-1, 3)
    figure, ax = plt.subplots(3, 1, figsize=(8, 6))
    title = ["雷达搜索目标", "高低", "斜距"]
    for i in range(3):
        ax[i].plot(predictions[:, i], 'r', linewidth=2, label="预测轨迹")
        ax[i].plot(targets[:, i], "b--", linewidth=2, label="真实轨迹")
        legend = ax[i].legend()
        ax[i].add_artist(legend)
        ax[i].set_title(title[i])
    plt.tight_layout()
    directory = "./saved_error/PatchFourier/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "轨迹预测曲线.png")
    # plt.show()


def visual_Error(predictions, targets):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    predictions = predictions[::12, :, :].reshape(-1, 3)
    targets = targets[::12, :, :].reshape(-1, 3)
    figure, ax = plt.subplots(3, 2, figsize=(15, 10))
    title = ["雷达搜索目标误差", "高低误差", "斜距误差"]
    title1 = ["雷达搜索目标百分比误差", "高低百分比误差", "斜距百分比误差"]
    for i in range(3):
        ax[i][0].plot(targets[:, i] - predictions[:, i], 'b', linewidth=2)
        ax[i][1].plot(np.abs(targets[:, i] - predictions[:, i]) / np.abs(targets[:, i]), 'k', linewidth=2)
        ax[i][0].set_title(title[i])
        ax[i][1].set_title(title1[i])
        formatter = ticker.PercentFormatter(xmax=1, decimals=1)
        ax[i][1].yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("./saved_error/PatchFourier/预测误差.png")
    # plt.show()