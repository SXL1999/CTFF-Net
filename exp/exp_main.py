from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CTFF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, visual, visual_Loss, \
    writer_Metrics, writer_Loss, visual_TrajectoryPrediction, visual_Error
from utils.metrics import metric, calculate_metrics, save_metrics
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'CTFF': CTFF
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _count_params(self):
        # 统计模型参数量 (M)
        param_count = sum(p.numel() for p in self.model.parameters())
        return param_count / 1e6  # 转为百万 (M)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # 验证
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        gt = []
        pr = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TST' in self.args.model or 'CTFF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)
                else:
                    if 'TST' in self.args.model or 'CTFF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                pred = pred[:, :, :3]
                true = true[:, :, :3]
                gt.append(true)
                pr.append(pred)

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        gt = np.concatenate(gt, 0)
        pr = np.concatenate(pr, 0)
        mae, mse, rmse, mape = metric(pr, gt)
        self.model.train()
        return total_loss, mae, mse, rmse, mape

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        train_losses = []
        test_losses = []
        MAE = []
        MSE = []
        RMSE = []
        MAPE = []
        best_test_loss = np.inf

        for epoch in range(self.args.train_epochs):
            train_loss = []
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TST' in self.args.model or 'CTFF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'TST' in self.args.model or 'CTFF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x, dec_inp, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'CTFF':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, mae, mse, rmse, mape = self.vali(test_data, test_loader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            MAE.append(mae)
            MSE.append(mse)
            RMSE.append(rmse)
            MAPE.append(mape)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss))

            # early_stopping(test_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            if self.args.lradj != 'CTFF':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

            if test_loss < best_test_loss:
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                best_test_loss = test_loss

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        visual_Loss(train_losses, test_losses, os.path.join(folder_path, "loss" + '.png'))
        writer_Loss(train_losses, test_losses, self.args.model)
        writer_Metrics(MAE, MSE, RMSE, MAPE, self.args.model)

        params_m = self._count_params()
        print(f"[Model Params] {params_m:.3f} M")

        save_dir = './results/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{setting}_params.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({"model": self.args.model, "params_M": round(params_m, 3)}, f, indent=2)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TST' in self.args.model or 'CTFF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)
                else:
                    if 'TST' in self.args.model or 'CTFF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]

                        else:
                            outputs = self.model(batch_x, dec_inp)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs[:, :, :3]
                true = batch_y[:, :, :3]

                preds.append(pred)
                trues.append(true)

                inputx.append(batch_x.detach().cpu().numpy())

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        inputx = np.concatenate(inputx,axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape= metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}'.format(mae, mse, rmse, mape))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}'.format(mae, mse, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        metrics = calculate_metrics(preds, trues)
        save_metrics(metrics, "results.txt")

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        mean = pred_data.mean
        std = pred_data.std

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []
        observations = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'TST' in self.args.model or 'CTFF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, dec_inp)[0]
                            else:
                                outputs = self.model(batch_x, dec_inp)
                else:
                    if 'TST' in self.args.model or 'CMultivariateTimeSeriesDatasetTFF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, dec_inp)[0]
                        else:
                            outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()
                observation = batch_x.cpu().detach().numpy()

                pred = pred[:, :, :3] * std + mean
                true = true[:, :, :3] * std + mean
                observation = observation[:, :, :3] * std + mean

                preds.append(pred)
                trues.append(true)
                observations.append(observation)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        observations = np.concatenate(observations, 0)

        # result save
        folder_path = './predict_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'prediction.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'observation.npy', observations)

        visual_TrajectoryPrediction(preds, trues)
        visual_Error(preds, trues)

        return
