import datetime
from datetime import datetime as dt
import math
import time
import numpy as np
from model import *
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import os
from train_util import *
import optuna
import torch.optim as optim
from torchsummary import summary
from trained_model_load import plot_model_predictions



def main(args: argparse.Namespace, trial=None):

    if trial is not None: args.batch_size = 2 ** trial.suggest_int('batch_size', 2, 6)

    model_log_path = os.path.join('model_log', args.save_file)
    model_path = os.path.join(model_log_path, 'trained_model')
    trials_log_path = os.path.join(model_log_path, 'trials_loss')
    modelError_path = os.path.join(model_log_path, 'modelError')
    path_exists(model_path, trials_log_path, modelError_path)

    file_name_prefix = trial.number if trial is not None else 'temp'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('---------Train on: ' + device + '----------')

    if args.seed is not None:
        set_seed(args.seed)

    train_loader, val_loader = get_loader(args)

    input_size = tuple(train_loader.dataset.x.shape[1:])
    output_size = int(train_loader.dataset.y.shape[1])

    # ECA_ResNet18 ResNet18 ResNet181D CNN ECA_ResNet181D
    model = CNN(output_size).to(device)
    summary(model, input_size=input_size)
    model.apply(init_weights)
    print('model init_weights completed.')

    # CrossEntropyLoss() MSELoss()
    criterion = nn.MSELoss(reduction='sum')

    # ["Adam", "RMSprop", "SGD", "Adagrad"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=int(0.25 * args.epochs), gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(0.15 * args.epochs))
    # scheduler = CosineAnnealingLR(optimizer, T_max=50)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.1,
                                                  mode='triangular2',
                                                  cycle_momentum=False,
                                                  step_size_up=int(0.05 * args.epochs),
                                                  step_size_down=int(0.15 * args.epochs))

    train_loss = np.array([])
    val_loss = np.array([])
    learning_rate = np.array([])

    train_type = 'regression'
    best_loss = math.inf
    goal_loss = math.inf
    best_epoch = 0

    # 调参项目
    if trial is not None:
        """超参数调整"""
        # 学习率范围
        # lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        # ["Adam", "RMSprop", "SGD", "Adagrad"]
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "Adagrad"])
        # 训练次数
        epochs = trial.suggest_int('epochs', 200, 500, step=50)

        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=int(0.1 * args.epochs))
        args.epochs = epochs

    for epoch in range(args.epochs):
        train_batch_loss = train(epoch, train_loader, model, optimizer, criterion, args, train_type)
        train_loss = np.append(train_loss, train_batch_loss)

        val_batch_acc = val(epoch, val_loader, model, criterion, args, train_type)
        val_loss = np.append(val_loss, val_batch_acc)

        if train_batch_loss < 1:  # 训练集损失函数阈值
            goal_loss = val_batch_acc
            # 保存训练过程中目标值最小的一个的模型
            if goal_loss < best_loss:
                best_loss = goal_loss
                best_epoch = epoch + 1
                torch.save(model, f'{model_path}/optuna_model_{file_name_prefix}_BEST.pt')
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # print(f'Epoch: {epoch + 1}/{args.epochs}, lr: {lr}')
        learning_rate = np.append(learning_rate, lr)
        # scheduler.step(val_batch_acc)
        scheduler.step()

    print(f'{file_name_prefix} best_val_loss: {best_loss}, epoch: {best_epoch}')
    draw_regression_loss(train_loss, val_loss,
                         title=f'{file_name_prefix}--Loss',
                         save_path=f'{trials_log_path}/{file_name_prefix}.png')
    save_losses(train_loss, val_loss, f'{trials_log_path}/{file_name_prefix}--Loss')  # 保存损失值
    torch.save(model, f'{model_path}/optuna_model_{file_name_prefix}.pt')  # 保存最终训练的模型

    _ = modelError(model, train_loader, save_data=1,
                   save_file_name=f'{modelError_path}/train_{file_name_prefix}')
    mean_rel_error = modelError(model, val_loader, save_data=1,
                                save_file_name=f'{modelError_path}/val_{file_name_prefix}')

    # 绘制学习率曲线
    plot_learning_rate(learning_rate, title=f'{file_name_prefix}--Learning Rate')
    # 绘制模型预测值
    plot_model_predictions(model, train_loader, val_loader, title=f'{file_name_prefix} Model Predictions')

    return mean_rel_error


def objective(trial):
    min_val = main(args, trial)
    return min_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seed', default=40, type=int)
    # parser.add_argument('-tp', '--x_path', default='data/reproducted/valid_origin_data_16plus50/coding_data.npy')
    # parser.add_argument('-vp', '--y_path', default='data/reproducted/valid_origin_data_16plus50/shift_value_manual.npy')
    parser.add_argument('-tp', '--x_path', default='data/reproducted/coding_data.npy')
    parser.add_argument('-vp', '--y_path', default='data/reproducted/shift_value.npy')

    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-ts', '--test_size', type=float, default=0.1)
    parser.add_argument('-lr', '--lr', type=float, default=0.1)
    parser.add_argument('-sf', '--save_file', default='temp_file')
    args = parser.parse_args()

    # optuna_model = int(input('Optimize model hyperparameters by optuna [eg: 0 name]: ')) != 0
    optuna_model, folder_name = input('Optimize model hyperparameters by optuna [eg: 1 name]: ').split() \
                                or (0, 'Temp_file_' + dt.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if int(optuna_model) == 1:
        optuna_model = True
    else:
        optuna_model = False

    if folder_name is None:
        file_prefix = 'Temp_file_' + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_file = folder_name

    print('---------Train type optuna_model = ' + str(optuna_model) + '----------')
    time.sleep(2)

    if optuna_model:
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                    direction='minimize',
                                    storage='sqlite:///db.sqlite3',
                                    study_name=args.save_file)
        study.optimize(objective, n_trials=15)
        # 命令行工具 optuna-dashboard sqlite:///db.sqlite3
        # http://127.0.0.1:8080/
        print(study.best_params)
    else:
        min_value = main(args)
        print(f'min_value= {min_value}')
