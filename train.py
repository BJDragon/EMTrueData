import datetime
from datetime import datetime as dt
import math
import time
from model import *
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import os
from train_util import *
import optuna
import torch.optim as optim
from torchsummary import summary
from trained_model_load import plot_model_predictions


def transform(x, y):
    # 函数内的函数，用于对数据转换变形等操作
    # x = x.reshape(-1, 1, 5, 5) - 0.1
    # x = x*100
    x = x.reshape(-1, 1, 25) - 0.1
    return x, y


def main(args: argparse.Namespace, trial=None):
    """
    主函数用于执行模型训练和验证。
    参数:
    - args: 包含训练过程所需参数的命名空间。
    - trial: Optuna试验对象，用于 hyper-parameter 调优。如果为None，则表示不进行调优。

    """
    today = datetime.date.today()

    model_log_path = os.path.join('model_log', args.save_file)
    model_path = os.path.join(model_log_path, 'trained_model')
    trials_log_path = os.path.join(model_log_path, 'trials_loss')
    modelError_path = os.path.join(model_log_path, 'modelError')
    path_exists(model_path)
    path_exists(trials_log_path)
    path_exists(modelError_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('---------Train on: ' + device + '----------')

    if args.seed is not None:
        set_seed(args.seed)

    train_loader, val_loader = get_loader(args)

    input_size = tuple(train_loader.dataset.x.shape[1:])
    output_size = int(train_loader.dataset.y.shape[1])



    # # ECA_ResNet18 ResNet18 ResNet181D
    # model = ECA_ResNet18(output_size).to(device)
    # summary(model, input_size=input_size)
    # model.apply(init_weights)
    # print('model init_weights completed.')

    # 数据准备和模型配置
    num_layers = trial.suggest_int('num_layers', 3, 10)
    hidden_units = [trial.suggest_int(f'n_units_l{i}', 32, 512, step=32) for i in range(num_layers)]
    activation = [trial.suggest_categorical(f'activation_l{i}', ['relu', 'tanh', 'sigmoid', 'None']) for i in
                  range(num_layers)]
    model = DynamicFCNN(input_size=25, output_size=1, num_layers=num_layers, hidden_units=hidden_units,
                        activation=activation).to(device)
    summary(model, input_size=input_size)

    # CrossEntropyLoss() MSELoss()
    criterion = nn.MSELoss(reduction='sum')

    # ["Adam", "RMSprop", "SGD", "Adagrad"]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=int(0.25 * args.epochs), gamma=0.1)

    train_loss = np.array([])
    val_loss = np.array([])

    train_type = 'regression'
    best_loss = math.inf
    goal_loss = math.inf
    best_epoch = 0

    # 调参项目
    if trial is not None:
        """超参数调整"""
        # 学习率范围
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        # ["Adam", "RMSprop", "SGD", "Adagrad"]
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "Adagrad"])
        # 训练次数
        epochs = trial.suggest_int('epochs', 100, 1000)

        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
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
                    torch.save(model, f'{model_path}/optuna_model_{trial.number}_BEST.pt')

            scheduler.step()

        print(f'{trial.number} best_val_loss: {best_loss}, epoch: {best_epoch}')
        draw_regression_loss(train_loss, val_loss,
                             title=f'{trial.number}--Loss',
                             save_path=f'{trials_log_path}/{trial.number}.png')
        save_losses(train_loss, val_loss, f'{trials_log_path}/{trial.number}--Loss')  # 保存损失值
        torch.save(model, f'{model_path}/optuna_model_{trial.number}.pt')  # 保存最终训练的模型

        modelError(model, train_loader, save_data=1,
                   save_file_name=f'{modelError_path}/train_{trial.number}')
        mean_rel_error = modelError(model, val_loader, save_data=1,
                                    save_file_name=f'{modelError_path}/val_{trial.number}')
        return mean_rel_error
    else:
        for epoch in range(args.epochs):

            train_batch_loss = train(epoch, train_loader, model, optimizer, criterion, args, train_type)
            train_loss = np.append(train_loss, train_batch_loss)

            val_batch_acc = val(epoch, val_loader, model, criterion, args, train_type)
            val_loss = np.append(val_loss, val_batch_acc)

            if train_batch_loss < 1:
                goal_loss = val_batch_acc
                # 保存训练过程中目标值最小的一个的模型
                if goal_loss < best_loss:
                    best_loss = goal_loss
                    best_epoch = epoch + 1
                    torch.save(model, f'{model_path}/model_{today}_best.pt')
            scheduler.step()

        draw_regression_loss(train_loss, val_loss, save_path=f'{trials_log_path}/Training and Validation Losses.png')
        save_losses(train_loss, val_loss, f'{trials_log_path}/Training and Validation Losses.csv')
        torch.save(model, f'{model_path}/model_{today}.pt')

        modelError(model, train_loader, save_data=1, save_file_name=f'{modelError_path}/train')
        mean_rel_error = modelError(model, val_loader, save_data=1, save_file_name=f'{modelError_path}/val')
        return mean_rel_error
    # plot_model_predictions(model, train_loader, val_loader, title='Model Predictions')


def objective(trial):
    min_val = main(args, trial)
    return min_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seed', default=42, type=int)
    parser.add_argument('-tp', '--x_path', default='data/reproducted/coding_data.npy')
    parser.add_argument('-vp', '--y_path', default='data/reproducted/shift_value.npy')
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('-ep', '--epochs', type=int, default=200)
    parser.add_argument('-ts', '--test_size', type=float, default=0.1)
    parser.add_argument('-lr', '--lr', type=float, default=0.01)
    parser.add_argument('-sf', '--save_file', default='temp_file')
    args = parser.parse_args()

    optuna_model = int(input('Optimize model hyperparameters by optuna [ 0: False, 1: True]: ')) != 0
    print('---------Train type optuna_model = ' + str(optuna_model) + '----------')
    time.sleep(2)

    if optuna_model:
        args.save_file = str(input('study_name:')) + '_' + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                    direction='minimize',
                                    storage='sqlite:///db.sqlite3',
                                    study_name=args.save_file)
        study.optimize(objective, n_trials=5)
        # 命令行工具 optuna-dashboard sqlite:///db.sqlite3
        print(study.best_params)
    else:
        if input('Rename the work [ 0: False, 1: True]: ') != 0:
            args.save_file = str(input('Rename the work:')) + '_' + dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        min_value = main(args)
        print(f'min_value= {min_value}')
