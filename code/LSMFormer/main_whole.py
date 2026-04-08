import pandas as pd
import numpy as np
import torch
import argparse
import os
from pathlib import Path
from torch import nn
from imblearn.over_sampling import SMOTE
from train_whole import Train
from utils_whole import *
from CL_GAF_probAtten_Res_for_RC import new_BY, BYOL
import sys


# 按给定比例获取原始数据和过采样数据
def get_balance_SMOTE_data(train_data, neg_data, rate=100, l_name='label', root_path=None):
    np.random.seed(256)
    pos_df = train_data[train_data[l_name]==1]
    shape_ = pos_df.shape[0]
    if shape_*rate > neg_data.shape[0]:
        print("Out of count of negative samples!")
        return None
    neg_random = neg_data.sample(frac=1).iloc[:shape_ * rate]
    f_name = train_data.columns[:-1]
    # 上采样原数据
    df_ori = pd.concat([pos_df, neg_random])
    X = df_ori[f_name]
    y = df_ori[l_name]
    # print(X.info())
    # print(X.info())
    print('y counts before Oversampling：', y.value_counts())
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    print('y counts after Oversampling：', y.value_counts())
    ovSam_data = pd.concat([X, y], axis=1, sort=False)
    smote_path = ''
    if root_path is not None:
        smote_path = os.path.join(root_path, f'rc_freqSMOTE{ovSam_data.shape[0]}.csv')
        ovSam_data.to_csv(smote_path)
        df_ori.to_csv(os.path.join(root_path, f'rc_ori{df_ori.shape[0]}.csv'))
    return smote_path, df_ori


# 获取扰动原始数据
def weak_augment(x, m):
    # x = x.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
    x=stander_data(x.iloc[:, :-1])
    mean_values = x.mean()
    var_values = x.var()
    data = x.T
    ai = []
    for mean, var in zip(mean_values, var_values):
        # ai.append(np.multiply(data, factor[:, :]))
        factor = np.random.normal(loc=mean, scale=var * m, size=(data.shape[0], 1))
        ai.append(np.multiply(data, factor))
        # print(type(ai))
        output = np.concatenate((ai), axis=1)
        ai.clear()
    augment1 = output.T
    for i in range(len(augment1)):
        random_num = np.random.randint(0, high=21, size=2)
        augment1[i][random_num[0]] = np.random.normal(loc=mean_values[random_num[0]], scale=var_values[random_num[0]],
                                                      size=1)
        augment1[i][random_num[1]] = np.random.normal(loc=mean_values[random_num[1]], scale=var_values[random_num[1]],
                                                      size=1)
    weak_data1 = pd.DataFrame(data=augment1, columns=x.columns)
    return weak_data1


class SongDataset(Dataset):
    def __init__(self, df1, df2, df3, device='cpu'):
        super().__init__()
        self.data1 = df1
        self.data2 = df2
        self.data3 = df3
        self.device = device
    def __len__(self):
        return self.data1.shape[0]

    def __getitem__(self, index):
        tens1 = torch.FloatTensor(self.data1.iloc[index, :].to_numpy()).to(self.device)
        tens2 = torch.FloatTensor(self.data2.iloc[index, :].to_numpy()).to(self.device)
        tens3 = torch.FloatTensor(self.data3.iloc[index, :].to_numpy()).to(self.device)

        return tens1, tens2, tens3


# 开始对比学习
def start_cl(ori_data, save_path, epoch, if_load=False, v1=0.5, v2=7.5, bat=2000):
    if if_load:
        wea1 = pd.read_csv(os.path.join(save_path, f'wea1_value{v1}.csv'))
        wea2 = pd.read_csv(os.path.join(save_path, f'wea2_value{v2}.csv'))
    else:
        wea1 = weak_augment(ori_data, v1)
        wea2 = weak_augment(ori_data, v2)
        wea1.to_csv(os.path.join(save_path, f'wea1_value{v1}_{wea1.shape[0]}.csv'))
        wea2.to_csv(os.path.join(save_path, f'wea2_value{v2}_{wea2.shape[0]}.csv'))
    st_data = stander_data(ori_data.iloc[:, :-1])
    ds = SongDataset(wea1, wea2, st_data, device='cuda')
    train_loader = DataLoader(ds, batch_size=bat, num_workers=0, shuffle=False)
    models_cl = BYOL(image_size=23).to('cuda')
    opt = torch.optim.Adam(models_cl.parameters(), lr=5e-4)
    min_loss = 1000
    for e in range(epoch):
        total_loss = 0
        star_time = time.time()
        for a, b, c in train_loader:
            loss1 = models_cl(a, b, c)
            loss1.backward()
            opt.step()
            opt.zero_grad()
            models_cl.update_moving_average()
            total_loss += loss1.detach().cpu()
        end_time = time.time()
        running_time = end_time - star_time
        if total_loss < min_loss:
            min_loss = total_loss
            # torch.save(models.state_dict(), r'D:\sss\contrast\BYOL\SaveModel\RC_GAF_IR1.pth')
        print(f"epoch{e} ", "cl_loss:", total_loss.item(), 'time cost: %.1f' % running_time)
    return models_cl


def hyperParameters():
    """
    所有参数
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda', help='device')

    parser.add_argument('--drop_ratio', type=float, default=0.1, help='#MLP中的两次dropout')

    parser.add_argument('--lr', type=float, default=4e-3, help='Initial learning rate.')

    parser.add_argument('--epoch', type=int, default=800, help='Number of epochs to train.')

    parser.add_argument('--batch_size', type=int, default=400, help='Number of loading.')

    parser.add_argument('--seed', type=int, default=256, help='Random seed.')

    parser.add_argument('--model_name', type=str, default="lstm", help='model Lstm, RNN or GRU.')

    parser.add_argument('--model', type=object, default=None, help='model Lstm, RNN or GRU.')

    parser.add_argument('--num_layers', type=int, default=1, help="模型层数")

    parser.add_argument('--factors', type=int, default=10, help="数据集因子数.")

    parser.add_argument('--data_path', type=str, default="../data/Anyuan_landslides_10factors_label.csv",
                        help="数据集所在路径.")

    parser.add_argument('--test_path', type=str, default=None, help="测试集集所在路径.若为空值，自动分出三个数据集，若为''，测试集与验证集相同")

    parser.add_argument('--valid_path', type=str, default=None, help="验证集路径")

    parser.add_argument('--dir_name', type=str, default=None, help="日志文件存放路径")

    parser.add_argument('--whole_data_path', type=str, default=None,
                        help="用于标准化的数据集所在路径.")

    parser.add_argument('--channels', type=int, default=20, help="CNN 的channels")

    parser.add_argument('--test_scalar', type=float, default=0.7, help="测试集比例")

    parser.add_argument('--target_recall', type=float, default=0.9, help="预测性能评估时固定的recall值")

    parser.add_argument('--logs_path', type=str, default='', help="excel_name.")

    parser.add_argument('--feature_names', type=list, default=[], help="factors_name")

    parser.add_argument('--target', type=object, default=None, help="factors_name")
    # parser.add_argument('--times', type=str, default=times, help="times.")

    parser.add_argument('--unsqueeze', type=bool, default=True, help="是否需要对输入进行扩维")
    opts = parser.parse_args()
    return opts


def pre_train(models_cl, args):
    # print(args.model)
    # 加载对比学习权重
    CL_Load = True
    if CL_Load:
        params_dict = args.model.state_dict()
        pre_dict = models_cl.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in params_dict}
        params_dict.update(pre_dict)
        # print([k for k, v in pre_dict.items()])
        args.model.load_state_dict(params_dict)
        del_list = [
            "online_encoder.cnn2d.Residual0.conv1.weight",
            "online_encoder.cnn2d.Residual0.bn1.weight",
            "online_encoder.cnn2d.Residual0.bn1.bias",
            "online_encoder.cnn2d.Residual0.conv2.weight",
            "online_encoder.cnn2d.Residual0.bn2.weight",
            "online_encoder.cnn2d.Residual0.bn2.bias",
            "online_encoder.cnn2d.Residual1.conv1.weight",
            "online_encoder.cnn2d.Residual1.bn1.weight",
            "online_encoder.cnn2d.Residual1.bn1.bias",
            "online_encoder.cnn2d.Residual1.conv2.weight",
            "online_encoder.cnn2d.Residual1.bn2.weight",
            "online_encoder.cnn2d.Residual1.bn2.bias",
            "online_encoder.cnn2d.Residual1.downsample.0.weight",
            "online_encoder.cnn2d.Residual1.downsample.1.weight",
            "online_encoder.cnn2d.Residual1.downsample.1.bias",
            "online_encoder.head.0.weight",
            "online_encoder.head.0.bias",
            "online_encoder.head.1.weight",
            "online_encoder.head.1.bias",
            "online_encoder.head.3.weight",
            "online_encoder.head.3.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            "mlp.fc4.weight",
            "mlp.fc4.bias",
            "mlp.bn.weight",
            "mlp.bn.bias",
        ]
        for name, params in args.model.named_parameters():
            if name not in del_list:
                params.requires_grad = False
        # for name, params in args.model.named_parameters():
        #     print(name, params.requires_grad)

    # 开始训练
    entity = Train(args)
    entity.train()
    return entity.pre_best_auc_model


def fine_tune(pre_train_model, args):
    # 开始微调
    # 加载预训练模型
    pre_train = True
    freeze = True
    if pre_train:
        args.model = pre_train_model
        # 冻结部分参数
        if freeze:
            # para_list = "head.0.weight head.0.bias head.1.weight head.1.bias head.3.weight head.3.bias".split()
            para_list = "online_encoder.head.0.weight online_encoder.head.0.bias online_encoder.head.1.weight online_encoder.head.1.bias online_encoder.head.3.weight online_encoder.head.3.bias".split()
            # para_list = "fc.0.weight fc.0.bias fc.2.weight fc.2.bias".split()
            for name, params in args.model.named_parameters():
                if name not in para_list:
                    # print(name)
                    params.requires_grad = False

        # for name, params in args.model.named_parameters():
        #     print(name, params.requires_grad)

    # 开始训练
    entity = Train(args)

    # 训练前测试"before_train_target.csv"
    entity.pre_results(save_path=None, test_model=args.model)
    print(entity.target.T)
    target_total = entity.target.T

    entity.train()

    # 根据最佳acc
    entity.pre_results(save_path=None, test_model=entity.pre_best_acc_model)
    target_total = pd.concat([target_total, entity.target.T], axis=1)

    # 根据最佳auc
    entity.pre_results(save_path=None, test_model=entity.pre_best_auc_model, results_path='pre_test_results.csv')
    target_total = pd.concat([target_total, entity.target.T], axis=1)

    target_total.columns = ["pre_train", "fine-best_acc", "fine-best_auc"]
    # print(target_total)

    target_total.T.to_csv(os.path.join(entity.logs_path, "targets.csv"))
    file_name = f"acc{round(target_total['fine-best_acc']['acc'], 4)}_auc{round(target_total['fine-best_auc']['auc'], 4)}.txt"
    with open(file=os.path.join(entity.logs_path, file_name), mode='w') as f:
        print(file_name)
        f.write(f"CNN drop_out {inner_d}\n")

    return target_total


# 总管
if __name__ == '__main__':
    # 简化数据路径：直接使用LSMFormer瑞昌市目录下的数据
    BASE_DIR_MAIN = Path(__file__).resolve().parent.parent.parent  # E:\yy_learn\Landslide\LSMFormer瑞昌市
    DATA_DIR = BASE_DIR_MAIN / "数据" / "瑞昌市数据"  # 已经是LSMFormer瑞昌市目录，不需要再加
    
    # 设置数据文件路径
    train_path = str(DATA_DIR / "rc_freqTrain3124.csv")
    test_path = str(DATA_DIR / "rc_freqTest1340.csv")
    # neg_data_path: 负样本数据路径，用于生成SMOTE数据（包含大量非滑坡样本）
    neg_data_path = str(DATA_DIR / "rc_freqNeg1600000.csv")
    
    # 读取数据
    train_data = pd.read_csv(train_path, index_col=0)
    neg_data = pd.read_csv(neg_data_path, index_col=0) if os.path.exists(neg_data_path) else pd.DataFrame()

    args = hyperParameters()
    # 数据不平衡比率
    rate = 20

    cl_epoch = 500#原10
    args.epoch = 500#原10
    # 阶段2（预训练）的epoch：根据最佳结果目录名CL_new_BY_RC_freq20231227-14-52_epoch500lr0.001_indr0.3
    # 预训练应该使用500个epoch，而不是10
    #args.epoch = 500   修复：从10改为500，与最佳结果配置一致
    ft_epoch = 100  # 100，与最佳结果配置一致

    cl_batch_size = 500
    args.batch_size = 500
    # 阶段2（预训练）的batch_size：根据最佳结果targets.csv，预训练阶段batch_size=100
    #args.batch_size = 100   修复：从500改为100，与最佳结果配置一致
    ft_batch_size = 100

    args.lr = 1e-3
    ft_lr = 1e-4

    inner_d = 0.3
    args.drop_ratio = 0.5
    args.target_recall = 1
    args.unsqueeze = False
    args.factors = 23
    args.model = new_BY(args.factors, args.drop_ratio, drop_cnn=inner_d).to(args.device)

    # 本次运行结果列表
    args.model_name = str(args.model).split('(')[0]
    args.target = pd.DataFrame([{"model": args.model_name,
                                 "rate": rate,
                                 "indr": inner_d,
                                 "drop_ratio": args.drop_ratio,
                                 "epoch": args.epoch,
                                 "batch_size": args.batch_size,
                                 "lr": args.lr}])
    # 生成日志文件夹
    args.dir_name = args.model_name + f"_rate{rate}_" + get_dir_name(epoch=args.epoch, lr=args.lr)
    # 文件名过长tensorboard会无法生成日志文件，报FileNotFound错
    print(args.dir_name)
    args.dir_name = mkdir(dir_name=args.dir_name)
    save_path = 'cache_data'
    # 按比率获取原始数据和SMOTE过采样数据
    smote_path, ori_data = get_balance_SMOTE_data(train_data, neg_data, rate, root_path=save_path)
    print('开始对比学习'.center(50, '*'))
    model_cl = start_cl(ori_data, save_path, epoch=cl_epoch, bat=cl_batch_size)
    torch.save(model_cl.state_dict(), os.path.join(args.dir_name, f'cl_last{cl_epoch}.pth'))
    # 预训练
    print('开始预训练'.center(50, '*'))
    landslides_data = "RC_smote"
    if landslides_data == "RC_smote":
        args.data_path = smote_path
        args.test_path = test_path
        args.feature_names = pd.read_csv(args.data_path, index_col=0).columns
        args.factors = 23
        args.test_scalar = 0.9  # 小于1表示按比例划分训练和验证
    pre_train_model = pre_train(model_cl, args)
    torch.save(pre_train_model.state_dict(), os.path.join(args.dir_name, f'preTrain{args.epoch}.pth'))

    # 为了微调阶段需要，设置SMOTE数据路径（用于标准化）
    # 使用阶段1生成的SMOTE数据（smote_path已在阶段1中生成）
    # 微调
    print('开始微调'.center(50, '*'))
    landslides_data = "RC_freq"
    if landslides_data == "RC_freq":
        args.data_path = train_path
        args.test_path = test_path
        args.whole_data_path = smote_path
        args.feature_names = pd.read_csv(args.data_path, index_col=0).columns
        args.factors = 23
        args.test_scalar = 1  # 等于1, 且验证集未赋值表示将测试集当成验证集去训练
        args.epoch = ft_epoch
        args.batch_size = ft_batch_size
        args.lr = ft_lr
    target = fine_tune(pre_train_model, args)
    print(target)
    print('开始绘制ROC曲线'.center(50, '*'))
    test_results = pd.read_csv(os.path.join(args.dir_name, 'pre_test_results.csv'), header=None)
    test_data = pd.read_csv(test_path, index_col=0)
    test_labels = test_data.iloc[:, -1].values
    y_pred_proba = test_results.iloc[:, 1].values
    fpr, tpr, _ = roc_curve(test_labels, y_pred_proba)
    auc_value = ROC_AUC(y_pred_proba, test_labels)
    draw_ROC_save(fpr, tpr, auc_value, args.dir_name)

    print('开始绘制图C：不同不平衡比率下的性能指标对比图'.center(50, '*'))
    logs_dir = './logs'
    rate_list = [10, 20, 50, 65, 100, 200, 300, 500, 1000]  # 不平衡比率列表
    
    # 收集不同rate的实验结果
    metrics_data = collect_rate_results(
        logs_dir=logs_dir,
        rate_list=rate_list,
        stage='fine-best_auc'  # 使用fine-tuning阶段的最佳AUC结果
    )

    if not metrics_data.empty:
        # 创建输出目录
        output_dir = os.path.join(logs_dir, 'imbalance_ratio_analysis')
        os.makedirs(output_dir, exist_ok=True)

        draw_imbalance_ratio_performance(
            metrics_data=metrics_data,
            save_path=output_dir,
            title='Performance Metrics Across Imbalance Ratios'
        )
        print(f"图C已保存到: {os.path.join(output_dir, 'imbalance_ratio_performance.svg')}")
    else:
        print("警告: 未收集到足够的数据，无法绘制图C。请确保logs目录下有多个不同rate的实验结果。")

# 发出警报音提示运行结束
# LingYa()
