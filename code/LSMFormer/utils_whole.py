# -*- encoding=utf-8 -*-
import os.path
import torch
import time
from torch import nn
import numpy as np
import pandas as pd
import plotly as py
from plotly.tools import FigureFactory as FF
import math

pd.set_option('display.float_format', lambda x: '%.4f' % x)
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno  # 可视化工具， pip install missingno

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import copy
from torch.utils.tensorboard import SummaryWriter
from thop import clever_format, profile
from torchsummary import summary


def mkdir(dir_name):
    """
    创建Logs文件夹，并以运行时间（年月日）+batch_size + epoch + Lr命名
    """
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    # if not os.path.exists('../model'):
    #     os.mkdir('../model')
    if len(dir_name.split('/')) > 1:
        mode_name = os.path.join("./logs", dir_name.split('/')[0])
        if not os.path.exists(mode_name):
            os.mkdir(mode_name)
    logs_name = os.path.join("./logs", dir_name)
    if not os.path.exists(logs_name):
        os.mkdir(logs_name)

    return logs_name  # , model_name


def get_dir_name(epoch, lr):
    """
    :return: 生成一个以参数和时间戳命名的文件夹名,最终存放在log里
    """
    epoch = "epoch" + str(epoch)
    # lr = "lr" + str(lr)
    _time = str(time.strftime("%Y%m%d-%H-%M", time.localtime()))  # 获取当前epoch的运行时刻
    dir_name = r'_{}'.format(epoch)
    dir_name = _time + dir_name

    return dir_name


def ROC_AUC(y_pred_proba, y, draw=False):
    fpr, tpr, thesholds_ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)  # 曲线下面积

    # 绘制 ROC曲线
    if draw:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.0])
        plt.ylim([-0.1, 1.01])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

    return roc_auc

def kappa_index(tn, fp, fn, tp, acc):
    p0 = acc
    m = tp + fp + fn + tn
    pe1 = (tp + fn) * (tp + fp)
    pe2 = (fn + tn) * (fp + tn)
    pe = (pe1 + pe2) / math.pow(m, 2)
    kappa = (p0 - pe) / (1 - pe)
    return kappa

def get_output_results(y_pred_proba, y, t=0.5):
    y_pred = y_pred_proba.iloc[:, 1] > t
    cnf_matrix = confusion_matrix(y, y_pred)
    recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    acc = (cnf_matrix[0, 0] + cnf_matrix[1, 1]) / (cnf_matrix.sum())
    auc = ROC_AUC(y_pred_proba.iloc[:, 1], y)
    tn, fp, fn, tp = cnf_matrix[0, 0], cnf_matrix[0, 1], cnf_matrix[1, 0], cnf_matrix[1, 1]
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    ka = kappa_index(tn, fp, fn, tp, acc)
    return acc, auc, recall, precision, ka,  f1, tn, fp, fn, tp


def get_recall(y_pred_proba, y, t=0.5):
    y_pred = y_pred_proba.iloc[:, 1] > t
    cnf_matrix = confusion_matrix(y, y_pred)
    return cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])


def get_res_by_fixed_recall(y_pred_proba, y, recall=0.9, mistake=1e-3):
    y = y.reset_index(drop=True)
    if type(y) == pd.DataFrame:
        y = y.iloc[:, 0]
    y_pos = y_pred_proba[y == 1].sort_values(by=1).iloc[:, 1]
    y_pos = y_pos.reset_index(drop=True)
    fixed_t = y_pos[int(y_pos.shape[0] * (1 - recall))]

    res = get_df_results(y_pred_proba, y, fixed_t)
    # 检索
    first_flag = 1 if res['recall'][0] > recall else -1
    sum = first_flag
    min_mistake = res['recall'][0] - recall
    min_mis_res = res
    while abs(res['recall'][0] - recall) >= mistake:
        flag = 1 if res['recall'][0] > recall else -1
        if flag != first_flag:
            # print(f"Mistake can't solve! :{min_mistake}")
            break
        sum += flag
        index_sum = int(y_pos.shape[0] * (1 - recall)) + sum
        if index_sum < 0 or index_sum >= y_pos.shape[0]:
            break
        fixed_t = y_pos[index_sum]
        res = get_df_results(y_pred_proba, y, fixed_t)
        if abs(min_mistake) > abs(res['recall'][0] - recall):
            min_mistake = res['recall'][0] - recall
            min_mis_res = res
    # print("fixed_t:", fixed_t)
    min_mis_res['fixed_t'] = fixed_t
    return min_mis_res


def get_model_results(train_data, test_data, model=RandomForestClassifier(n_estimators=100)):
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    model.fit(X_train, y_train)

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    y_pred_proba = model.predict_proba(X_test)

    res_df = pd.DataFrame(list(get_output_results(y_pred_proba, y_test))).T
    res_df.columns = ['acc', 'auc', 'recall', 'precision', 'kappa', 'f1', 'tn', 'fp', 'fn', 'tp']
    # display(res_df)
    return res_df

def softmax(z):
    # 计算softmax函数
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def get_df_results(y_pred_proba, y, t=0.5):
    # print(y_pred_proba)
    y_pred_proba = y_pred_proba.apply(softmax, axis=1)
    # print(y_pred_proba)
    res_df = pd.DataFrame(list(get_output_results(y_pred_proba, y, t=t))).T
    res_df.columns = ['acc', 'auc', 'recall', 'precision', 'kappa', 'f1', 'tn', 'fp', 'fn', 'tp']
    # print(res_df)
    return res_df


class MyDataset(Dataset):
    def __init__(self, data, target=None):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        _dict = {'data': torch.FloatTensor(data)}
        if self.target is not None:
            target = self.target[idx].item()
            _dict.update({'target': torch.tensor(target, dtype=torch.long)})
        return _dict

def get_max_min_df(df):
    min_max_df = pd.concat([pd.DataFrame(df.min(axis=0)), pd.DataFrame(df.max(axis=0))], axis=1, ignore_index=True).T
    # display(min_max_df)
    print(min_max_df)
    return min_max_df

def restrict_stander(data, whole_path):
    min_max_df = get_max_min_df(pd.read_csv(whole_path, index_col=0))
    data = data.iloc[:]
    features_names = data.columns
    for i in range(data.shape[1]):
        # print(f'{features_names[i]}', mt_df[f'{features_names[i]}'])
        max_min = (min_max_df[f'{features_names[i]}'][1] - min_max_df[f'{features_names[i]}'][0])
        data[f'{features_names[i]}'] = data[f'{features_names[i]}'].apply(
            lambda x: (x - min_max_df[f'{features_names[i]}'][0]) / max_min)
    return data

def fit_transform_train(train_data, path=None):
    min_max_df = pd.DataFrame([train_data.min(axis=0), train_data.max(axis=0)])
    if path is not None:
        min_max_df.to_csv(os.path.join(path, 'mean_std.csv'), header=True, index=False)
    scaled_train = train_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0, axis=0)
    return scaled_train, min_max_df

def transform_test(test_data, min_max_df):
    scaled_test = test_data.copy()
    for col in test_data.columns:
        min_val = min_max_df[col][0]
        max_val = min_max_df[col][1]
        if max_val - min_val != 0:
            scaled_test[col] = (test_data[col] - min_val) / (max_val - min_val)
        else:
            scaled_test[col] = 0
    return scaled_test

def stander_data(data, path=None, whole_path=None):
    """
        数据集标准化,data不带标签，path用来存标准化时的均值方差
    """
    if whole_path is not None:
        return restrict_stander(data, whole_path)
    if path is not None:
        mean_std = pd.concat([pd.DataFrame(data.min(axis=0)), pd.DataFrame(data.max(axis=0))], axis=1).T
        # mean_std = pd.concat([pd.DataFrame(data1.mean(axis=0)), pd.DataFrame(data1.std(axis=0))], axis=1).T
        mean_std.to_csv(os.path.join(path, 'mean_std.csv'), header=True, index=False)
    data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    return data


def whole_stander_data(data, mean_std):
    '''
    全区数据标准化，使用训练集的最大最小值
    '''
    data1 = data.iloc[:, :21]
    data2 = data.iloc[:, 21:]
    features_names = data1.columns
    data1 = data1
    for i in range(mean_std.shape[1]):
        max_min = (mean_std[f'{features_names[i]}'][1] - mean_std[f'{features_names[i]}'][0])
        data1[f'{features_names[i]}'] = data1[f'{features_names[i]}'].apply(
            lambda x: (x - mean_std[f'{features_names[i]}'][0]) / max_min)
    # lambda x: (x - mean_std[f'{features_names[i]}'][0]) /mean_std[f'{features_names[i]}'][1])
    data = pd.concat([data1, data2], axis=1)
    return data

def get_dataFrame(data_path, test_path, log_path, s=0.7, valid_path=None, whole_path=None):
    np.random.seed(256)
    df = pd.read_csv(data_path).sample(frac=1)
    if df.columns[0] == 'Unnamed: 0':
        df = df.iloc[:, 1:]
    len1 = int(s * df.shape[0])

    if s == 1 and valid_path is None:
        df_test = pd.read_csv(test_path)
        if df_test.columns[0] == 'Unnamed: 0':
            df_test = df_test.iloc[:, 1:]
        train_label, valid_label, test_label = df.iloc[:, -1], df_test.iloc[:, -1], df_test.iloc[:, -1]
        train_inputs, min_max_df = fit_transform_train(df.iloc[:, :-1], path=log_path)
        test_inputs = transform_test(df_test.iloc[:, :-1], min_max_df)
        return train_inputs, test_inputs, test_inputs, train_label, valid_label, test_label

    elif s == 1 and valid_path is not None:
        df_test = pd.read_csv(test_path)
        if df_test.columns[0] == 'Unnamed: 0':
            df_test = df_test.iloc[:, 1:]
        df_valid = pd.read_csv(valid_path)
        if df_valid.columns[0] == 'Unnamed: 0':
            df_valid = df_valid.iloc[:, 1:]
        train_label, valid_label, test_label = df.iloc[:, -1], df_valid.iloc[:, -1], df_test.iloc[:, -1]
        train_inputs, min_max_df = fit_transform_train(df.iloc[:, :-1], path=log_path)
        valid_inputs = transform_test(df_valid.iloc[:, :-1], min_max_df)
        test_inputs = transform_test(df_test.iloc[:, :-1], min_max_df)
        return train_inputs, valid_inputs, test_inputs, train_label, valid_label, test_label

    elif test_path is None:
        len2 = int(s * len1)
        train_label, valid_label, test_label = df.iloc[:len2, -1], df.iloc[len2:len1, -1], df.iloc[len1:, -1]
        train_inputs, min_max_df = fit_transform_train(df.iloc[:len2, :-1], path=log_path)
        valid_inputs = transform_test(df.iloc[len2:len1, :-1], min_max_df)
        test_inputs = transform_test(df.iloc[len1:, :-1], min_max_df)
        return train_inputs, valid_inputs, test_inputs, train_label, valid_label, test_label

    elif test_path == '':
        train_label, valid_label = df.iloc[:len1, -1], df.iloc[len1:, -1]
        train_inputs, min_max_df = fit_transform_train(df.iloc[:len1, :-1], path=log_path)
        valid_inputs = transform_test(df.iloc[len1:, :-1], min_max_df)
        return train_inputs, valid_inputs, valid_inputs, train_label, valid_label, valid_label

    else:
        df_test = pd.read_csv(test_path)
        if df_test.columns[0] == 'Unnamed: 0':
            df_test = df_test.iloc[:, 1:]
        len2 = df.shape[0]
        train_label, valid_label, test_label = df.iloc[:len1, -1], df.iloc[len1:, -1], df_test.iloc[:, -1]
        train_inputs, min_max_df = fit_transform_train(df.iloc[:len1, :-1], path=log_path)
        valid_inputs = transform_test(df.iloc[len1:len2, :-1], min_max_df)
        test_inputs = transform_test(df_test.iloc[:, :-1], min_max_df)
        return train_inputs, valid_inputs, test_inputs, train_label, valid_label, test_label

def with_valid_dataloader(data_path: str, log_path=None, batch_size=200, test_path=None, s=0.7, valid_path=None, whole_path=None):
    """
        获取数据集与标签，带有验证集,test_path未赋值时自动划分验证集
    """
    train_inputs, valid_features, test_inputs, train_labels, valid_labels, test_labels = \
        get_dataFrame(data_path, test_path, log_path, s, valid_path, whole_path=whole_path)
    train_len, test_len, valid_len = len(train_labels), len(test_labels), len(valid_labels)

    train_factors = torch.FloatTensor(train_inputs.to_numpy())
    valid_factors = torch.FloatTensor(valid_features.to_numpy())
    test_factors = torch.FloatTensor(test_inputs.to_numpy())
    train_labels = torch.FloatTensor(train_labels.to_numpy())
    valid_labels = torch.FloatTensor(valid_labels.to_numpy())
    test_labels = torch.FloatTensor(test_labels.to_numpy())

    # train_factors = torch.FloatTensor(train_inputs.iloc[:500, :].to_numpy())
    # valid_factors = torch.FloatTensor(valid_features.to_numpy())
    # test_factors = torch.FloatTensor(test_inputs.to_numpy())
    # train_labels = torch.FloatTensor(train_labels.iloc[:500].to_numpy())
    # valid_labels = torch.FloatTensor(valid_labels.to_numpy())
    # test_labels = torch.FloatTensor(test_labels.to_numpy())

    train_dataset = MyDataset(data=train_factors, target=train_labels)
    valid_dataset = MyDataset(data=valid_factors, target=valid_labels)
    test_dataset = MyDataset(data=test_factors, target=test_labels)
    print(train_labels.shape)
    print(batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("数据集加载成功！")
    return train_dataloader, test_dataloader, valid_dataloader, train_len, test_len, valid_len


def draw_ROC_save(fpr, tpr, auc, path):
    plt.figure(figsize=(8, 7), dpi=80, facecolor='w')  # dpi:每英寸长度的像素点数；facecolor 背景颜色
    plt.xlim((-0.01, 1.02))  # x,y 轴刻度的范围
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))  # 绘制刻度
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot(fpr, tpr, 'r-', lw=2, label='AUC=%.4f' % auc)  # 绘制AUC 曲线
    plt.legend(loc='lower right')  # 设置显示标签的位置
    plt.xlabel('False Positive Rate', fontsize=14)  # 绘制x,y 坐标轴对应的标签
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(visible=True, ls=':')  # b=True改为visible=True
    plt.title(u'TabTransformer ROC curve And  AUC', fontsize=18)  # 打印标题
    roc_image_path = os.path.join(path, "ROC.svg")
    plt.savefig(roc_image_path, format="svg")
    plt.close('all')
    # plt.show()


class NMTCritierion(nn.Module):
    """
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # print('label', labels[:10])
        # conduct label_smoothing module
        gtruth = labels.view(-1)
        # print('gtruth', gtruth[:10])
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            # print('one_hot', one_hot[:10])
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
            # print('gtruth',  gtruth[:10])
        loss = self.criterion(scores, gtruth)
        return loss


def whole_region_dataloader(path: str, batch_size, logs):
    original_whole_region_df = pd.read_csv(path)  # 全区数据有表头，无标签 .sample(frac=1)
    # print('原始的全区数据\n', original_whole_region_df[:10])
    mean_std = pd.read_csv(os.path.join(logs, 'mean_std.csv'))
    standard_whole_region_df = whole_stander_data(original_whole_region_df, mean_std)
    # print('标准化后的全区数据\n', standard_whole_region_df[:10])
    standard_whole_region_dataset = standard_whole_region_df.to_numpy()  # df->>numpy->tensor
    standard_whole_region_dataset = MyDataset(data=standard_whole_region_dataset, target=None)
    standard_whole_region_dataset = DataLoader(standard_whole_region_dataset, batch_size=batch_size, shuffle=False)
    return standard_whole_region_dataset, original_whole_region_df, standard_whole_region_df


def display_df(df):
    pyplt = py.offline.plot
    table = FF.create_table(df, index=True, index_title='Date')
    pyplt(table, filename=r'table_pandas.html', show_link=False)


# === From utils1_whole.py: functions for imbalance ratio performance plot (图C) ===
import glob
from matplotlib.gridspec import GridSpec


def collect_rate_results(logs_dir, rate_list=[10, 20, 50, 65, 100, 200, 300, 500, 1000], stage='fine-best_auc'):
    """
    从多个不同rate的实验目录中收集性能指标（用于绘制图C）

    参数:
        logs_dir: 日志根目录，例如 './logs'
        rate_list: 不平衡比率列表
        stage: 要收集的阶段，'fine-best_auc' 或 'fine-best_acc'

    返回:
        DataFrame，包含不同rate下的性能指标
    """
    results_list = []

    for rate in rate_list:
        # 查找包含该rate的目录
        pattern = f"*rate{rate}_*"
        matching_dirs = glob.glob(os.path.join(logs_dir, pattern))

        if not matching_dirs:
            print(f"警告: 未找到rate={rate}的实验目录")
            continue

        # 使用最新的目录（按修改时间排序）
        matching_dirs.sort(key=os.path.getmtime, reverse=True)
        target_dir = matching_dirs[0]

        targets_path = os.path.join(target_dir, 'targets.csv')
        if not os.path.exists(targets_path):
            print(f"警告: {targets_path} 不存在")
            continue

        try:
            targets_df = pd.read_csv(targets_path, index_col=0)
            # 筛选指定阶段的结果
            stage_df = targets_df[targets_df.index.str.contains(stage, na=False)]

            if stage_df.empty:
                print(f"警告: {targets_path} 中未找到阶段 '{stage}'")
                continue

            # 取第一行（通常只有一个结果）
            result = stage_df.iloc[0].to_dict()
            result['imbalance_ratio'] = rate
            results_list.append(result)
        except Exception as e:
            print(f"错误: 读取 {targets_path} 时出错: {e}")
            continue

    if not results_list:
        print("错误: 未收集到任何结果")
        return pd.DataFrame()

    # 转换为DataFrame
    results_df = pd.DataFrame(results_list)

    # 确保列名正确
    metric_columns = ['auc', 'acc', 'f1', 'recall', 'precision', 'kappa']
    available_columns = [col for col in metric_columns if col in results_df.columns]

    # 重命名列以匹配图C的要求
    column_mapping = {
        'auc': 'AUC',
        'acc': 'ACC',
        'f1': 'F1',
        'recall': 'Recall',
        'precision': 'Precision',
        'kappa': 'Kappa'
    }

    for col in available_columns:
        if col in results_df.columns:
            results_df = results_df.rename(columns={col: column_mapping[col]})

    return results_df


def draw_imbalance_ratio_performance(metrics_data, save_path, title='Performance Metrics Across Imbalance Ratios'):
    """
    绘制图C：不同不平衡比率下的性能指标对比图（使用原生matplotlib）

    参数:
        metrics_data: DataFrame，包含以下列：
            - 'imbalance_ratio': 不平衡比率（10, 20, 50, 65, 100, 200, 300, 500, 1000）
            - 'AUC': AUC值
            - 'ACC': 准确率
            - 'F1': F1分数
            - 'Recall': 召回率
            - 'Precision': 精确率
            - 'Kappa': Kappa系数
        save_path: 保存路径
        title: 图表标题
    """
    # 创建图形，包含主图和子图
    fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='w')

    # 使用GridSpec创建布局：主图（线图）+ 子图（柱状图）
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # ========== 主图：线图 ==========
    ax_main = fig.add_subplot(gs[0, :])

    imbalance_ratios = metrics_data['imbalance_ratio'].values
    metrics = ['AUC', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa']
    colors = ['b', 'g', 'r', 'c', 'm', 'pink']
    linestyles = ['-', '-', '-', '-', '-', '--']

    for metric, color, ls in zip(metrics, colors, linestyles):
        if metric in metrics_data.columns:
            values = metrics_data[metric].values
            ax_main.plot(imbalance_ratios, values, color=color, linestyle=ls,
                         linewidth=2, marker='o', markersize=4, label=metric)

    ax_main.set_xlabel('Imbalance Ratios', fontsize=14)
    ax_main.set_ylabel('Performance Metrics', fontsize=14)
    ax_main.set_ylim(0.75, 1.00)
    ax_main.set_xscale('log')  # X轴使用对数刻度
    ax_main.set_xticks(imbalance_ratios)
    ax_main.set_xticklabels(imbalance_ratios)
    ax_main.grid(visible=True, ls=':', alpha=0.5)
    ax_main.legend(loc='best', fontsize=10)
    ax_main.set_title(title, fontsize=16)

    # ========== 子图：柱状图（平均值）==========
    ax_bar = fig.add_subplot(gs[1, 1])

    # 计算平均值和标准差
    mean_values = []
    std_values = []
    metric_names = []
    for metric in metrics:
        if metric in metrics_data.columns:
            mean_values.append(metrics_data[metric].mean())
            std_values.append(metrics_data[metric].std())
            metric_names.append(metric)

    x_pos = np.arange(len(metric_names))
    bars = ax_bar.bar(x_pos, mean_values, yerr=std_values, capsize=5,
                      color=colors[:len(metric_names)], alpha=0.7, edgecolor='black')

    ax_bar.set_ylabel('Mean Value', fontsize=12)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=10)
    ax_bar.set_ylim(0.75, 1.00)
    ax_bar.grid(visible=True, ls=':', alpha=0.5, axis='y')

    # 在柱状图上添加数值标签
    for bar, mean_val in zip(bars, mean_values):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

    # 保存图像
    output_path = os.path.join(save_path, "imbalance_ratio_performance.svg")
    plt.savefig(output_path, format="svg", bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f"不平衡比率性能对比图已保存: {output_path}")
