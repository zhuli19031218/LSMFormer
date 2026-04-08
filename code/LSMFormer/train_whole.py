# -*- encoding:utf-8 -*-
import argparse
import os
import pandas as pd
import torch.cuda
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import copy
from utils_whole import *
from torch.utils.tensorboard import SummaryWriter
from thop import clever_format, profile
from torchsummary import summary
from torchsummary import summary


class Train(object):
    def __init__(self, args):
        super(Train, self).__init__()
        self.args = args
        # 各种指标
        self.pre_best_auc_model, self.pre_best_acc_model = None, None
        self.best_precision_model, self.second_precision_model = None, None
        self.pre_best_auc, self.pre_best_auc_acc, self.pre_best_acc, self.pre_best_acc_auc = \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32), \
            torch.tensor([0.], dtype=torch.float32)
        # 文件夹名：
        # self.dir_name = args.dir_name
        # 创建Logs文件夹，并以运行时间（年月日）+batch_size + epoch + Lr命名
        self.logs_path = args.dir_name  # mkdir(dir_name=self.dir_name)

        # 获取数据集与标签
        # print(args.batch_size)
        self.train_dataset, self.test_dataset, self.valid_dataset, self.train_len, self.test_len, self.valid_len = \
            with_valid_dataloader(data_path=args.data_path, batch_size=args.batch_size, log_path=self.logs_path,
                                  test_path=args.test_path, s=args.test_scalar, valid_path=args.valid_path, whole_path=args.whole_data_path)
        print(f"训练集：{self.train_len}\n验证集：{self.valid_len}\n测试集：{self.test_len}")

        # 模型
        self.pre_model = args.model
        # 选出要更新权重的参数
        params = [p for p in self.pre_model.parameters() if p.requires_grad]
        # 定义优化器和损失函数
        self.pre_optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)

        self.pre_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.pre_optimizer,
                                                                           T_max=args.epoch + 2000,
                                                                           eta_min=0, last_epoch=-1)

        self.pre_loss_fn = nn.CrossEntropyLoss().to(self.args.device)
        # 存放各项指标
        self.target = args.target
        # tensorBoard 记录
        # 将相对路径转换为绝对路径，避免TensorFlow gfile API处理相对路径时的错误
        abs_logs_path = os.path.abspath(self.logs_path)
        # 确保目录存在
        os.makedirs(abs_logs_path, exist_ok=True)
        self.logs_path = abs_logs_path  # 更新为绝对路径
        self.write = SummaryWriter(log_dir=self.logs_path)
        self.feature_names = args.feature_names

    def train(self):
        fixed_best_res = None
        best_precision = 0
        best_precision_epoch = 0
        best_auc = 0
        best_auc_epoch = 0
        pre_best_acc_epoch = 0
        fixed_t = 0.5
        # 计算时间
        pre_start_time = time.time()
        for i in range(self.args.epoch):
            # 存放训练和验证的正确率
            pre_total_train_loss, pre_total_train_acc = torch.tensor([0.], dtype=torch.float32), \
                                                        torch.tensor([0.], dtype=torch.float32)
            pre_total_test_loss, pre_total_test_acc = torch.tensor([0.], dtype=torch.float32), \
                                                      torch.tensor([0.], dtype=torch.float32)
            pre_total_test_results, pre_total_test_labels = torch.Tensor(), torch.Tensor()  # cpu的张量
            self.pre_model.train()
            # 训练集
            for pre_train_data in self.train_dataset:
                if self.args.unsqueeze:
                    pre_sub_train_output = self.pre_model(
                        torch.unsqueeze(input=pre_train_data['data'], dim=0).to(self.args.device))  #
                else:
                    pre_sub_train_output = self.pre_model(pre_train_data['data'].to(self.args.device))
                pre_sub_train_loss = self.pre_loss_fn(pre_sub_train_output,
                                                      pre_train_data['target'].to(self.args.device))
                # pre_sub_train_loss： GPU张量、带梯度
                pre_sub_train_loss.backward()  # 反向传播，计算当前梯度
                self.pre_optimizer.step()  # 根据梯度，更新模型权重
                self.pre_optimizer.zero_grad()
                # 累计总损失值，torch.float32
                pre_total_train_loss += pre_sub_train_loss.cpu().detach()
                # 该batch上训练集的判断正确的样本个数
                pre_sub_train_accuracy = (pre_sub_train_output.detach().argmax(axis=1) == pre_train_data['target'].to(
                    self.args.device)).sum()
                # 累计所有batch训练集上的正确率
                pre_total_train_acc += pre_sub_train_accuracy.cpu()
            # 更新学习率
            # self.pre_lr_scheduler.step()
            # tensorboard summaryWriter 记录损失值和正确率变化
            self.write.add_scalar(tag='pre_train_loss', scalar_value=pre_total_train_loss.detach().cpu(), global_step=i)
            self.write.add_scalar(tag='pre_train_acc', scalar_value=pre_total_train_acc / self.train_len,
                                  global_step=i)
            self.pre_model.eval()

            # 验证集上表现
            with torch.no_grad():
                for valid_data in self.valid_dataset:
                    # valid_output = self.pre_model(
                    #     torch.unsqueeze(input=valid_data['data'], dim=0).to(self.args.device))  # .permute(1, 0, 2)
                    if self.args.unsqueeze:
                        valid_output = self.pre_model(
                            torch.unsqueeze(input=valid_data['data'], dim=0).to(self.args.device))  #
                    else:
                        valid_output = self.pre_model(valid_data['data'].to(self.args.device))
                    pre_sub_test_loss = self.pre_loss_fn(valid_output,
                                                         valid_data['target'].to(self.args.device))

                    pre_total_test_loss += pre_sub_test_loss.cpu()
                    pre_sub_test_accuracy = (
                            valid_output.cpu().argmax(axis=1) == valid_data['target']).sum()
                    pre_total_test_acc += pre_sub_test_accuracy

                    # 用于求 AUC等指标
                    pre_total_test_labels = torch.cat(
                        [pre_total_test_labels, valid_data['target']])  # 将所有标签拼接在一起，最后是(1419,2) # cpu的张量
                    pre_total_test_results = torch.cat(
                        [pre_total_test_results, valid_output.cpu()])  # 将模型的输出结果拼接在一起，最后是(1419,2)  cpu的张量
            assert (pre_total_test_results.shape[0] == self.valid_len), '测试集长度不一致'
            self.write.add_scalar(tag='pre_test_loss', scalar_value=pre_total_test_loss.item(), global_step=i)
            self.write.add_scalar(tag='pre_test_acc',
                                  scalar_value=(pre_total_test_acc / self.valid_len).item(),
                                  global_step=i)

            pre_auc = roc_auc_score(pre_total_test_labels, pre_total_test_results[:, -1])
            # res_df = get_df_results(pd.DataFrame(pre_total_test_results), pd.DataFrame(pre_total_test_labels))
            fixed_recall = self.args.target_recall
            if self.args.target_recall <= 0 or self.args.target_recall >= 1:
                fixed_res = get_df_results(pd.DataFrame(pre_total_test_results), pd.DataFrame(pre_total_test_labels))
            else:
                fixed_res = get_res_by_fixed_recall(pd.DataFrame(pre_total_test_results),
                                                    pd.DataFrame(pre_total_test_labels), recall=fixed_recall)
                fixed_t = fixed_res['fixed_t'][0]
            recall = fixed_res['recall'][0]
            precision = fixed_res['precision'][0]
            test_auc = fixed_res['auc'][0]
            F1 = fixed_res['f1'][0]
            self.write.add_scalar(tag='test_auc', scalar_value=test_auc, global_step=i)

            # 更新最佳Precision的轮并保存模型
            if precision > best_precision:
                self.second_precision_model = self.best_precision_model
                self.best_precision_model = copy.deepcopy(self.pre_model)
                best_precision = precision
                fixed_best_res = fixed_res
                torch.save(self.best_precision_model.state_dict(),  # {:.4f}_t{:.4f}
                           os.path.join(self.logs_path, 'best_precision_recall.pth'.format(recall, fixed_t)))
                best_precision_epoch = i + 1

            # 更新最佳Acc的轮并保存模型
            if (pre_total_test_acc / self.valid_len) > self.pre_best_acc:
                self.pre_best_acc_model = copy.deepcopy(self.pre_model)
                self.pre_best_acc = pre_total_test_acc / self.valid_len
                self.pre_best_acc_auc = pre_auc
                # print(f"pre_auc{pre_auc},test_auc{test_auc}")
                torch.save(self.pre_best_acc_model.state_dict(),
                           os.path.join(self.logs_path, 'pre_best_acc_model.pth'))
                pre_best_acc_epoch = i + 1

            if test_auc > best_auc:
                self.pre_best_auc_model = copy.deepcopy(self.pre_model)
                best_auc = test_auc
                torch.save(self.pre_best_auc_model.state_dict(),
                           os.path.join(self.logs_path, 'post_best_auc_model.pth'))
                best_auc_epoch = i + 1

            # 每10轮输出一次指标
            if (i + 1) % (max(int(self.args.epoch / 50), 1)) == 0:
                print('Epoch: {:04d}'.format(i + 1),
                      'train_loss: {:.4f}'.format(pre_total_train_loss.item()),  # .item()的目的是只取张量的数值
                      'train_acc: {:.4f}'.format((pre_total_train_acc / self.train_len).item()),
                      'val_loss: {:.4f}'.format(pre_total_test_loss.item()),
                      'val_acc: {:.4f}'.format((pre_total_test_acc / self.valid_len).item()),
                      'val_auc: {:.4f}'.format(test_auc),
                      'val_recall:{:.4f}'.format(recall),
                      'val_precision:{:.4f}'.format(precision),
                      'val_f1:{:.4f}'.format(F1),
                      't:{:.4f}'.format(fixed_t),
                      'lr: {:.6f}'.format(self.pre_optimizer.param_groups[0]['lr']),
                      'time: {:.4f}s'.format(time.time() - pre_start_time))
        # 保存最后的各项指标
        self.target['best_precision_epoch'] = best_precision_epoch
        self.target['best_acc_epoch'] = pre_best_acc_epoch
        self.target['best_auc_epoch'] = best_auc_epoch
        fixed_best_res.columns = ["val_" + name for name in fixed_best_res.columns]
        self.target[fixed_best_res.columns] = fixed_best_res

    # -----------------------------------------------#
    # 基于best_acc_model获取鄱阳湖数据的输出结果
    # -----------------------------------------------#
    def pre_results(self, save_path="IDontKnow.csv", test_model=None, results_path=None):
        # print('自筛选之鄱阳湖数据'.center(100, '='))
        all_train_results, all_test_results = torch.Tensor(), torch.Tensor()
        all_test_labels = torch.Tensor()
        if test_model is not None:
            model = test_model
        else:
            model = self.best_precision_model
            results_path = "best_precision_results.csv"
        model.eval()
        with torch.no_grad():
            for data in self.test_dataset:
                if self.args.unsqueeze:
                    sub_test_output = model(
                        torch.unsqueeze(input=data['data'], dim=0).to(self.args.device)).cpu()  #
                else:
                    sub_test_output = model(data['data'].to(self.args.device)).cpu()
                all_test_labels = torch.cat([all_test_labels, data['target']])
                all_test_results = torch.cat([all_test_results, sub_test_output])
        print('test_results.shape', all_test_results.shape)  # torch.Size([1419, 2])

        all_test_results = pd.DataFrame(all_test_results.numpy())  # tensor->numpy->pd
        all_test_labels = pd.DataFrame(all_test_labels.numpy())  # tensor->numpy->pd
        if self.args.target_recall <= 0 or self.args.target_recall >= 1:
            res_df = get_df_results(all_test_results, all_test_labels)
        else:
            res_df = get_res_by_fixed_recall(all_test_results, all_test_labels, recall=self.args.target_recall)
        # self.target = pd.concat([self.target, res_df], axis=1)
        self.target[res_df.columns] = res_df
        if save_path is not None:
            self.target.T.to_csv(os.path.join(self.logs_path, save_path))
        # 只保存最佳acc的模型输出
        if results_path is not None:
            all_test_results.to_csv(os.path.join(self.logs_path, results_path), header=False, index=False)  # pd->表格

    # 获取全区输出
    def Pre_Susceptibility(self, path, csv_file, account=2941):
        model = self.pre_best_acc_model
        total_output = torch.Tensor()
        chunksize = 10000
        total_whole_features = pd.read_csv(path, iterator=True)
        mean_std = pd.read_csv(os.path.join(self.logs_path, 'mean_std.csv'))
        for i in range(account):  # 2941
            print(i)
            sub_whole_features = total_whole_features.get_chunk(size=chunksize)
            # print('标准化前', i, '\t', sub_whole_features.shape, sub_whole_features[:10])
            # sub_whole_features = stander_data(sub_whole_features, stage='pre', logs=self.logs_name)
            sub_whole_features = whole_stander_data(sub_whole_features, mean_std)
            # print('标准化后', i, '\t', sub_whole_features.shape, sub_whole_features[:10])
            sub_whole_features = torch.FloatTensor(sub_whole_features.to_numpy()).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(sub_whole_features.to(self.args.device))
                total_output = torch.cat([total_output, output.cpu()])
        print(total_output.shape)
        pre_results = pd.DataFrame(total_output[:, -1].numpy())
        pre_results.to_csv(csv_file, header=False, index=False)
