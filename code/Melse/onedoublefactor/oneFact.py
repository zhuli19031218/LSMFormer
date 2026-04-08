import pandas as pd
import torch.nn

from one_FactPlotUtils import *
import warnings as wn

wn.filterwarnings("ignore")   # 忽略警告消息

# 是否将FR映射为真实值，无需操作则False。
# 没用FR值（鄱阳湖）为True
f2v = False

# cols = ['DEM', 'Population density', 'Flow Accumulation', 'Elevation variation coefficient',
#           'Road network density',
#           'TWI', 'Fault density', 'Rock weathering index', 'Soil composition', 'Annual rainfall',
#           'Slope', 'River network density', 'Profile curvature', 'Plan curvature',
#           'Noctilucent light',
#           'NDVI', 'NDBI', 'MNDWI', 'Slope level', 'Soil type', 'Lithology']

cols = ['dem', 'rc_zongfushe', 'rc_yeguang', 'rc_turangliangdu', 'rc_turangchengfen', 'rc_slope', 'rc_rock_weathering_indexes',
        'rc_renkoumidu', 'rc_poumianqvlv', 'rc_pinmianqvlv', 'rc_nianjunjiangyu', 'rc_NDVI', 'rc_MNDWI', 'rc_luwangmidu',
        'rc_hewangmidu', 'rc_gaochenbianyixishu', 'rc_duancengmidu', 'rc_dixingshidu', 'rc_flowacc_log', 'rc_turanglx',
        'rc_poxing', 'rc_poweitu', 'yanxing']
# cols = ['APTT', 'INR(PT)','Lymphocytes', 'Monocytes',
#         'Neutrophil', 'White_Blood_Cells', 'Platelet_Count','patients_age']
# df = pd.read_csv(r'F:\俞泽峰\test_features08.csv', header=None)  # 包含标签
df = pd.read_csv(r'D:\研究生工作\课题组代码及数据\瑞昌市数据\rc_ori_test1340.csv').iloc[:, 1:]
# df = pd.read_csv('../logs/20221205-18-59_epoch600_lr0.004/original_post_pyh_test_features_labels.csv', header=None)  # 包含标签
# df = pd.read_csv(r'D:\Landslide\comparison\logs\20230321-11-40_epoch500_lr4.8e-05_yuandata\original_post_pyh_test_features_labels.csv', header=None)
# df = pd.read_csv('../logs/20221205-20-57_epoch150_lr0.004/original_post_pyh_test_features_labels.csv', header=None)  # 包含标签
print(df.shape)

df.columns = cols + ["Label"]

# pred = pd.read_csv(r'F:\俞泽峰\pre_test_results08.csv', header=None)

# pred = pd.read_csv('../logs/20221205-18-59_epoch600_lr0.004/post_best_acc_test_results.csv', header=None)
pred = pd.read_csv(r'D:\研究生工作\研二上\开题\师兄\pre_test_results.csv', header=None)

# pred = pd.read_csv(r'D:\Landslide\comparison\logs\20230321-11-40_epoch500_lr4.8e-05_yuandata\post_best_acc_test_results.csv', header=None)
# pred = pd.read_csv('../logs/20221205-20-57_epoch150_lr0.004/post_best_acc_test_results.csv', header=None)
# pred = pd.read_csv('../logs/20221205-20-31_epoch100_lr0.004/pre_test_results.csv', header=None)
pred.columns = ['no', 'yes']
print(pred.shape)
print(type(pred))
# print(pred.dtype)

x = df.copy()
x = x.round(5)             # round()是python自带的一个函数，用于数字的四舍五入
Y = x.pop('Label')
pred = pred.round(6)

# pred_1 = pred[(pred.iloc[:, -1] > 0.9)]
# print('pred_1.shape', pred_1.shape)
# pred_0 = pred[(pred.iloc[:, -1] < 0.1)]
# print('pred_0.shape', pred_0.shape)

# sk = pred.iloc[:, -1].skew()
# # # # # print(' pre_pyh_concat_train_test.shape', pre_pyh_concat_train_test.shape)   # (5673+1419,)=(7092,)
# sk = sk / (1 + 2 * abs(sk))
# T1 = abs(0.6 * sk) ** 0.5
# T0 = 1 - T1
# # # #
# test_concat = pd.concat([df, pred], axis=1)
# test_concat.to_csv('./test_concat.csv', header=True, index=False)
#
# # # # print(test_concat.shape)
# test_concat = test_concat[((test_concat.iloc[:, -3] == 1) & (test_concat.iloc[:, -1] > 0.5)) |
#                           ((test_concat.iloc[:, -3] == 0) & (test_concat.iloc[:, -1] < 0.5))]
# # # print('hhhhju',test_concat.shape)
# #
# #
# # #print('x.shape', X.shape)  # x.shape (2128, 42)
# pred = test_concat.iloc[:, -2:]
# # # # print(pred.shape)
# x = test_concat.iloc[:, :-3]
# # # print(x.shape)
# q2_all = pd.DataFrame([])
# print(q2_all, q2_all)

for idx, col in enumerate(cols):
    fig, axes, summary_df = onefact_actual_plot(pred.to_numpy(), x, feature=col, feature_name=col, num_grid_points=11,
                                                grid_type='percentile', percentile_range=None,
                                                grid_range=None, cust_grid_points=None, show_percentile=True,
                                                show_outliers=False, endpoint=True,
                                                which_classes=None, predict_kwds={}, ncols=2, figsize=None,
                                                plot_params=None, f2v=f2v)
    # fig.show()

    fig.savefig(f"../one_factor_results/{idx+1}-{col}.svg", format="svg")

    print(f"[{idx + 1}|{len(cols)}] -> {col}.png Done.")
    #
    # exit()   # 退出当前循环
