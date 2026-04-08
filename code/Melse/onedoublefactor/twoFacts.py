import matplotlib.pyplot as plt

from FactPlotUtils import *
import warnings as wn

import pdpbox
print(pdpbox.__version__)
# exit()
wn.filterwarnings("ignore")


# 是否将FR映射为真实值，无需操作则False。
# 没用FR值（鄱阳湖）为True
f2v = False

# cols = ['DEM',
#         'Slope',
#         'Aspect',
#         'Plan curvature',
#         'Profile curvature',
#         'Relief amplitude',
#         'Lithology',
#         'TWI',
#         'NDVI',
#         'NDBI',
#         'MNDWI',
#         'Total surface radiation']
#
# df = pd.read_csv('data/testAfterSceening.csv', header=None)
# df.columns = cols + ["Label"]
# pred = pd.read_csv('data/pred.csv', header=None)

# cols = ['DEM', 'renkoumidu', 'huiliuliejiliang', 'gaochenbianyixishu', 'luwangmidu', 'dixingshiduzhishu',
#         'duancengmidu', 'yanshifenghuazhishu', 'turangchengfen', 'nianjunjiangyuliang', 'podu', 'hewangmidu','poumianqulv',
#         'pingmianqulv','yeguang', 'NDVI', 'NDBI', 'MNDWI', '下坡','中坡','上坡','黄壤',
#         '棕壤', '酸性粗骨土', '中性紫色土', '其他土', '红壤','水稻土','灰潮土',
#         '岩性1','岩性2','岩性3','岩性4','岩性5','岩性6','岩性7','岩性8','岩性9','岩性10',
#         '岩性11','岩性12','岩性13',
#         ]
cols = ['DEM', 'renkoumidu', 'huiliuliejiliang', 'gaochenbianyixishu',
       'luwangmidu', 'dixingshiduzhishu', 'duancengmidu',
       'yanshifenghuazhishu', 'turangchengfen', 'nianjunjiangyuliang', 'podu',
       'hewangmidu', 'Profile curvature', 'pingmianqulv', 'yeguang', 'NDVI', 'NDBI',
       'MNDWI', 'powei', 'turangleixin', 'yanxing']
# df = pd.read_csv('./data/test_inputs.csv')
# y_df = pd.read_csv('data/test_labels.csv')
# df = pd.concat([df.iloc[:,1:],y_df.iloc[:,-1]],axis=1)  #(2128,43)
#
# # df = pd.read_csv('data/testAfterSceening.csv', header=None)  #(2060,13)
# df.columns = cols + ["Label"]
# pred = pd.read_csv('data/test_result.csv')
# pred = pred.iloc[:,1:]

df = pd.read_csv(r'D:\Landslide\comparison\logs\20221205-18-59_epoch600_lr0.004\original_post_pyh_test_features_labels.csv', header=None)  # 包含标签
print(df.shape)  # (1418, 22)

df.columns = cols + ["Label"]
pred = pd.read_csv(r'D:\Landslide\comparison\logs\20221205-18-59_epoch600_lr0.004\post_best_acc_test_results.csv', header=None)

# """双因子
X = df.copy()
X['prediction'] = pred.iloc[:, 1]
X = X.round(4)

length = len(cols) * (len(cols) - 1) // 2
idx = 0

filterCols = []

n_L = ['DEM']  #, 'podu', 'Slope'

n_name = ['DEM']
m_L = ['huiliuliejiliang', 'Profile curvature']# 'nianjunjiangyuliang', 'dixingshiduzhishu', 'MNDWI']
m_name = ['Flow Accumulation', "Profile curvature"]# 'Annual Rainfall', 'TWI', 'MNDWI']


for m in range(len(m_L)):
        for n in range(len(n_L)):
            filterCols.append((m, n))

color_group = ['GnBu', 'PuBu']
for color in color_group:
    for (i, j) in filterCols:
        feat_name2 = m_L[i]
        feat_name1 = n_L[j]
        if feat_name1 != feat_name2:
            # plt.clf()
            #                 sns.set_style('darkgrid')
            #                 sns.set(font_scale=1.5)
            #                 sns.set_palette('husl', 8)
            # fig, axes, summary_df = info_plots.target_plot_interact(df=X, features=[feat_name1, feat_name2],
            fig, axes, summary_df = target_plot_interact(df=X, features=[feat_name1, feat_name2],
                                                                    feature_names=[n_name[j], m_name[i]],
                                                                    target='prediction',
                                                                    # show_percentile=True,
                                                                    # endpoint=True,
                                                                    # ncols=1,
                                                                    annotate=True,
                                                                   plot_params={  # 调色盘色号   "cmap": "BuGn",
                                                                                "cmap": color,
                                                                                 "marker_size_min": 500,
                                                                                 "marker_size_max": 1800,
                                                                                 "xticks_rotation": 45},
                                                                    f2v=f2v)
            print(axes)
            axes['value_ax'].set_title(f"Average prediction", fontdict={'family': 'Times New Roman', "size": 18})  # {n_name[j]} & {m_name[i]}
            axes['value_ax'].set_xlabel(n_name[j], fontdict={'family': 'Times New Roman', "size": 18})
            axes['value_ax'].set_ylabel(m_name[i], fontdict={'family': 'Times New Roman', "size": 18})
            # axes['value_ax'].set_xticks(fontdict={'family': 'Times New Roman', "size": 8})
            idx += 1
            # fig.savefig(f"fig/2021-10-31-可解释性/数据-双因子/{idx}-{feat_name1} & {feat_name2}.png", dpi=600)
            # fig.savefig(f"./data/双因子plot_result/{idx}-{feat_name1} & {feat_name2}.svg", dpi=600)
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.pdf")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.eps")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.pdf")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.svg")
            # plt.clf()
            plt.show()

            print(f"[{idx}|{len(filterCols)}] -> {feat_name1} & {feat_name2}.png Done.")
        # break
# 渐进颜色板--['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
#            'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
'''
try:
    
    for m in range(17):
        for n in range(m+1,18):
        # filterCols = [(0, 16), (1, 6), (5, 6), (5, 7), (5, 9), (6, 7)]
            filterCols.append((m,n))
            
    for (i, j) in filterCols:
        feat_name1 = cols[i]
        feat_name2 = cols[j]
        if feat_name1 != feat_name2:
            # plt.clf()
            #                 sns.set_style('darkgrid')
            #                 sns.set(font_scale=1.5)
            #                 sns.set_palette('husl', 8)
            # fig, axes, summary_df = info_plots.target_plot_interact(df=X, features=[feat_name1, feat_name2],
            fig, axes, summary_df = target_plot_interact(df=X, features=[feat_name1, feat_name2],
                                                                    feature_names=[feat_name1, feat_name2],
                                                                    target='prediction',
                                                                    # show_percentile=True,
                                                                    # endpoint=True,
                                                                    # ncols=1,
                                                                    annotate=True,
                                                                    plot_params={"cmap": "Oranges",
                                                                                 "marker_size_min": 500,
                                                                                 "marker_size_max": 2800,
                                                                                 "xticks_rotation": 0},
                                                                    f2v=f2v)

            idx += 1
            # fig.savefig(f"fig/2021-10-31-可解释性/数据-双因子/{idx}-{feat_name1} & {feat_name2}.png", dpi=600)
            # fig.savefig(f"./data/双因子plot_result/{idx}-{feat_name1} & {feat_name2}.svg", dpi=600)
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.pdf")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.eps")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.pdf")
            # fig.savefig(f"D:/Users/ZZH.DESKTOP-97KAAD3/Desktop/test/2/{idx}-{feat_name1} & {feat_name2}.svg")
            # plt.clf()
            plt.show()

            print(f"[{idx}|{len(filterCols)}] -> {feat_name1} & {feat_name2}.png Done.")
            # exit()
except ValueError:
    print("ValueError: cannot reindex from a duplicate axis")
'''
# """
