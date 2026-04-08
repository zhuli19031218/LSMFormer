import pandas as pd
import matplotlib.pyplot as plt
def get_qujian_mean(data, qujian1, qujian2, name1, name2, i, j):
    one_q = data[(data[name1] >= qujian1[i]) & (data[name1] < qujian1[i + 1]) & (data[name2] >= qujian2[j]) & (data[name2] < qujian2[j + 1])]
    _shape = one_q.shape
    # print(f"{i},{j}:", _shape)
    return one_q.iloc[:, -1].mean() if _shape[0]>0 else 0


# df = pd.read_csv(r'D:\sss\3D_Waterfall\data\rc_freqTest1340.csv').iloc[:, 1:]
# print(df[:5])
# label = pd.read_csv(r'D:\sss\3D_Waterfall\data\result.csv').iloc[:, 1]

all_data = pd.read_csv(r"D:\sss\whole_results\data\Ruichang_row_data_sort.csv")
print(all_data.shape)
# exit()
all_prediction = pd.read_csv(r'D:\sss\3D_Waterfall\data\DP-GIR_瑞昌市易发性.csv', header=None)
print(all_prediction.shape)

data = pd.concat([all_data, all_prediction], axis=1)

feature_names = ['rc_zongfushe', 'rc_yeguang', 'dem', 'rc_pinmianqvlv',
       'rc_hewangmidu', 'rc_dixingshidu', 'rc_MNDWI', 'rc_luwangmidu',
       'rc_slope', 'rc_gaochenbianyixishu', 'rc_poumianqvlv',
       'rc_nianjunjiangyu', 'rc_rock_weathering_indexes',
       'rc_NDVI', 'rc_flowacc', 'rc_turangliangdu', 'rc_turangchengfen',
       'rc_renkoumidu', 'rc_duancengmidu']

K = 25
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):

        # name1 = 'rc_nianjunjiangyu'
        # name2 = 'rc_MNDWI'
        # name2 = 'rc_nianjunjiangyu'
        cut_num1 = K
        cut_num2 =K
        # _, qujian1 = pd.cut(data[name1], bins=cut_num1, retbins=True, right=False)
        _, qujian1 = pd.cut(data[feature_names[i]], bins=cut_num1, retbins=True, right=False)

        # qujian1 = [ 0., 4.63489838,  9.26979676, 13.90469514, 18.53959352, 27.80939028, 41.71408542, 46.39533278]
        print(qujian1)
        # _, qujian2 = pd.cut(data[name2], bins=cut_num2, retbins=True, right=False)
        _, qujian2 = pd.cut(data[feature_names[j]], bins=cut_num2, retbins=True, right=False)
        print(qujian2)
        list_all = []
        for m in range(cut_num1):
            list_all.append([])
            for n in range(cut_num2):
                # prediction = get_qujian_mean(data, qujian1, qujian2, name1, name2, i, j)
                prediction = get_qujian_mean(data, qujian1, qujian2, feature_names[i], feature_names[j], m, n)
                # print(prediction)
                list_all[m].append(prediction)
        print(list_all)
        pd_list = pd.DataFrame(list_all)
        print(pd_list)
        # pd_list.to_csv(r"data/slope_TWI.csv", index=False, header=None)
        plt.imshow(pd_list)
        print(f'{feature_names[i]} and {feature_names[j]}')
        plt.title(f'{feature_names[i]} and {feature_names[j]}')
        plt.savefig(f'D:/sss/3D_Waterfall/new_fig/{feature_names[i]} and {feature_names[j]}.svg')
        # plt.show()
# pd_list.to_csv(r"data/Whole_FlowACC_MNDWI100.csv", header=None, index=False)
exit()

dem = [8.1627121,  116.94579424, 225.72887639, 334.51195853, 443.29504067,
 552.07812281, 660.86120496, 770.40576867]
NDVI = [0., 0.1,   0.2,   0.3,   0.4,   0.5,   0.6,   0.7,   0.8,   0.9,   1.001]
profile = [0., 6.49260221, 12.98520443, 19.47780664, 25.97040886, 32.46301107,
 38.95561329, 45.49366372]
a, b = pd.cut(data["dem"], bins=7, retbins=True, right=False)
print(a, b)
print(b[0])
# one_q = data[(data['dem'] >= dem[0]) & (data['dem'] < dem[1]) & (data['rc_NDVI'] >= NDVI[0]) & (data['rc_NDVI'] < NDVI[1])]
# print(one_q.shape)
# prediction = one_q.iloc[:, -1].mean()
# print(prediction)
# for i in b:
#     print(i)


# c, d = pd.cut(data["rc_poumianqvlv"], bins=7, retbins=True, right=False)
# print(c, d)
# list_all = []
# for i in range(7):
#     list_all.append([])
#     for j in range(7):
#         prediction = get_qujian_mean(data, dem, profile, "dem", "rc_poumianqvlv", i, j)
#         # print(prediction)
#         list_all[i].append(prediction)
# print(list_all)

