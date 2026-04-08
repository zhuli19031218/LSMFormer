# -*- encoding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from joypy import joyplot
from scipy.stats import gaussian_kde



def get_new_color_list(color_list, total_len):
    new_list = []
    one_len = total_len//len(color_list)
    print(one_len)
    for i in range(total_len):
        if i <= one_len//2:
            left=0
            right=color_list[0]
            new_list.append(2*i/one_len*(right-left)+left)
            continue
        if i >= total_len-one_len//2-1:
            left = color_list[-1]
            right = 0
            new_list.append(2*(i+one_len//2-total_len) / one_len * (right - left) + left)
            continue
        left_inx = (i-int(one_len/2))//one_len
        left = color_list[left_inx]
        # print(left_inx+1)
        right = color_list[left_inx+1]
        new_list.append(((i+50) % one_len)/one_len*(right-left)+left)
    return new_list

# color_list = [1, 2, 3, 4, 3]
# new_color_list = get_new_color_list(color_list, 100)
# print(new_color_list)
# print(len(new_color_list))
# exit()

# def get_color(color_list, inx, total_len):
#     position_p = inx / total_len * len(color_list)
#     size = total_len/len(color_list)
#     color_inx_right = (position_p+0.5)//1 #
#     value = color_list[color_inx_right] - (position_p-postion_left)*(color_list[color_inx_right]-color_list[color_inx_right-1])
#     return

def fig_generate(x, y, y_top=2.8, y_len=800, color_list=None):
    if color_list is None:
        color_list = [1, 2, 3, 4, 3, 2, 1]
    fig = []
    new_color_list = get_new_color_list(color_list, len(x))
    for i in range(len(x)):
        fig.append([new_color_list[i] if y_top-j<y[i] else 0 for j in np.linspace(0, y_top, y_len)])
    return np.array(fig).T, new_color_list


def fig_generate_with_color(x, y, y_top=2.8, y_len=800, color_list=None):
    if color_list is None:
        color_list = [1, 2, 3, 4, 3, 2, 1]
    fig = []
    for i in range(len(x)):
        fig.append([color_list[i] if y_top-j<y[i] else np.nan for j in np.linspace(0, y_top, y_len)])
    return np.array(fig).T, new_color_list

def compute_kde(data, v_name='value', bw_method=0.2, p=1000):
    values = data[v_name].values
    kde = gaussian_kde(values, bw_method)
    x = np.linspace(values.min(), values.max(), p)
    y = kde.evaluate(x)
    return x,y

def norm_process(data, name, m):
    data = data.loc[:, [name[m], 'Prediction']]
    data.columns = ['dem', 'Prediction']
    data['factors'] = name[m]
    return data

def get_qujian_mean(data, qujian1, name1, i):
    t = '[' + f'{round(qujian1[i], 2)}' + ',' + f'{round(qujian1[i + 1], 2)}' + ')'
    one_q = data[(data[name1] >= qujian1[i]) & (data[name1] < qujian1[i + 1])]
    _shape = one_q.shape
    print(f"{i}:", _shape)
    return one_q.iloc[:, -1].mean() if _shape[0] else 0


feature_names = ['dem', 'rc_zongfushe', 'rc_yeguang', 'rc_turangliangdu',
                 'rc_turangchengfen', 'rc_slope', 'rc_rock_weathering_indexes',
                 'rc_renkoumidu', 'rc_poumianqvlv', 'rc_pinmianqvlv',
                 'rc_nianjunjiangyu', 'rc_NDVI', 'rc_MNDWI', 'rc_luwangmidu',
                 'rc_hewangmidu', 'rc_gaochenbianyixishu', 'rc_duancengmidu',
                 'rc_dixingshidu', 'rc_flowacc_log', 'rc_turanglx', 'rc_poxing',
                 'rc_poweitu', 'yanxing', 'Prediction']
#
# df = pd.read_csv(r'D:\sss\Ridgeline\data\rc_freqTest1340.csv').iloc[:, 1:]
# df1 = df.pop('label')
# df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# label =  pd.read_csv(r'D:\sss\3D_Waterfall\data\result.csv').iloc[:, 1]
# data = pd.concat([df, label], axis=1)
# data.columns = feature_names
# print(data)

all_data = pd.read_csv(r"D:\sss\whole_results\data\Ruichang_row_data_sort.csv")
all_data = all_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(all_data['rc_nianjunjiangyu'].max())
print(all_data['rc_nianjunjiangyu'].min())
# print(all_data.shape)
all_prediction = pd.read_csv(r'D:\sss\3D_Waterfall\data\DP-GIR_瑞昌市易发性.csv', header=None)
print(all_prediction.shape)
data = pd.concat([all_data, all_prediction], axis=1)


feature_names1 = ['rc_zongfushe', 'rc_yeguang', 'rc_turangliangdu',
                 'rc_turangchengfen', 'rc_slope', 'rc_rock_weathering_indexes',
                 'rc_renkoumidu', 'rc_poumianqvlv', 'rc_pinmianqvlv',
                 'rc_nianjunjiangyu', 'rc_luwangmidu',
                 'rc_hewangmidu', 'rc_gaochenbianyixishu', 'rc_duancengmidu',
                 'rc_dixingshidu', 'rc_flowacc']
for i in range(len(feature_names1)):
# name1 = 'rc_flowacc'
    name1 = feature_names1[i]
    saved = False
    if not saved:
        x_a, y_a = compute_kde(data, v_name='dem')
        print(x_a, y_a)
        save = pd.DataFrame([x_a, y_a]).T
        print(save)
        save.to_csv(f"kde_data/{name1}1000.csv")
    else:
        save = pd.read_csv(f"kde_data/{name1}1000.csv", index_col=0)
        x_a = save.iloc[:, 0]
        y_a = save.iloc[:, 1]
    # plt.plot(x_a, y_a)
    # plt.show()

    # all_data = pd.read_csv(r'D:\sss\3D_Waterfall\data\rc_freqTest1340.csv').iloc[:, 1:]
    # # print(df[:5])
    # all_prediction = pd.read_csv(r'D:\sss\3D_Waterfall\data\result.csv').iloc[:, 1]
    # concat_data = pd.concat([all_data, all_prediction], axis=1)
    concat_data = data
    cut_num1 = 20  # 最好是能被1000整除
    _, qujian1 = pd.cut(concat_data[name1], bins=cut_num1, retbins=True, right=False)


    list_all = []
    # plot_data = pd.DataFrame()
    for m in range(cut_num1):
        # list_all.append([])
        # for n in range(1):
        prediction = get_qujian_mean(concat_data, qujian1, name1, m)
        # prediction = get_qujian_mean(data, qujian1, qujian2, feature_names[i], feature_names[j], m, n)
        # print(prediction)
        # list_all[m].append(prediction)
        # list_all.append(count)
        list_all.append(prediction)

    print(list_all)
    fig, new_color_list = fig_generate(x_a, y_a ,color_list=list_all, y_len=300)

    # plt.imshow(fig, cmap='Blues')
    # plt.show()

    linspace = np.linspace(0, 1, cut_num1)
    z = np.polyfit(linspace, list_all, 10)
    p = np.poly1d(z)

    linspace = np.linspace(0, 1, 1000)
    new_color_value = p(linspace)

    plt.plot(x_a, new_color_list)
    plt.plot(linspace, new_color_value)

    # plt.show()

    fig, new_color_list = fig_generate_with_color(x_a, y_a ,color_list=new_color_value, y_len=300)
    plt.imshow(fig, cmap='YlOrRd')
    # plt.colorbar()
    plt.savefig(f"pic/{name1}_{cut_num1}.svg", dpi=1200, bbox_inches="tight")
# plt.show()

