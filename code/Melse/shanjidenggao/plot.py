import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from joypy import joyplot


def norm_process(data, name, m):
    data = data.loc[:, [name[m], 'Prediction']]
    data.columns = ['dem', 'Prediction']
    data['factors'] = name[m]
    return data


feature_names = ['dem', 'rc_zongfushe', 'rc_yeguang', 'rc_turangliangdu',
                 'rc_turangchengfen', 'rc_slope', 'rc_rock_weathering_indexes',
                 'rc_renkoumidu', 'rc_poumianqvlv', 'rc_pinmianqvlv',
                 'rc_nianjunjiangyu', 'rc_NDVI', 'rc_MNDWI', 'rc_luwangmidu',
                 'rc_hewangmidu', 'rc_gaochenbianyixishu', 'rc_duancengmidu',
                 'rc_dixingshidu', 'rc_flowacc_log', 'rc_turanglx', 'rc_poxing',
                 'rc_poweitu', 'yanxing', 'Prediction']

df = pd.read_csv(r'D:\sss\Ridgeline\data\rc_freqTest1340.csv').iloc[:, 1:]
df1 = df.pop('label')
df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
label = pd.read_csv(r'D:\sss\Ridgeline\data\pre_test_results.csv').iloc[:, 1]
data = pd.concat([df, label], axis=1)
data.columns = feature_names
print(data)


total_data = pd.DataFrame()
for i in range(len(feature_names)-1):
    process_data = norm_process(data, feature_names, i)
    total_data = pd.concat([total_data, process_data])
print(total_data.reset_index(drop=True))


fig, ax = joyplot(total_data, by='factors', column='dem', figsize=(10, 12),
                  # 传入数据，y轴，x轴，设置图片尺寸
                  linecolor="white",  # 山脊线的颜色
                  # colormap=sns.color_palette(color, as_cmap=True),
                  colormap=sns.color_palette("coolwarm", as_cmap=True),
                  # 设置山脊图的填充色，用seaborn库的色盘，选择离散型颜色，as_cmap参数用来更改显示的颜色范围是离散的还是连续的
                  background='white')
# 设置背景色
fig.set_facecolor('#eae0d5')  # 设置画布背景色
plt.xlabel('name')  # 添加x轴名称
plt.title("Netflix Originals - IMDB Scores by Language")  # 添加标题
plt.subplots_adjust(top=0.95, bottom=0.1)  # 调整图形距离边框位置
plt.show()
# plt.savefig('joyplot_coolwarm.png', dpi=300)




