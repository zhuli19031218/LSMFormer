# from DataAugment import weak_augment
import pandas as pd
import numpy as np


def weak_augment(x, m):
    # x = x.apply(lambda x: (x-x.min()) / (x.max()-x.min()))
    mean_values = x.mean()
    var_values = x.var()
    data = x.T
    ai = []
    for mean, var in zip(mean_values, var_values):
        # ai.append(np.multiply(data, factor[:, :]))
        factor = np.random.normal(loc=mean, scale=var * m, size=(data.shape[0], 1))
        print(factor)
        ai.append(np.multiply(data, factor))
        # print(type(ai))
        output = np.concatenate((ai), axis=1)
        ai.clear()
    return output.T



# origin_data = pd.read_csv(r'D:\sss\contrast\BYOL\data\contrast_data200000.csv').sample(frac=1)
origin_data = pd.read_csv(r'D:\sss\contrast\BYOL\data\new_RC_data\RC_norm300000.csv')
print(origin_data[:5])
# origin_data.to_csv(r'D:\sss\contrast\BYOL\data\origin_data_whole.csv', index=False)
print('原始数据保存完成'.center(100, '-'))
mean_values = origin_data.mean()
# print(mean_values)
var_values = origin_data.var()
# print(var_values)
augment1 = weak_augment(origin_data, m=7.5)
for i in range(len(augment1)):
    random_num = np.random.randint(0, high=21, size=2)
    augment1[i][random_num[0]] = np.random.normal(loc=mean_values[random_num[0]], scale=var_values[random_num[0]], size=1)
    augment1[i][random_num[1]] = np.random.normal(loc=mean_values[random_num[1]], scale=var_values[random_num[1]], size=1)
weak_data1 = pd.DataFrame(data=augment1, columns=origin_data.columns)
weak_data1.to_csv(r'D:\sss\contrast\BYOL\data\new_RC_data\rc_augment3.csv', index=False)
print('瑞昌市数据增强1保存完成'.center(100, '-'))

augment2 = weak_augment(origin_data, m=0.5)
for i in range(len(augment1)):
    random_num = np.random.randint(0, high=21, size=2)
    augment1[i][random_num[0]] = np.random.normal(loc=mean_values[random_num[0]], scale=var_values[random_num[0]], size=1)
    augment1[i][random_num[1]] = np.random.normal(loc=mean_values[random_num[1]], scale=var_values[random_num[1]], size=1)
weak_data2 = pd.DataFrame(data=augment2, columns=origin_data.columns)
weak_data2.to_csv(r'D:\sss\contrast\BYOL\data\new_RC_data\rc_augment4.csv', index=False)
print('瑞昌市数据增强2保存完成'.center(100, '-'))

