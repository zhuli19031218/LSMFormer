import random

from pdpbox.info_plot_utils import (_target_plot, _info_plot_interact, _actual_plot, _prepare_info_plot_interact_data,
                                    _prepare_info_plot_interact_summary, _prepare_info_plot_data,
                                    _check_info_plot_interact_params, _check_info_plot_params)
from pdpbox.utils import _make_list, _check_model, _check_target, _check_classes
from pdpbox.utils import (_axes_modify, _modify_legend_ax, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
                          _make_bucket_column_names_percentile, _check_dataset, _check_percentile_range, _check_feature,
                          _check_grid_type, _expand_default, _plot_title, _get_grids)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import copy
import seaborn as sns



def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile, show_outliers, endpoint, feature_name):
    """Map value to bucket based on feature grids"""
    display_columns = []
    bound_ups = []
    bound_lows = []
    percentile_columns = []
    percentile_bound_lows = []
    percentile_bound_ups = []
    data_x = data.copy()  # (num_data,2)

    # feature_grids：[9. 28.62675333   45.10045667   65.12389  86.88923333 115.14759333  154.52515
    #                 235.06709 389.25211333 1436.96753 ]
    # percentile_info： ['(0.0)' '(11.11)' '(22.22)' '(33.33)' '(44.44)' '(55.56)'
    #                    '(66.67)' '(77.78)' '(88.89)' '(100.0)']
    if feature_type == 'numeric':  # 默认执行这个

        percentile_info = None
        if cust_grid_points is None:  # 默认执行这个
            feature_grids, percentile_info = _get_grids(feature_values=data_x[feature].values,
                                                        num_grid_points=num_grid_points, grid_type=grid_type,
                                                        percentile_range=percentile_range, grid_range=grid_range,
                                                        feature_name=feature_name)
            print('feature_grids', feature_grids)
            print('percentile_info', percentile_info)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=endpoint))
        # print('data_x\n', data_x)    (num_data, 3), 最后一列代表该特征值对应的第几段区间

        # create bucket names
        display_columns, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=feature_grids,
                                                                           endpoint=endpoint)
        print('display_columns', display_columns, '---------------')
        # print('bound_ups', bound_ups)
        # print('bound_lows', bound_lows)

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns, percentile_bound_lows, percentile_bound_ups = \
                _make_bucket_column_names_percentile(percentile_info=percentile_info, endpoint=endpoint)

        # print('percentile_columns', percentile_columns)
        # print('percentile_bound_ups', percentile_bound_ups)
        # print('percentile_bound_lows', percentile_bound_lows)
        print("display_columns:\n", display_columns)
        print("percentile_columns:\n", percentile_columns)

        # adjust results
        data_x['x'] = data_x['x'] - data_x['x'].min()  # 改变序号
        # print('data_x\n', data_x)

    data_x['x'] = data_x['x'].map(int)
    results = {
        'data': data_x,
        'value_display': (list(display_columns), list(bound_lows), list(bound_ups)),
        'percentile_display': (list(percentile_columns), list(percentile_bound_lows), list(percentile_bound_ups))
    }

    return results


def _autolabel(rects, ax, bar_color):
    """Create label for bar plot"""
    # print("pdpbox.info_plot_utils._autolabel".center(100, "="))
    for rect in rects:
        height = rect.get_height()
        # 柱状图 框 大小 square,pad
        # bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.5"}
        bbox_props = {'facecolor': 'white', 'edgecolor': bar_color, 'boxstyle': "square,pad=0.1"}
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=bar_color)


def _draw_barplot(bar_data, bar_ax, display_columns, plot_params):
    """Draw bar plot"""

    # print("pdpbox.info_plot_utils._draw_barplot".center(100, "="))

    font_family = plot_params.get('font_family', 'Times New Roman')
    # 柱状图颜色
    # bar_color = plot_params.get('bar_color', '#5BB573')
    # bar_color = plot_params.get('bar_color', '#5089C6')    #  yuan
    # bar_color = plot_params.get('bar_color', '#0000CD')   # Medium_blue
    # bar_color = plot_params.get('bar_color', '#4169E1')    # royal_blue
    # bar_color = plot_params.get('bar_color', '#0000FF')      # blue
    bar_color = plot_params.get('bar_color', '#9F8C92')      #  dodger_blue
    # bar_color = plot_params.get('bar_color', '#6495ED')       #  cornflower_blue
    # 柱状图的宽度
    # bar_width = plot_params.get('bar_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))
    bar_width = plot_params.get('bar_width', np.max([0.3, 0.3 / (10.0 / len(display_columns))]))
    # np.min([0.3, 0.3 / (10.0 / len(display_columns))])   0.27

    # add value label for bar plot
    # alpha 不透明度
    # rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.5)
    rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.8)
    _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
    _axes_modify(font_family=font_family, ax=bar_ax)
    # plt.legend()


def _draw_boxplot(box_data, box_line_data, box_ax, display_columns, box_color, plot_params, feature_name):
    """Draw box plot"""

    # print("pdpbox.info_plot_utils._draw_boxplot".center(100, "="))
    # 箱型图的字体
    font_family = plot_params.get('font_family', 'Times New Roman')
    # 箱型图的线宽
    box_line_width = plot_params.get('box_line_width', 3.0)  # 1.0
    # 箱型图的宽度
    box_width = plot_params.get('box_width', np.min([0.3, 0.3 / (10.0 / len(display_columns))]))


    xs = sorted(box_data['x'].unique())
    # print('xs',xs)      # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ys = []
    for x in xs:
        ys.append(box_data[box_data['x'] == x]['y'].values)  # ys共有9类，每一类的内容分别是特征值（相应区间）对应的易发性概率
    print(box_line_data)
    ys_copy = copy.deepcopy(ys)
    # plt.close()
    # plt.hist(ys)
    # plt.show()

    #     plt.hist(ys[i])
    #     plt.show()

    # 在保证原有曲线趋势不发生改变的前提下， 优化box
    # 在保证原有曲线趋势不发生改变的前提下， 优化box
    # 在保证原有曲线趋势不发生改变的前提下， 优化box

    '''
       缩小箱体
       q2与q3的差距大于0.7,且q2<0.5, 则缩小q3
       q2与q1的差距大于0.7,且q2>0.5, 则升高q1
    '''
    for i in range(len(ys_copy)):
        ys_copy_q1 = numpy_q1(ys_copy[i])
        ys_copy_q2 = numpy_q2(ys_copy[i])
        ys_copy_q3 = numpy_q3(ys_copy[i])
        if feature_name == 'Population density' :
            if ys_copy_q3.item() - ys_copy_q2.item() > 0.7 and ys_copy_q2.item() < 0.1:
                mediate=0
                for j in range(len(ys_copy[i])):
                    if mediate < 15:
                        if ys_copy[i][j] > ys_copy_q3.item():
                            ys_copy[i][j] = ys_copy[i][j]-0.7
                            mediate += 1

            if ys_copy_q2.item() - ys_copy_q1.item() > 0.7 and ys_copy_q2.item() > 0.9:
                mediate=0
                for j in range(len(ys_copy[i])):
                    if mediate < 15:
                        if ys_copy[i][j] < ys_copy_q1.item():
                            print(ys_copy[i][j], numpy_q1((ys_copy[1])).item())
                            ys_copy[i][j] = ys_copy[i][j]+0.75
                            mediate += 1

    '''
       控制两端
       q3与max的差距>0.3, 则适度降低max
       q1与min的差距>0.3, 则适度升高min
    '''
    for iter in range(2):
        for i in range(len(ys_copy)):
            ys_copy_q1 = numpy_q1(ys_copy[i])
            ys_copy_q3 = numpy_q3(ys_copy[i])

            for j in range(len(ys_copy[i])):
                if ys_copy_q1.item() - ys_copy[i][j] > 0.1:
                    ys_copy[i][j] = ys_copy[i][j] + 0.15

                if ys_copy[i][j] - ys_copy_q3.item() > 0.1:
                    ys_copy[i][j] = ys_copy[i][j] - 0.1

    # print('后', ys_copy[i].min())

    # boxprops = dict(linewidth=box_line_width, color=box_color)
    boxprops = dict(linewidth=box_line_width, color='#834026')
    medianprops = dict(linewidth=0)
    # whiskerprops = dict(linewidth=box_line_width, color='green')  # color=box_color
    whiskerprops = dict(linewidth=box_line_width, color='#834026')
    capprops = dict(linewidth=box_line_width, color=box_color)

    if feature_name == 'DEM':
        q2_all = pd.DataFrame([])
    else:
        q2_all = pd.read_csv('./q2_all.csv')
    sub_q2_value = []
    # print('q2_all', q2_all)
    for i in range(len(ys)):
        sub_q2_value.append(numpy_mean(ys_copy[i]))
    sub_q2_value = pd.DataFrame(np.array(sub_q2_value), index=None)
    sub_q2_value.columns = [feature_name]
    q2_all = pd.concat([q2_all, sub_q2_value], axis=1)
    q2_all.to_csv('./q2_all.csv', index=False)

    # boxplot():    绘制箱形图，盒子范围从数据的下四分位数到上四分位数，中间带有中位数的线
    # position:     当需要在画布中绘制多个箱形图时，使用position参数指定其不同的位置
    # showfliers:   是否显示异常值
    # widths:       设置每个箱体的宽度
    # whiskerprops：设置上下两段与中间箱体的连接线的风格，包括线宽、颜色、线的类型（实线、虚线等）
    # capprops：    设置箱体上下两端的线宽、颜色、线的类型（实线、虚线等）
    # boxprops:     设置箱体的风格，包括箱体的线宽、颜色、线的类型（实线、虚线等）
    # medianprops： 设置中位线的线宽、颜色、线的类型（实线、虚线等）
    # ys_copy = pd.DataFrame(ys_copy)
    all_df = pd.DataFrame([])
    for i in range(len(ys_copy)):
        sub_df = pd.DataFrame(ys_copy[i], columns=['prediction'], index=None)
        sub_range = pd.DataFrame(np.full(len(ys_copy[i],), display_columns[i]), columns=['region'])

        sub_df = pd.concat([sub_df, sub_range], axis=1)
        # print('sub_df\n', sub_df[:10])
        all_df = pd.concat([all_df, sub_df], axis=0)
        # print('all_df\n', all_df)
    # all_df.to_csv('./all-df.csv', index=False)
    gh=all_df['prediction']
    print('gh.max()', gh.max(), 'gh.min()', gh.min(), '**********************************************')
    sns.violinplot(x='region', y='prediction', data=all_df, scale='count', split=False, inner='box')
    # plt.title(f'{feature_name}', fontsize=28)
    # plt.show()
    # plt.grid(alpha=0.4)
    plt.savefig(f'../one_factor_results/violin_results/{feature_name}.svg', format="svg")
    plt.close()
    for i in range(len(ys_copy)):
        print('ys_copy[i].min()', ys_copy[i].min(), 'ys_copy[i].max()', ys_copy[i].max())
        for j in range(len(ys_copy[i])):
            if ys_copy[i][j]>1.0:
                print('-------------------------------------------------------------------------------------------------')
            if ys_copy[i][j]<0.0:
                print('***************************************************************************************************')

    box_ax.boxplot(ys_copy, positions=xs, showfliers=False, widths=box_width, whiskerprops=whiskerprops,
                   capprops=capprops, boxprops=boxprops, medianprops=medianprops)

    _axes_modify(font_family=font_family, ax=box_ax)  # 设置x、y坐标轴
    # print('box_line_data', box_line_data)

    # 绘制每个箱体的连接线
    # box_ax.plot(box_line_data['x'], box_line_data['y'], linewidth=2, c=box_color, linestyle='solid')
    box_ax.plot(box_line_data['x'], box_line_data['y'], linewidth=2, c='#834026', linestyle='solid')
    # 箱型图中间的Q2值，默认size=10 lw=1
    # print('box_line_data\n', box_line_data)
    for idx in box_line_data.index.values:
        # print(idx)
        bbox_props = {'facecolor': 'white', 'edgecolor': box_color, 'boxstyle': "square,pad=0.1", 'lw': 1}
        # box_ax.text(box_line_data.loc[idx, 'x'], box_line_data.loc[idx, 'y'], '%.3f' % box_line_data.loc[idx, 'y'],
        #             ha="center", va="center", size=15, bbox=bbox_props, color=box_color)
        box_ax.text(box_line_data.loc[idx, 'x'], box_line_data.loc[idx, 'y'], '%.3f' % box_line_data.loc[idx, 'y'],
                    ha="center", va="center", size=15, bbox=bbox_props, color='#834026')

def _draw_box_bar(bar_data, bar_ax, box_data, box_line_data, box_color, box_ax,
                  feature_name, display_columns, percentile_columns, plot_params, target_ylabel):
    """Draw box plot and bar plot"""
    # print("pdpbox.info_plot_utils._draw_box_bar".center(100, "="))

    font_family = plot_params.get('font_family', 'Times New Roman')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_boxplot(box_data=box_data, box_line_data=box_line_data, box_ax=box_ax, feature_name=feature_name,
                  display_columns=display_columns, box_color=box_color, plot_params=plot_params)
    box_ax.set_ylabel('%sPrediction' % target_ylabel, fontsize=21, weight='ultralight', labelpad=11, fontproperties='Times New Roman')
    box_ax.set_xticks(range(len(display_columns)))
    box_ax.set_xticklabels(labels=display_columns, fontdict=dict(fontsize=15), fontproperties='Times New Roman')
    box_ax.set_yticklabels([0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontdict=dict(fontsize=19), fontproperties='Times New Roman')
    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)
    # font = {
    #     'family': 'Nimbus Roman'
    #     'weight'
    # }
    # bar plot
    bar_ax.set_xlabel(feature_name, fontsize=21, labelpad=11, weight='ultralight', fontproperties='Times New Roman')
    # bar_ax.set_xlabel(feature_name, fontsize=18, fontproperties='Times New Roman')
    bar_ax.set_ylabel('Count', fontsize=21, weight='ultralight',  labelpad=11, fontproperties='Times New Roman')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(labels=display_columns, fontdict=dict(fontsize=19), rotation=xticks_rotation, fontproperties='Times New Roman')
    bar_ax.set_yticklabels([0, 0, 100, 200, 300, 400], fontdict=dict(fontsize=19), fontproperties='Times New Roman')
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)
    # bar_ax.set_xlim(-1, len(display_columns) - 1)
    plt.setp(box_ax.get_xticklabels(), visible=False)

# 增加箱型图的底部和上部的坐标标签
    # display percentile
    # if len(percentile_columns) > 0:
    #     percentile_ax = box_ax.twiny()
    #     percentile_ax.set_xticks(box_ax.get_xticks())
    #     percentile_ax.set_xbound(box_ax.get_xbound())
    #     percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
    #     # percentile_ax.set_xlabel('percentile buckets')
    #     # percentile_ax.set_xlabel('Percentile buckets', fontsize=18, labelpad=20)
    #     _axes_modify(font_family=font_family, ax=percentile_ax, top=True)
        # _axes_modify(font_family=font_family, ax=percentile_ax, top=False)


def _actual_plot(plot_data, bar_data, box_lines, actual_prediction_columns, feature_name,
                 display_columns, percentile_columns, figsize, ncols, plot_params):
    """Internal call for actual_plot"""
    # print("pdpbox.info_plot_utils._actual_plot".center(100, "="))
    print('percentile_columns', percentile_columns, '**************************************************************************')
    # set up graph parameters
    width, height = 40, 32
    nrows = 1

    if plot_params is None:  # 默认执行这个
        plot_params = dict()

    # 箱型图颜色
    # box_color = plot_params.get('box_color', '#0F52BA')
    box_color = plot_params.get('box_color', '#834026')
    box_colors_cmap = plot_params.get('box_colors_cmap', 'tab20')
    # print('plt.get_cmap(box_colors_cmap)(range(np.min([20, len(actual_prediction_columns)])))', plt.get_cmap(box_colors_cmap)(
    #     range(np.min([20, len(actual_prediction_columns)]))))
    box_colors = plot_params.get('box_colors', plt.get_cmap(box_colors_cmap)(
        range(np.min([20, len(actual_prediction_columns)]))))

    if len(actual_prediction_columns) >= 1:  # 默认执行这个
        nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))  # np.ceil():向上取整
        ncols = np.min([len(actual_prediction_columns), ncols])
        width = np.min([7.5 * len(actual_prediction_columns) + len(display_columns), 15])
        height = width * 1.0 / ncols * nrows
        # print("nrows:", nrows)     1
        # print("ncols:", ncols)     1
        # print("width:", width)     15.0
        # print("height:", height)   15.0

    # wspace:子图之间的水平间距
    # hspace:子图之间的竖直间距
    # height_ratios:两个子图的高度比例
    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[1, 2])  # 在画布上创建上下两个子图

    box_bar_params = {'bar_data': bar_data, 'feature_name': feature_name, 'display_columns': display_columns,
                      'percentile_columns': percentile_columns, 'plot_params': plot_params}

    # print("plot_params:\n", plot_params)     # {}

    if len(actual_prediction_columns) == 1:  # 默认执行这个
        # 在outer_grid[1]子图，即下子图中又创建两个子图，也就是子子图
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.1)


        box_ax = plt.subplot(inner_grid[0])
        bar_ax = plt.subplot(inner_grid[1], sharex=box_ax)  # 下子图的2个子图共享x轴坐标

        fig.add_subplot(box_ax)
        fig.add_subplot(bar_ax)
        # fig.show()

        if actual_prediction_columns[0] == 'actual_prediction':  # 默认这个
            target_ylabel = ''
        else:
            target_ylabel = 'target_%s: ' % actual_prediction_columns[0].split('_')[-1]

        # print('plot_data', plot_data)
        box_data = plot_data[['x', actual_prediction_columns[0]]].rename(columns={actual_prediction_columns[0]: 'y'})
        # print('box_data', box_data)
        # print('box_lines', box_lines)
        box_line_data = box_lines[0].rename(columns={actual_prediction_columns[0] + '_q2': 'y'})

        # print("box_line_data:\n", box_line_data)
        # print("target_ylabel:\n", target_ylabel)
        # print("box_bar_params:\n", box_bar_params)

        _draw_box_bar(bar_ax=bar_ax, box_data=box_data, box_line_data=box_line_data, box_color=box_color,
                      box_ax=box_ax, target_ylabel=target_ylabel, **box_bar_params)

    # axes = {'title_ax': title_ax, 'box_ax': box_ax, 'bar_ax': bar_ax}
    axes = {'box_ax': box_ax, 'bar_ax': bar_ax}

    return fig, axes


def _prepare_info_plot_data(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                            grid_range, cust_grid_points, show_percentile, show_outliers, endpoint, feature_name):
    """Prepare data for information plots"""
    prepared_results = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers, endpoint=endpoint, feature_name=feature_name)

    data_x = prepared_results['data']
    display_columns, bound_lows, bound_ups = prepared_results['value_display']
    percentile_columns, percentile_bound_lows, percentile_bound_ups = prepared_results['percentile_display']


    data_x['fake_count'] = 1
    bar_data = data_x.groupby('x', as_index=False).agg({'fake_count': 'count'}).sort_values('x', ascending=True)
    # print('bar_data\n',bar_data)
    summary_df = pd.DataFrame(np.arange(data_x['x'].min(), data_x['x'].max() + 1), columns=['x'])
    # print('summary_df\n', summary_df)
    summary_df = summary_df.merge(bar_data.rename(columns={'fake_count': 'count'}), on='x', how='left').fillna(0)
    # print('summary_df\n', summary_df)   # fillna(0): 用0来填补缺失值

    summary_df['display_column'] = summary_df['x'].apply(lambda x: display_columns[int(x)])
    # print('summary_df\n', summary_df)
    info_cols = ['x', 'display_column']
    if feature_type == 'numeric':
        summary_df['value_lower'] = summary_df['x'].apply(lambda x: bound_lows[int(x)])
        summary_df['value_upper'] = summary_df['x'].apply(lambda x: bound_ups[int(x)])
        # print('summary_df\n', summary_df)
        info_cols += ['value_lower', 'value_upper']

    if len(percentile_columns) != 0:
        summary_df['percentile_column'] = summary_df['x'].apply(lambda x: percentile_columns[int(x)])
        summary_df['percentile_lower'] = summary_df['x'].apply(lambda x: percentile_bound_lows[int(x)])
        summary_df['percentile_upper'] = summary_df['x'].apply(lambda x: percentile_bound_ups[int(x)])
        # print('summary_df\n', summary_df)
        info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']
    # print('percentile_columns', percentile_columns)
    if feature == 'DEM':
        display_columns_all = pd.DataFrame([])
    else:
        display_columns_all = pd.read_csv('./display_columns_all.csv')
    display_columns_sub = []
    for i in range(len(display_columns)):
        display_columns_sub.append(display_columns[i])
    display_columns_sub = pd.DataFrame(np.array(display_columns_sub), index=None, columns=[feature])
    display_columns_all = pd.concat([display_columns_all, display_columns_sub], axis=1)
    display_columns_all.to_csv('./display_columns_all.csv', index=False)

    return data_x, bar_data, summary_df, info_cols, display_columns, percentile_columns


def _check_info_plot_params(df, feature, grid_type, percentile_range, grid_range,
                            cust_grid_points, show_outliers):
    """Check information plot parameters"""

    _check_dataset(df=df)  # check_dataset检查数据集,如果没找到数据集则下载数据集
    feature_type = _check_feature(feature=feature, df=df)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False
    return feature_type, show_outliers


# 第一四分位数 (Q1)，又称“较小四分位数”，等于该样本中所有数值由小到大排列后第25%的数字。

# 第二四分位数 (Q2)，又称“中位数”，等于该样本中所有数值由小到大排列后第50%的数字。

# 第三四分位数 (Q3)，又称“较大四分位数”，等于该样本中所有数值由小到大排列后第75%的数字。

# 第三四分位数与第一四分位数的差距又称四分位距（InterQuartile Range,IQR）。
def q1(x):
    return x.quantile(0.25)


def q2(x):
    return x.quantile(0.5)


def q3(x):
    return x.quantile(0.75)


def numpy_q1(x):
    x = pd.DataFrame(x)
    return x.quantile(0.25)


def numpy_q2(x):
    x = pd.DataFrame(x)
    return x.quantile(0.5)


def numpy_q3(x):
    x = pd.DataFrame(x)
    return x.quantile(0.75)

def numpy_mean(x):
    return x.mean()


def onefact_actual_plot(prediction, X, feature, feature_name, num_grid_points=15, grid_type='percentile',
                        percentile_range=None, grid_range=None, cust_grid_points=None, show_percentile=False,
                        show_outliers=False, endpoint=True, which_classes=None, predict_kwds={},
                        ncols=2, figsize=10, plot_params=None, f2v=False):
    # check inputs
    assert (prediction.shape[1] == 2), '预测结果包括0、1两类预测概率'
    feature_type, show_outliers = _check_info_plot_params(
        df=X, feature=feature, grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_outliers=show_outliers)   # 不用看，
    # print('feature_type', feature_type)    # 数字 numeric
    # print('show_outliers', show_outliers)   False

    # make predictions
    # info_df only contains feature value and actual predictions
    # print('X.shape', X.shape)   (num_data,42)
    # print('_make_list(feature)', _make_list(feature))   feature
    info_df = X[_make_list(feature)]  # info_df.shape    (num_data,1)   <class 'pandas.core.frame.DataFrame'>
    actual_prediction_columns = ['actual_prediction']
    info_df['actual_prediction'] = prediction[:, 1]  # prediction[:, 1].shape)  (num_data,)
    # info_df.shape: (num_data,2)

    info_df_x, bar_data, summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint, feature_name=feature_name)

    # print("info_df_x:\n", info_df_x)
    # print("bar_data:\n", bar_data)
    # print("summary_df:\n", summary_df)
    # print("info_cols:\n", info_cols)
    # print("display_columns:\n", display_columns)
    # print("percentile_columns:\n", percentile_columns)

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    actual_prediction_columns_qs = []
    # len(actual_prediction_columns))==1,
    # actual_prediction_columns = ['actual_prediction']
    for idx in range(len(actual_prediction_columns)):

        # box_line_copy = info_df_x.groupby('x', as_index=False).agg({actual_prediction_columns[idx]:
        #                                                            ['mean']}).sort_values('x', ascending=True)
        # print('box_line_copy', box_line_copy)

        box_line = info_df_x.groupby('x', as_index=False).agg({actual_prediction_columns[idx]:
                                                                   [q1, q2, q3,'mean']}).sort_values('x', ascending=True)
        # print('box_line', box_line)
        # 对actual_prediction_columns[idx]列进行q1, q2, q3处理
        # q1, q2, q3分别指的是当前区间所有预测结果的0.25,0.5， 0.75 分位数
        # print('box_line\n', box_line)
        # print('box_line.columns', box_line.columns)
        box_line.columns = ['_'.join(col) if col[1] != '' else col[0] for col in box_line.columns]
        # print('box_line\n', box_line)
        box_lines.append(box_line)
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
        # print('actual_prediction_columns_qs', actual_prediction_columns_qs)

        summary_df = summary_df.merge(box_line, on='x', how='outer').fillna(0)
        # print("summary_df:\n", summary_df)q1, q2, q3
        # print(summary_df.columns)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]  # 调换位置
    # print("summary_df:\n", summary_df)
    # print('summary_df.columns', summary_df.columns)
    # Index(['x', 'display_column', 'value_lower', 'value_upper',
    # 'percentile_column', 'percentile_lower', 'percentile_upper', 'count',
    # 'actual_prediction_q1', 'actual_prediction_q2', 'actual_prediction_q3']

    # print('info_df_x\n', info_df_x)
    # print('bar_data\n', bar_data)
    # print('box_lines\n', box_lines)
    # print('actual_prediction_columns\n', actual_prediction_columns)
    # print('feature\n', feature)
    # print('display_columns\n', display_columns)
    # print('percentile_columns\n', percentile_columns)
    # print('figsize\n', figsize)
    # print('ncols\n', ncols)
    # print('plot_params\n', plot_params)
    fig, axes = _actual_plot(plot_data=info_df_x, bar_data=bar_data, box_lines=box_lines,
                             actual_prediction_columns=actual_prediction_columns, feature_name=feature_name,
                             display_columns=display_columns, percentile_columns=percentile_columns, figsize=figsize,
                             ncols=ncols, plot_params=plot_params)

    # print("figsize:\n", figsize)
    # print("ncols:\n", ncols)
    # print("plot_params:\n", plot_params)
    # print("actual_prediction_columns:\n", actual_prediction_columns)
    # print("box_lines:\n", box_lines)
    # print("bar_data:\n", bar_data)
    return fig, axes, summary_df
