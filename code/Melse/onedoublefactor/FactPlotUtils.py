import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects

from pdpbox import pdp, get_dataset, info_plots

# from pdpbox.utils import (_make_list, _check_model, _check_target, _check_classes,_axes_modify, _modify_legend_ax, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
#                     _make_bucket_column_names_percentile, _check_dataset, _check_percentile_range, _check_feature,
#                     _check_grid_type, _expand_default, _plot_title, _get_grids)
from pdpbox.info_plot_utils import (_target_plot, _info_plot_interact, _actual_plot, _prepare_info_plot_interact_data,
                                    _prepare_info_plot_interact_summary, _prepare_info_plot_data,
                                    _check_info_plot_interact_params, _check_info_plot_params)


# from pdpbox.info_plot_utils import (_target_plot, _info_plot_interact, _actual_plot, _prepare_info_plot_interact_data,
#                               _prepare_info_plot_interact_summary, _prepare_info_plot_data,
#                               _check_info_plot_interact_params, _check_info_plot_params)
from pdpbox.utils import _make_list, _check_model, _check_target, _check_classes


# fr = pd.read_excel("data/fr2values.xlsx")

from pdpbox.utils import (_axes_modify, _modify_legend_ax, _find_bucket, _make_bucket_column_names, _find_onehot_actual,
                    _make_bucket_column_names_percentile, _check_dataset, _check_percentile_range, _check_feature,
                    _check_grid_type, _expand_default, _plot_title, _get_grids)

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects


def _prepare_data_x(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                    grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    """Map value to bucket based on feature grids"""
    display_columns = []
    bound_ups = []
    bound_lows = []
    percentile_columns = []
    percentile_bound_lows = []
    percentile_bound_ups = []
    data_x = data.copy()
    # print('data_x',data_x.shape,data_x)   # (num_data,2)
    # print('data_x[feature].values',data_x[feature].values)   # [ 35.0444   97.54814  80.20493 ...  32.42236  51.91011 262.50522]
    # print('data_x[feature].values',data_x[feature].values.shape)  # (num-data,)

    if feature_type == 'binary':
        feature_grids = np.array([0, 1])
        display_columns = ['%s_0' % feature, '%s_1' % feature]
        data_x['x'] = data_x[feature]
    if feature_type == 'numeric':         # 默认执行这个

        percentile_info = None
        if cust_grid_points is None:      # 默认执行这个
            feature_grids, percentile_info = _get_grids(
                feature_values=data_x[feature].values, num_grid_points=num_grid_points, grid_type=grid_type,
                percentile_range=percentile_range, grid_range=grid_range, feature_name=feature)
        else:
            feature_grids = np.array(sorted(cust_grid_points))

        if not show_outliers:             #  默认执行这个
            # print("show_outliers")
            data_x = data_x[(data_x[feature] >= feature_grids[0])
                            & (data_x[feature] <= feature_grids[-1])].reset_index(drop=True)
            # print('new_data_x',data_x.shape)   (num_data,2)

        # map feature value into value buckets
        data_x['x'] = data_x[feature].apply(lambda x: _find_bucket(x=x, feature_grids=feature_grids, endpoint=endpoint))
        # print('data_x\n',data_x)
        uni_xs = sorted(data_x['x'].unique())
        # print(' sorted(data_x[x]', sorted(data_x['x']))
        # print('sorted(data_x[x].unique())', sorted(data_x['x'].unique()))


        # create bucket names
        display_columns, bound_lows, bound_ups = _make_bucket_column_names(feature_grids=feature_grids,
                                                                           endpoint=endpoint)
        # display_columns = np.array(display_columns)[range(uni_xs[0], uni_xs[-1] + 1)]
        # bound_lows = np.array(bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
        # bound_ups = np.array(bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]
        # print('display_columns', display_columns)
        # print('bound_ups', bound_ups)
        # print('bound_lows', bound_lows)

        # create percentile bucket names
        if show_percentile and grid_type == 'percentile':
            percentile_columns, percentile_bound_lows, percentile_bound_ups = \
                _make_bucket_column_names_percentile(percentile_info=percentile_info, endpoint=endpoint)
            percentile_columns = np.array(percentile_columns)[range(uni_xs[0], uni_xs[-1] + 1)]
            percentile_bound_lows = np.array(percentile_bound_lows)[range(uni_xs[0], uni_xs[-1] + 1)]
            percentile_bound_ups = np.array(percentile_bound_ups)[range(uni_xs[0], uni_xs[-1] + 1)]

        # print('percentile_columns',percentile_columns)
        # print('percentile_bound_ups', percentile_bound_ups)
        # print('percentile_bound_lows', percentile_bound_lows)

        # adjust results
        data_x['x'] = data_x['x'] - data_x['x'].min()
        # print('data_x\n', data_x)

    if feature_type == 'onehot':    # 默认不执行这个
        # print("feature_type == 'onehot'")
        feature_grids = display_columns = np.array(feature)
        data_x['x'] = data_x[feature].apply(lambda x: _find_onehot_actual(x=x), axis=1)
        data_x = data_x[~data_x['x'].isnull()].reset_index(drop=True)

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

    print("pdpbox.info_plot_utils._draw_barplot".center(100, "="))
    #
    font_family = plot_params.get('font_family', 'Times New Roman')
    # 柱状图颜色
    # bar_color = plot_params.get('bar_color', '#5BB573')
    bar_color = plot_params.get('bar_color', '#5089C6')
    # 柱状图的宽度
    # bar_width = plot_params.get('bar_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))
    bar_width = plot_params.get('bar_width', np.min([0.3, 0.3 / (10.0 / len(display_columns))]))

    # add value label for bar plot
    # alpha 不透明度
    # rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.5)
    rects = bar_ax.bar(x=bar_data['x'], height=bar_data['fake_count'], width=bar_width, color=bar_color, alpha=0.8)
    _autolabel(rects=rects, ax=bar_ax, bar_color=bar_color)
    _axes_modify(font_family=font_family, ax=bar_ax)


def _draw_lineplot(line_data, line_ax, line_color, plot_params):
    """Draw line plot"""

    font_family = plot_params.get('font_family', 'Times New Roman')
    line_width = plot_params.get('line_width', 1)

    line_ax.plot(line_data['x'], line_data['y'], linewidth=line_width, c=line_color, marker='o')
    for idx in line_data.index.values:
        bbox_props = {'facecolor': line_color, 'edgecolor': 'none', 'boxstyle': "square,pad=0.5"}
        line_ax.text(line_data.loc[idx, 'x'], line_data.loc[idx, 'y'],
                     '%.3f' % line_data.loc[idx, 'y'],
                     ha="center", va="top", size=10, bbox=bbox_props, color='#ffffff')

    _axes_modify(font_family=font_family, ax=line_ax)


def _draw_bar_line(bar_data, bar_ax, line_data, line_ax, line_color, feature_name, display_columns,
                   percentile_columns, plot_params, target_ylabel):
    """Draw bar and line plot"""

    font_family = plot_params.get('font_family', 'Times New Roman')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = bar_ax.twiny()
        percentile_ax.set_xticks(bar_ax.get_xticks())
        percentile_ax.set_xbound(bar_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)

    _draw_lineplot(line_data=line_data, line_ax=line_ax, line_color=line_color, plot_params=plot_params)
    line_ax.get_yaxis().tick_right()
    line_ax.grid(False)
    line_ax.set_ylabel(target_ylabel)


def _target_plot(feature_name, display_columns, percentile_columns, target, bar_data,
                 target_lines, figsize, ncols, plot_params):
    """Internal call for target_plot"""

    # set up graph parameters
    width, height = 15, 9
    nrows = 1

    if len(target) > 1:
        nrows = int(np.ceil(len(target) * 1.0 / ncols))
        ncols = np.min([len(target), ncols])
        width = np.min([7.5 * len(target), 15])
        height = width * 1.0 / ncols * nrows

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    # font_family = plot_params.get('font_family', 'Times New Roman')
    line_color = plot_params.get('line_color', '#1A4E5D')
    line_colors_cmap = plot_params.get('line_colors_cmap', 'tab20')
    line_colors = plot_params.get('line_colors', plt.get_cmap(line_colors_cmap)(
        range(np.min([20, len(target)]))))
    title = plot_params.get('title', 'Target plot for feature "%s"' % feature_name)
    subtitle = plot_params.get('subtitle', 'Average target value through different feature values.')

    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)
    _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)

    bar_line_params = {'bar_data': bar_data, 'feature_name': feature_name, 'display_columns': display_columns,
                       'percentile_columns': percentile_columns, 'plot_params': plot_params}

    if len(target) == 1:
        bar_ax = plt.subplot(outer_grid[1])
        fig.add_subplot(bar_ax)
        line_ax = bar_ax.twinx()

        line_data = target_lines[0].rename(columns={target[0]: 'y'}).sort_values('x', ascending=True)
        _draw_bar_line(bar_ax=bar_ax, line_data=line_data, line_ax=line_ax, line_color=line_color,
                       target_ylabel='Average %s' % target[0], **bar_line_params)

    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35)
        plot_axes = []
        for inner_idx in range(len(target)):
            ax = plt.subplot(inner_grid[inner_idx])
            plot_axes.append(ax)
            fig.add_subplot(ax)

        bar_ax = []
        line_ax = []

        # get max average target value
        ys = []
        for target_idx in range(len(target)):
            ys += list(target_lines[target_idx][target[target_idx]].values)
        y_max = np.max(ys)

        for target_idx in range(len(target)):
            inner_line_color = line_colors[target_idx % len(line_colors)]
            inner_bar_ax = plot_axes[target_idx]
            inner_line_ax = inner_bar_ax.twinx()

            line_data = target_lines[target_idx].rename(
                columns={target[target_idx]: 'y'}).sort_values('x', ascending=True)

            _draw_bar_line(
                bar_ax=inner_bar_ax, line_data=line_data, line_ax=inner_line_ax, line_color=inner_line_color,
                target_ylabel='Average %s' % target[target_idx], **bar_line_params)

            inner_line_ax.set_ylim(0., y_max)
            bar_ax.append(inner_bar_ax)
            line_ax.append(inner_line_ax)

            if target_idx % ncols != 0:
                inner_bar_ax.set_yticklabels([])
                inner_bar_ax.tick_params(which="major", left=False)

            if (target_idx % ncols + 1 != ncols) and target_idx != len(plot_axes) - 1:
                inner_line_ax.set_yticklabels([])
                inner_line_ax.tick_params(which="major", right=False)

        if len(plot_axes) > len(target):
            for idx in range(len(target), len(plot_axes)):
                plot_axes[idx].axis('off')

    axes = {'title_ax': title_ax, 'bar_ax': bar_ax, 'line_ax': line_ax}
    return fig, axes


def _draw_boxplot(box_data, box_line_data, box_ax, display_columns, box_color, plot_params):
    """Draw box plot"""

    print("pdpbox.info_plot_utils._draw_boxplot".center(100, "="))

    font_family = plot_params.get('font_family', 'Times New Roman')
    # 箱型图的线宽
    # box_line_width = plot_params.get('box_line_width', 1.5)
    box_line_width = plot_params.get('box_line_width', 1.0)
    # 箱型图的宽度
    # box_width = plot_params.get('box_width', np.min([0.4, 0.4 / (10.0 / len(display_columns))]))
    box_width = plot_params.get('box_width', np.min([0.3, 0.3 / (10.0 / len(display_columns))]))

    xs = sorted(box_data['x'].unique())
    ys = []
    for x in xs:
        ys.append(box_data[box_data['x'] == x]['y'].values)

    boxprops = dict(linewidth=box_line_width, color=box_color)
    medianprops = dict(linewidth=0)
    whiskerprops = dict(linewidth=box_line_width, color=box_color)
    capprops = dict(linewidth=box_line_width, color=box_color)

    box_ax.boxplot(ys, positions=xs, showfliers=False, widths=box_width, whiskerprops=whiskerprops, capprops=capprops,
                   boxprops=boxprops, medianprops=medianprops)

    _axes_modify(font_family=font_family, ax=box_ax)

    box_ax.plot(box_line_data['x'], box_line_data['y'], linewidth=1, c=box_color, linestyle='--')
    # 箱型图中间的Q2值，默认size=10 lw=1
    for idx in box_line_data.index.values:
        # bbox_props = {'facecolor': 'white', 'edgecolor': box_color, 'boxstyle': "square,pad=0.5", 'lw': 1}
        bbox_props = {'facecolor': 'white', 'edgecolor': box_color, 'boxstyle': "square,pad=0.1", 'lw': 1}
        box_ax.text(box_line_data.loc[idx, 'x'], box_line_data.loc[idx, 'y'], '%.3f' % box_line_data.loc[idx, 'y'],
                    # ha="center", va="top", size=10, bbox=bbox_props, color=box_color)
                    ha="center", va="center", size=10, bbox=bbox_props, color=box_color)


def _draw_box_bar(bar_data, bar_ax, box_data, box_line_data, box_color, box_ax,
                  feature_name, display_columns, percentile_columns, plot_params, target_ylabel):
    """Draw box plot and bar plot"""
    print("pdpbox.info_plot_utils._draw_box_bar".center(100, "="))

    font_family = plot_params.get('font_family', 'Times New Roman')
    xticks_rotation = plot_params.get('xticks_rotation', 0)

    _draw_boxplot(box_data=box_data, box_line_data=box_line_data, box_ax=box_ax,
                  display_columns=display_columns, box_color=box_color, plot_params=plot_params)
    box_ax.set_ylabel('%sprediction dist' % target_ylabel)
    box_ax.set_xticklabels([])

    _draw_barplot(bar_data=bar_data, bar_ax=bar_ax, display_columns=display_columns, plot_params=plot_params)

    # bar plot
    bar_ax.set_xlabel(feature_name)
    bar_ax.set_ylabel('count')

    bar_ax.set_xticks(range(len(display_columns)))
    bar_ax.set_xticklabels(display_columns, rotation=xticks_rotation)
    bar_ax.set_xlim(-0.5, len(display_columns) - 0.5)

    plt.setp(box_ax.get_xticklabels(), visible=False)

    # display percentile
    if len(percentile_columns) > 0:
        percentile_ax = box_ax.twiny()
        percentile_ax.set_xticks(box_ax.get_xticks())
        percentile_ax.set_xbound(box_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns, rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)


def _actual_plot(plot_data, bar_data, box_lines, actual_prediction_columns, feature_name,
                 display_columns, percentile_columns, figsize, ncols, plot_params):
    """Internal call for actual_plot"""

    print("pdpbox.info_plot_utils._actual_plot".center(100, "="))

    # set up graph parameters
    width, height = 15, 9
    # width, height = 8, 5
    nrows = 1

    if plot_params is None:    # 默认执行这个
        plot_params = dict()

    font_family = plot_params.get('font_family', 'Times New Roman')
    print('font_family', font_family)
    # 箱型图颜色
    # box_color = plot_params.get('box_color', '#3288bd')
    box_color = plot_params.get('box_color', '#0F52BA')
    box_colors_cmap = plot_params.get('box_colors_cmap', 'tab20')
    box_colors = plot_params.get('box_colors', plt.get_cmap(box_colors_cmap)(
        range(np.min([20, len(actual_prediction_columns)]))))
    title = plot_params.get('title', 'Actual predictions plot for %s' % feature_name)
    subtitle = plot_params.get('subtitle', 'Distribution of actual prediction through different feature values.')

    if len(actual_prediction_columns) >= 1:
        nrows = int(np.ceil(len(actual_prediction_columns) * 1.0 / ncols))   # np.ceil():向上取整
        ncols = np.min([len(actual_prediction_columns), ncols])
        width = np.min([7.5 * len(actual_prediction_columns) + len(display_columns), 15])
        height = width * 1.0 / ncols * nrows
        # print("nrows:\n", nrows)
        # print("ncols:\n", ncols)
        # print("width:\n", width)
        # print("height:\n", height)

    if figsize is not None:
        width, height = figsize

    if plot_params is None:
        plot_params = dict()

    fig = plt.figure(figsize=(width, height))
    # outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height-2])
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[1, 1])  # height_ratios:两个子图的高度比例
    # print(outer_grid[0], outer_grid[1])
    # 添加标题
    # title_ax = plt.subplot(outer_grid[0])
    # fig.add_subplot(title_ax)
    # _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)

    box_bar_params = {'bar_data': bar_data, 'feature_name': feature_name, 'display_columns': display_columns,
                      'percentile_columns': percentile_columns

        , 'plot_params': plot_params}

    print("bar_data:\n", bar_data)
    print("feature_name:\n", feature_name)
    print("display_columns:\n", display_columns)
    print("percentile_columns:\n", percentile_columns)
    print("plot_params:\n", plot_params)

    if len(actual_prediction_columns) == 1:  # 默认这个
        print('len(actual_prediction_columns) == 1')
        # inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.2)
        print('inner_grid', inner_grid)

        box_ax = plt.subplot(inner_grid[0])
        print('box_ax', box_ax)
        bar_ax = plt.subplot(inner_grid[1], sharex=box_ax)
        fig.add_subplot(box_ax)
        fig.add_subplot(bar_ax)

        if actual_prediction_columns[0] == 'actual_prediction':  # 默认这个
            target_ylabel = ''
        else:
            target_ylabel = 'target_%s: ' % actual_prediction_columns[0].split('_')[-1]

        print('plot_data', plot_data)
        box_data = plot_data[['x', actual_prediction_columns[0]]].rename(columns={actual_prediction_columns[0]: 'y'})
        print('box_data', box_data)
        print('box_lines', box_lines )
        box_line_data = box_lines[0].rename(columns={actual_prediction_columns[0] + '_q2': 'y'})

        print("box_line_data:\n", box_line_data)
        print("target_ylabel:\n", target_ylabel)
        print("box_bar_params:\n", box_bar_params)

        _draw_box_bar(bar_ax=bar_ax, box_data=box_data, box_line_data=box_line_data, box_color=box_color,
                      box_ax=box_ax, target_ylabel=target_ylabel, **box_bar_params)
    else:
        inner_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.35)

        box_ax = []
        bar_ax = []

        # get max average target value
        ys = []
        for idx in range(len(box_lines)):
            ys += list(box_lines[idx][actual_prediction_columns[idx] + '_q2'].values)
        y_max = np.max(ys)

        for idx in range(len(actual_prediction_columns)):
            box_color = box_colors[idx % len(box_colors)]

            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[idx], wspace=0, hspace=0.2)
            inner_box_ax = plt.subplot(inner[0])
            inner_bar_ax = plt.subplot(inner[1], sharex=inner_box_ax)
            fig.add_subplot(inner_box_ax)
            fig.add_subplot(inner_bar_ax)

            inner_box_data = plot_data[['x', actual_prediction_columns[idx]]].rename(
                columns={actual_prediction_columns[idx]: 'y'})
            inner_box_line_data = box_lines[idx].rename(columns={actual_prediction_columns[idx] + '_q2': 'y'})
            _draw_box_bar(bar_ax=inner_bar_ax, box_data=inner_box_data,
                          box_line_data=inner_box_line_data, box_color=box_color, box_ax=inner_box_ax,
                          target_ylabel='target_%s: ' % actual_prediction_columns[idx].split('_')[-1], **box_bar_params)

            inner_box_ax.set_ylim(0., y_max)

            if idx % ncols != 0:
                inner_bar_ax.set_yticklabels([])
                inner_box_ax.set_yticklabels([])

            box_ax.append(inner_box_ax)
            bar_ax.append(inner_bar_ax)

    # axes = {'title_ax': title_ax, 'box_ax': box_ax, 'bar_ax': bar_ax}
    axes = {'box_ax': box_ax, 'bar_ax': bar_ax}

    return fig, axes


def _plot_interact(plot_data, y, plot_ax, feature_names, display_columns, percentile_columns,
                   marker_sizes, cmap, line_width, xticks_rotation, font_family, annotate):
    """Interact scatter plot"""

    plot_ax.set_xticks(range(len(display_columns[0])))
    plot_ax.set_xticklabels(display_columns[0], rotation=xticks_rotation, ha='right')  # 标签旋转后与中心刻度错位，修改ha参数
    plot_ax.set_yticks(range(len(display_columns[1])))
    plot_ax.set_yticklabels(display_columns[1])
    plot_ax.set_xlim(-0.5, len(display_columns[0]) - 0.5)
    plot_ax.set_ylim(-0.5, len(display_columns[1]) - 0.5)

    percentile_ax = None
    percentile_ay = None
    # display percentile
    if len(percentile_columns[0]) > 0:
        percentile_ax = plot_ax.twiny()
        percentile_ax.set_xticks(plot_ax.get_xticks())
        percentile_ax.set_xbound(plot_ax.get_xbound())
        percentile_ax.set_xticklabels(percentile_columns[0], rotation=xticks_rotation)
        percentile_ax.set_xlabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ax, top=True)
        percentile_ax.grid(False)

    # display percentile
    if len(percentile_columns[1]) > 0:
        percentile_ay = plot_ax.twinx()
        percentile_ay.set_yticks(plot_ax.get_yticks())
        percentile_ay.set_ybound(plot_ax.get_ybound())
        percentile_ay.set_yticklabels(percentile_columns[1])
        percentile_ay.set_ylabel('percentile buckets')
        _axes_modify(font_family=font_family, ax=percentile_ay, right=True)
        percentile_ay.grid(False)

    # value_min, value_max = plot_data[y].min(), plot_data[y].max()
    ##################################### 2021-10-12-修改legend最大最小值 #####################################
    value_min, value_max = 0, 1
    colors = [plt.get_cmap(cmap)(float(v - value_min) / (value_max - value_min)) for v in plot_data[y].values]

    # print("line_width:\n", line_width)
    # print("colors:\n", colors)

    # line_width 表示圈圈外面的线
    plot_ax.scatter(plot_data['x1'].values, plot_data['x2'].values, s=marker_sizes, c=colors,
                    linewidth=line_width, edgecolors=plt.get_cmap(cmap)(1.))
    """原始的
    if annotate:
        for text_idx in range(plot_data.shape[0]):
            plot_data_idx = plot_data.iloc[text_idx]
            text_s = '%d\n%.3f' % (plot_data_idx['fake_count'], plot_data_idx[y])
            txt = plot_ax.text(x=plot_data_idx['x1'], y=plot_data_idx['x2'], s=text_s,
                               fontdict={'family': font_family, 'color': plt.get_cmap(cmap)(1.), 'size': 11},
                               va='center', ha='left')
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    """
    ##################################### 2021-10-12-修改annotate的标签 #####################################
    if annotate:
        for text_idx in range(plot_data.shape[0]):
            plot_data_idx = plot_data.iloc[text_idx]
            text_s = '%d\n%.2f' % (plot_data_idx['fake_count'], plot_data_idx[y])
            txt = plot_ax.text(x=plot_data_idx['x1'], y=plot_data_idx['x2'], s=text_s,
                               # ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys']
                               # fontdict={'family': font_family, 'color': plt.get_cmap("Blues")(1.), 'size': 10},

                               fontdict={'family': font_family, 'color': "black", 'size': 10, "weight": "bold"},###  圆圈里面数字的大小
                               va='center', ha='center')
            # 圈圈里面的字在这里加特效
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            # txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

    plot_ax.set_xlabel(feature_names[0])
    plot_ax.set_ylabel(feature_names[1])
    _axes_modify(font_family=font_family, ax=plot_ax)   ##  跳转到可修改坐标轴大小的函数
    return value_min, value_max, percentile_ax, percentile_ay


def _plot_legend_colorbar(value_min, value_max, colorbar_ax, cmap, font_family, height="50%", width="50%"):
    """Plot colorbar legend"""

    norm = mpl.colors.Normalize(vmin=float(value_min), vmax=float(value_max))

    # color bar
    cax = inset_axes(colorbar_ax, height=height, width=width, loc=10)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm, orientation='horizontal')
    cb.outline.set_linewidth(0)
    cb.set_ticks([])

    width_float = float(width.replace('%', '')) / 100
    text_params = {'fontdict': {'color': plt.get_cmap(cmap)(1.), 'fontsize': 10},
                   'transform': colorbar_ax.transAxes, 'va': 'center'}
    tmin = cb.ax.text((1. - width_float) / 2, 0.5, '%.3f ' % value_min, ha='left', **text_params)
    tmin.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    tmax = cb.ax.text(1. - (1. - width_float) / 2, 0.5, '%.3f ' % value_max, ha='right', **text_params)
    tmax.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
    _modify_legend_ax(colorbar_ax, font_family)


def _plot_legend_circles(count_min, count_max, circle_ax, cmap, font_family):
    """Plot circle legend"""
    # circle size
    # circle_ax.plot([0.75, 2], [1] * 2, color=plt.get_cmap(cmap)(1.), zorder=1, ls='--')
    # circle_ax.scatter([0.75, 2], [1] * 2, s=[50, 500], edgecolors=plt.get_cmap(cmap)(1.), color='white', zorder=2)
    # circle_ax.set_xlim(0., 3)
    #
    # text_params = {'fontdict': {'color': '#424242', 'fontsize': 10}, 'ha': 'center', 'va': 'center'}
    # circle_ax.text(0.75, 1., count_min, **text_params)
    # circle_ax.text(2, 1., count_max, **text_params)
    # _modify_legend_ax(circle_ax, font_family)


    ##  修改后
    # circle_ax.plot([0.25, 1], [1] * 2, color=plt.get_cmap(cmap)(1.), zorder=1, ls='--')
    # circle_ax.scatter([0.25, 1], [1] * 2, s=[500, 1500], edgecolors=plt.get_cmap(cmap)(1.), color='white', zorder=2)
    # circle_ax.set_xlim(0., 3)

    text_params = {'fontdict': {'color': '#424242', 'fontsize': 10}, 'ha': 'center', 'va': 'center'}
    # circle_ax.text(0.25, 1, count_min, **text_params)
    # circle_ax.text(1, 1, count_max, **text_params)
    _modify_legend_ax(circle_ax, font_family)


def _info_plot_interact(feature_names, display_columns, percentile_columns, ys, plot_data, title, subtitle,
                        figsize, ncols, annotate, plot_params, is_target_plot=True):
    """Internal call for _info_plot_interact"""

    # print("pdpbox.info_plot_utils._info_plot_interact".center(100, "="))

    # print("feature_names\n", feature_names)
    # print("display_columns\n", display_columns)
    # print("plot_data\n", plot_data)
    # exit(0)

    width, height = 15, 10
    # width, height = 8, 9
    nrows = 1

    # 根据 target的数量来画图
    # if len(ys) > 1:
    if len(ys) >= 1:
        nrows = int(np.ceil(len(ys) * 1.0 / ncols))
        ncols = np.min([len(ys), ncols])
    #     # width = np.min([7.5 * len(ys), 15])
    #     width = np.min([8 * len(ys), 15])
    #     height = width * 1.2 / ncols * nrows
    #     print("nrows:\n", nrows)
    #     print("ncols:\n", ncols)
    #     print("width:\n", width)
    #     print("height:\n", height)

    # 修改为根据横纵坐标数据量来
    # print(len(max(display_columns[0])), max(display_columns[0]))
    width = np.max([1.2 * len(display_columns[0]) + 0.2 * len(max(display_columns[0])), 10])
    height = np.max([1.5 * len(display_columns[1]) + 0.1 * len(max(display_columns[0])), 6.4])
    # print("width:\n", width)
    # print("height:\n", height)

    if figsize is not None:
        width, height = figsize

    font_family = plot_params.get('font_family', 'Times New Roman')
    cmap = plot_params.get('cmap', 'Blues')
    cmaps = plot_params.get('cmaps', ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples', 'Greys'])
    line_width = plot_params.get('line_width', 1)
    xticks_rotation = plot_params.get('xticks_rotation', 0)
    marker_size_min = plot_params.get('marker_size_min', 50)
    marker_size_max = plot_params.get('marker_size_max', 1500)

    # plot title
    fig = plt.figure(figsize=(width, height))
    outer_grid = GridSpec(2, 1, wspace=0.0, hspace=0.1, height_ratios=[2, height - 2])
    title_ax = plt.subplot(outer_grid[0])
    fig.add_subplot(title_ax)

    _plot_title(title=title, subtitle=subtitle, title_ax=title_ax, plot_params=plot_params)

    # draw value plots and legend
    count_min, count_max = plot_data['fake_count'].min(), plot_data['fake_count'].max()
    marker_sizes = []
    for count in plot_data['fake_count'].values:
        size = float(count - count_min) / (count_max - count_min) * (
                    marker_size_max - marker_size_min) + marker_size_min
        marker_sizes.append(size)

    interact_params = {'plot_data': plot_data, 'feature_names': feature_names, 'display_columns': display_columns,
                       'percentile_columns': percentile_columns, 'marker_sizes': marker_sizes, 'line_width': line_width,
                       'xticks_rotation': xticks_rotation, 'font_family': font_family, 'annotate': annotate}
    if len(ys) == 1:
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1], height_ratios=[height - 3, 1],
                                             hspace=0.25)
        value_ax = plt.subplot(inner_grid[0])
        fig.add_subplot(value_ax)

        # print("interact_params:\n", interact_params)

        value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
            y=ys[0], plot_ax=value_ax, cmap=cmap, **interact_params)

        # draw legend
        legend_grid = GridSpecFromSubplotSpec(1, 4, subplot_spec=inner_grid[1], wspace=0)
        legend_ax = [plt.subplot(legend_grid[0]), plt.subplot(legend_grid[1])]
        fig.add_subplot(legend_ax[0])
        fig.add_subplot(legend_ax[1])

        _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=legend_ax[0],
                              cmap=cmap, font_family=font_family)
        _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=legend_ax[1],
                             cmap=cmap, font_family=font_family)

        if is_target_plot:
            subplot_title = 'Average %s' % ys[0]
        elif ys[0] == 'actual_prediction_q2':
            subplot_title = 'Median Prediction'
        else:
            subplot_title = 'target_%s: median prediction' % ys[0].split('_')[-2]
        if len(percentile_columns[0]) > 0:
            subplot_title += '\n\n\n'
        value_ax.set_title(subplot_title, fontdict={'fontsize': 11, 'fontname': font_family})
    else:
        value_grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[1], wspace=0.2, hspace=0.2)
        value_ax = []
        legend_ax = []

        for idx in range(len(ys)):
            inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=value_grid[idx], height_ratios=[7, 1], hspace=0.3)
            inner_value_ax = plt.subplot(inner_grid[0])
            fig.add_subplot(inner_value_ax)
            value_ax.append(inner_value_ax)

            cmap_idx = cmaps[idx % len(cmaps)]
            value_min, value_max, percentile_ax, percentile_ay = _plot_interact(
                y=ys[idx], plot_ax=inner_value_ax, cmap=cmap_idx, **interact_params)

            # draw legend
            inner_legend_grid = GridSpecFromSubplotSpec(1, 4, subplot_spec=inner_grid[1], wspace=0)
            inner_legend_ax = [plt.subplot(inner_legend_grid[0]), plt.subplot(inner_legend_grid[1])]
            fig.add_subplot(inner_legend_ax[0])
            fig.add_subplot(inner_legend_ax[1])
            _plot_legend_colorbar(value_min=value_min, value_max=value_max, colorbar_ax=inner_legend_ax[0],
                                  cmap=cmap_idx, font_family=font_family)
            _plot_legend_circles(count_min=count_min, count_max=count_max, circle_ax=inner_legend_ax[1],
                                 cmap=cmap_idx, font_family=font_family)
            legend_ax.append(inner_legend_ax)

            if is_target_plot:
                subplot_title = 'Average %s' % ys[idx]
            else:
                subplot_title = 'target_%s: median prediction' % ys[idx].split('_')[-2]
            if len(percentile_columns[0]) > 0:
                subplot_title += '\n\n\n'
            inner_value_ax.set_title(subplot_title, fontdict={'fontsize': 11, 'fontname': font_family})

            if idx % ncols != 0:
                inner_value_ax.set_yticklabels([])

            if (idx % ncols + 1 != ncols) and idx != len(value_ax) - 1:
                if percentile_ay is not None:
                    percentile_ay.set_yticklabels([])

        if len(value_ax) > len(ys):
            for idx in range(len(ys), len(value_ax)):
                value_ax[idx].axis('off')

    axes = {'title_ax': title_ax, 'value_ax': value_ax, 'legend_ax': legend_ax}
    return fig, axes


def _prepare_info_plot_data(feature, feature_type, data, num_grid_points, grid_type, percentile_range,
                            grid_range, cust_grid_points, show_percentile, show_outliers, endpoint):
    """Prepare data for information plots"""
    prepared_results = _prepare_data_x(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points, grid_type=grid_type,
        percentile_range=percentile_range, grid_range=grid_range, cust_grid_points=cust_grid_points,
        show_percentile=show_percentile, show_outliers=show_outliers, endpoint=endpoint)
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
        print('summary_df\n', summary_df)
        info_cols += ['percentile_column', 'percentile_lower', 'percentile_upper']

    return data_x, bar_data, summary_df, info_cols, display_columns, percentile_columns


def _prepare_info_plot_interact_data(data_input, features, feature_types, num_grid_points, grid_types,
                                     percentile_ranges, grid_ranges, cust_grid_points, show_percentile,
                                     show_outliers, endpoint, agg_dict):
    """Prepare data for information interact plots"""
    prepared_results = []
    for i in range(2):
        prepared_result = _prepare_data_x(
            feature=features[i], feature_type=feature_types[i], data=data_input,
            num_grid_points=num_grid_points[i], grid_type=grid_types[i], percentile_range=percentile_ranges[i],
            grid_range=grid_ranges[i], cust_grid_points=cust_grid_points[i],
            show_percentile=show_percentile, show_outliers=show_outliers[i], endpoint=endpoint)
        prepared_results.append(prepared_result)
        if i == 0:
            data_input = prepared_result['data'].rename(columns={'x': 'x1'})

    data_x = prepared_results[1]['data'].rename(columns={'x': 'x2'})
    data_x['fake_count'] = 1
    # print("data_x\n", data_x[(data_x["x1"] == 2) & (data_x["x2"] == 0)])
    # print("data_x\n", data_x[data_x["x1"] == 2])
    # data_x.to_csv("F:\ZZH\Bi-LSTM\CODE\LSTMVis\lookLookAndDelete.csv", index=None)
    plot_data = data_x.groupby(['x1', 'x2'], as_index=False).agg(agg_dict)
    # print("plot_data", plot_data)
    # print("data_x", data_x)
    # exit(0)
    return data_x, plot_data, prepared_results


def _prepare_info_plot_interact_summary(data_x, plot_data, prepared_results, feature_types):
    """Prepare summary data frame for interact plots"""
    # print(data_x)
    x1_values = []
    x2_values = []
    for x1_value in range(data_x['x1'].min(), data_x['x1'].max() + 1):
        for x2_value in range(data_x['x2'].min(), data_x['x2'].max() + 1):
            x1_values.append(x1_value)
            x2_values.append(x2_value)
    summary_df = pd.DataFrame()
    summary_df['x1'] = x1_values
    summary_df['x2'] = x2_values
    summary_df = summary_df.merge(plot_data.rename(columns={'fake_count': 'count'}),
                                  on=['x1', 'x2'], how='left').fillna(0)

    info_cols = ['x1', 'x2', 'display_column_1', 'display_column_2']
    display_columns = []
    percentile_columns = []
    for i in range(2):
        display_columns_i, bound_lows_i, bound_ups_i = prepared_results[i]['value_display']
        percentile_columns_i, percentile_bound_lows_i, percentile_bound_ups_i = prepared_results[i][
            'percentile_display']
        display_columns.append(display_columns_i)
        percentile_columns.append(percentile_columns_i)

        summary_df['display_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
            lambda x: display_columns_i[int(x)])
        if feature_types[i] == 'numeric':
            summary_df['value_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_lows_i[int(x)])
            summary_df['value_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(lambda x: bound_ups_i[int(x)])
            info_cols += ['value_lower_%d' % (i + 1), 'value_upper_%d' % (i + 1)]

        if len(percentile_columns_i) != 0:
            summary_df['percentile_column_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_columns_i[int(x)])
            summary_df['percentile_lower_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_lows_i[int(x)])
            summary_df['percentile_upper_%d' % (i + 1)] = summary_df['x%d' % (i + 1)].apply(
                lambda x: percentile_bound_ups_i[int(x)])
            info_cols += ['percentile_column_%d' % (i + 1), 'percentile_lower_%d' % (i + 1),
                          'percentile_upper_%d' % (i + 1)]

    return summary_df, info_cols, display_columns, percentile_columns


def _check_info_plot_params(df, feature, grid_type, percentile_range, grid_range,
                            cust_grid_points, show_outliers):
    """Check information plot parameters"""

    _check_dataset(df=df)    # check_dataset检查数据集,如果没找到数据集则下载数据集
    feature_type = _check_feature(feature=feature, df=df)
    _check_grid_type(grid_type=grid_type)
    _check_percentile_range(percentile_range=percentile_range)

    # show_outliers should be only turned on when necessary
    if (percentile_range is None) and (grid_range is None) and (cust_grid_points is None):
        show_outliers = False
    return feature_type, show_outliers


def _check_info_plot_interact_params(num_grid_points, grid_types, percentile_ranges, grid_ranges, cust_grid_points,
                                     show_outliers, plot_params, features, df):
    """Check interact information plot parameters"""

    _check_dataset(df=df)
    num_grid_points = _expand_default(num_grid_points, 10)
    grid_types = _expand_default(grid_types, 'percentile')
    _check_grid_type(grid_type=grid_types[0])
    _check_grid_type(grid_type=grid_types[1])

    percentile_ranges = _expand_default(percentile_ranges, None)
    _check_percentile_range(percentile_range=percentile_ranges[0])
    _check_percentile_range(percentile_range=percentile_ranges[1])

    grid_ranges = _expand_default(grid_ranges, None)
    cust_grid_points = _expand_default(cust_grid_points, None)

    if not show_outliers:
        show_outliers = [False, False]
    else:
        show_outliers = [True, True]
        for i in range(2):
            if (percentile_ranges[i] is None) and (grid_ranges[i] is None) and (cust_grid_points[i] is None):
                show_outliers[i] = False

    # set up graph parameters
    if plot_params is None:
        plot_params = dict()

    # check features
    feature_types = [_check_feature(feature=features[0], df=df), _check_feature(feature=features[1], df=df)]

    return {
        'num_grid_points': num_grid_points,
        'grid_types': grid_types,
        'percentile_ranges': percentile_ranges,
        'grid_ranges': grid_ranges,
        'cust_grid_points': cust_grid_points,
        'show_outliers': show_outliers,
        'plot_params': plot_params,
        'feature_types': feature_types
    }





def target_plot(df, feature, feature_name, target, num_grid_points=10, grid_type='percentile',
                percentile_range=None, grid_range=None, cust_grid_points=None, show_percentile=False,
                show_outliers=False, endpoint=True, figsize=None, ncols=2, plot_params=None):
    """
    Plot average target value across different feature values (feature grids)

    Parameters
    ----------
    df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    feature: string or list
        feature or feature list to investigate,
        for one-hot encoding features, feature list is required
    feature_name: string
        name of the feature, not necessary a column name
    target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal'
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points
        for numeric feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets
        for numeric feature when grid_type='percentile'
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    endpoint: bool, optional, default=True
        If True, stop is the last grid point
        Otherwise, it is not included
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    plot_params: dict or None, optional, default=None
        parameters for the plot

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with target_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']
        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature='Sex', feature_name='Sex', target=titanic_target)


    With One-hot encoding features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature=['Embarked_C', 'Embarked_Q', 'Embarked_S'],
            feature_name='Embarked', target=titanic_target)


    With numeric features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.target_plot(
            df=titanic_data, feature='Fare', feature_name='Fare',
            target=titanic_target, show_percentile=True)


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_target = test_otto['target']
        fig, axes, summary_df = info_plots.target_plot(
            df=otto_data, feature='feat_67', feature_name='feat_67',
            target=['target_0', 'target_2', 'target_5', 'target_8'])

    """

    # check inputs
    _ = _check_target(target=target, df=df)
    feature_type, show_outliers = _check_info_plot_params(
        df=df, feature=feature, grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_outliers=show_outliers)

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = _make_list(feature) + target

    # map feature values to grid point buckets (x)
    data = df[useful_features]
    data_x, bar_data, summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=data, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile, show_outliers=show_outliers,
        endpoint=endpoint)
    print('--------------------------------------------------------------')
    print('data_x_new\n', data_x)

    # prepare data for target lines
    # each target line contains 'x' and mean target value
    target_lines = []
    for target_idx in range(len(target)):   # target:feature_name
        target_line = data_x.groupby('x', as_index=False).agg(
            {target[target_idx]: 'mean'}).sort_values('x', ascending=True)
        target_lines.append(target_line)
        summary_df = summary_df.merge(target_line, on='x', how='outer')
    summary_df = summary_df[info_cols + ['count'] + target]

    # inner call target plot
    fig, axes = _target_plot(
        feature_name=feature_name, display_columns=display_columns, percentile_columns=percentile_columns,
        target=target, bar_data=bar_data, target_lines=target_lines, figsize=figsize, ncols=ncols,
        plot_params=plot_params)

    return fig, axes, summary_df


def q1(x):
    return x.quantile(0.25)


def q2(x):
    return x.quantile(0.5)


def q3(x):
    return x.quantile(0.75)


def actual_plot(model, X, feature, feature_name, num_grid_points=10, grid_type='percentile', percentile_range=None,
                grid_range=None, cust_grid_points=None, show_percentile=False, show_outliers=False, endpoint=True,
                which_classes=None, predict_kwds={}, ncols=2, figsize=None, plot_params=None):
    """Plot prediction distribution across different feature values (feature grid)

    Parameters
    ----------

    model: a fitted sklearn model
    X: pandas DataFrame
        data set on which the model is trained
    feature: string or list
        feature or feature list to investigate
        for one-hot encoding features, feature list is required
    feature_name: string
        name of the feature, not necessary a column name
    num_grid_points: integer, optional, default=10
        number of grid points for numeric feature
    grid_type: string, optional, default='percentile'
        'percentile' or 'equal',
        type of grid points for numeric feature
    percentile_range: tuple or None, optional, default=None
        percentile range to investigate,
        for numeric feature when grid_type='percentile'
    grid_range: tuple or None, optional, default=None
        value range to investigate,
        for numeric feature when grid_type='equal'
    cust_grid_points: Series, 1d-array, list or None, optional, default=None
        customized list of grid points for numeric feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets,
        for numeric feature when grid_type='percentile'
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets
        for numeric feature when percentile_range or grid_range is not None
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    which_classes: list, optional, default=None
        which classes to plot, only use when it is a multi-class problem
    predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    plot_params: dict or None, optional, default=None
        parameters for the plot

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with actual_plot

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_features = test_titanic['features']
        titanic_target = test_titanic['target']
        titanic_model = test_titanic['xgb_model']
        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature='Sex', feature_name='Sex')


    With One-hot encoding features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature=['Embarked_C', 'Embarked_Q', 'Embarked_S'], feature_name='Embarked')


    With numeric features

    .. highlight:: python
    .. code-block:: python

        fig, axes, summary_df = info_plots.actual_plot(
            model=titanic_model, X=titanic_data[titanic_features],
            feature='Fare', feature_name='Fare')


    With multi-class

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_otto = get_dataset.otto()
        otto_data = test_otto['data']
        otto_model = test_otto['rf_model']
        otto_features = test_otto['features']
        otto_target = test_otto['target']

        fig, axes, summary_df = info_plots.actual_plot(
            model=otto_model, X=otto_data[otto_features],
            feature='feat_67', feature_name='feat_67', which_classes=[1, 2, 3])

    """

    # check inputs
    n_classes, predict = _check_model(model=model)
    feature_type, show_outliers = _check_info_plot_params(
        df=X, feature=feature, grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_outliers=show_outliers)

    # make predictions
    # info_df only contains feature value and actual predictions
    prediction = predict(X, **predict_kwds)
    info_df = X[_make_list(feature)]
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            _check_classes(classes_list=which_classes, n_classes=n_classes)
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    info_df_x, bar_data, summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint)

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        box_line = info_df_x.groupby('x', as_index=False).agg(
            {actual_prediction_columns[idx]: [q1, q2, q3]}).sort_values('x', ascending=True)
        box_line.columns = ['_'.join(col) if col[1] != '' else col[0] for col in box_line.columns]
        box_lines.append(box_line)
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
        summary_df = summary_df.merge(box_line, on='x', how='outer').fillna(0)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    fig, axes = _actual_plot(plot_data=info_df_x, bar_data=bar_data, box_lines=box_lines,
                             actual_prediction_columns=actual_prediction_columns, feature_name=feature_name,
                             display_columns=display_columns, percentile_columns=percentile_columns, figsize=figsize,
                             ncols=ncols, plot_params=plot_params)
    return fig, axes, summary_df


def target_plot_interact(df, features, feature_names, target, num_grid_points=None, grid_types=None,
                         percentile_ranges=None, grid_ranges=None, cust_grid_points=None, show_percentile=False,
                         show_outliers=False, endpoint=True, figsize=None, ncols=2, annotate=False, plot_params=None, f2v=False):
    """Plot average target value across different feature value combinations (feature grid combinations)

    Parameters
    ----------

    df: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    features: list
        two features to investigate
    feature_names: list
        feature names
    target: string or list
        column name or column name list for target value
        for multi-class problem, a list of one-hot encoding target column
    num_grid_points: list, optional, default=None
        number of grid points for each feature
    grid_types: list, optional, default=None
        type of grid points for each feature
    percentile_ranges: list of tuple, optional, default=None
        percentile range to investigate for each feature
    grid_ranges: list of tuple, optional, default=None
        value range to investigate for each feature
    cust_grid_points: list of (Series, 1d-array, list), optional, default=None
        customized list of grid points for each feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets for both feature
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets for both features
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    annotate: bool, default=False
        whether to annotate the points
    plot_params: dict or None, optional, default=None
        parameters for the plot

    Notes
    -----

    - Parameters are consistent with the ones for function target_plot
    - But for this function, you need to specify parameter value for both features in list format
    - For example:
        - percentile_ranges = [(0, 90), (5, 95)] means
        - percentile_range = (0, 90) for feature 1
        - percentile_range = (5, 95) for feature 2

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Examples
    --------

    Quick start with target_plot_interact

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_target = test_titanic['target']

        fig, axes, summary_df = info_plots.target_plot_interact(
            df=titanic_data, features=['Sex', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
            feature_names=['Sex', 'Embarked'], target=titanic_target)

    """

    # check inputs
    _ = _check_target(target=target, df=df)
    check_results = _check_info_plot_interact_params(
        num_grid_points=num_grid_points, grid_types=grid_types, percentile_ranges=percentile_ranges,
        grid_ranges=grid_ranges, cust_grid_points=cust_grid_points, show_outliers=show_outliers,
        plot_params=plot_params, features=features, df=df)

    num_grid_points = check_results['num_grid_points']
    grid_types = check_results['grid_types']
    percentile_ranges = check_results['percentile_ranges']
    grid_ranges = check_results['grid_ranges']
    cust_grid_points = check_results['cust_grid_points']
    show_outliers = check_results['show_outliers']
    plot_params = check_results['plot_params']
    feature_types = check_results['feature_types']

    # create feature grids and bar counts
    target = _make_list(target)
    useful_features = _make_list(features[0]) + _make_list(features[1]) + target

    # prepare data for bar plot
    data = df[useful_features].copy()

    # prepare data for target interact plot
    agg_dict = {}
    for t in target:
        agg_dict[t] = 'mean'
    agg_dict['fake_count'] = 'count'

    data_x, target_plot_data, prepared_results = _prepare_info_plot_interact_data(
        data_input=data, features=features, feature_types=feature_types, num_grid_points=num_grid_points,
        grid_types=grid_types, percentile_ranges=percentile_ranges, grid_ranges=grid_ranges,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint, agg_dict=agg_dict)

    # prepare summary data frame
    summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_interact_summary(
        data_x=data_x, plot_data=target_plot_data, prepared_results=prepared_results, feature_types=feature_types)
    summary_df = summary_df[info_cols + ['count'] + target]
    ##  添加左上角的标签
    # title = plot_params.get('title', 'Target plot for feature "%s"' % ' & '.join(feature_names))
    # subtitle = plot_params.get('subtitle', 'Average target value through different feature value combinations.')
    title = plot_params.get('title', '')
    subtitle = plot_params.get('subtitle', '')

    # print("feature_names:\n", feature_names)
    # print("display_columns:\n", len(display_columns), "\n", display_columns[0], "\n", display_columns[1])
    # print("percentile_columns:\n", percentile_columns)
    # print("target:\n", target)
    # print("target_plot_data:\n", target_plot_data)
    # print("title:\n", title)
    # print("subtitle:\n", subtitle)
    # print("figsize:\n", figsize)
    # print("ncols:\n", ncols)
    # print("annotate:\n", annotate)
    # print("plot_params:\n", plot_params)
    #
    # print("替换……")
    # print(type(display_columns), len(display_columns))
    if f2v:
	    fr2values = [[] for _ in feature_names]
	    new_display_columns = [[] for _ in display_columns]
	    for i in range(len(feature_names)):
	        fr2values[i] = fr[fr['fact'] == feature_names[i]]
	        fr2values[i] = dict(zip(fr2values[i]["fr"], fr2values[i]["values"]))
	        for j in display_columns[i]:
	            if fr2values[i].get(j) is None:
	                print(j)
	            new_display_columns[i].append(fr2values[i].get(j).replace("^", "\n"))
	    display_columns = new_display_columns
    # print("display_columns:\n", display_columns)
    # print("new_display_columns:\n", new_display_columns)




    # inner call target plot interact
    fig, axes = _info_plot_interact(
        feature_names=feature_names, display_columns=display_columns, percentile_columns=percentile_columns,
        ys=target, plot_data=target_plot_data, title=title, subtitle=subtitle, figsize=figsize,
        ncols=ncols, annotate=annotate, plot_params=plot_params)

    return fig, axes, summary_df


def actual_plot_interact(model, X, features, feature_names, num_grid_points=None, grid_types=None,
                         percentile_ranges=None, grid_ranges=None, cust_grid_points=None, show_percentile=False,
                         show_outliers=False, endpoint=True, which_classes=None, predict_kwds={}, ncols=2,
                         figsize=None, annotate=False, plot_params=None, f2v=False):
    """Plot prediction distribution across different feature value combinations (feature grid combinations)

    Parameters
    ----------

    model: a fitted sklearn model
    X: pandas DataFrame
        data set to investigate on, should contain at least
        the feature to investigate as well as the target
    features: list
        two features to investigate
    feature_names: list
        feature names
    num_grid_points: list, optional, default=None
        number of grid points for each feature
    grid_types: list, optional, default=None
        type of grid points for each feature
    percentile_ranges: list of tuple, optional, default=None
        percentile range to investigate for each feature
    grid_ranges: list of tuple, optional, default=None
        value range to investigate for each feature
    cust_grid_points: list of (Series, 1d-array, list), optional, default=None
        customized list of grid points for each feature
    show_percentile: bool, optional, default=False
        whether to display the percentile buckets for both feature
    show_outliers: bool, optional, default=False
        whether to display the out of range buckets for both features
    endpoint: bool, optional
        If True, stop is the last grid point, default=True
        Otherwise, it is not included
    which_classes: list, optional, default=None
        which classes to plot, only use when it is a multi-class problem
    predict_kwds: dict, default={}
        keywords to be passed to the model's predict function
    figsize: tuple or None, optional, default=None
        size of the figure, (width, height)
    ncols: integer, optional, default=2
        number subplot columns, used when it is multi-class problem
    annotate: bool, default=False
        whether to annotate the points
    plot_params: dict or None, optional, default=None
        parameters for the plot

    Returns
    -------
    fig: matplotlib Figure
    axes: a dictionary of matplotlib Axes
        Returns the Axes objects for further tweaking
    summary_df: pandas DataFrame
        Graph data in data frame format

    Notes
    -----

    - Parameters are consistent with the ones for function actual_plot
    - But for this function, you need to specify parameter value for both features in list format
    - For example:
        - percentile_ranges = [(0, 90), (5, 95)] means
        - percentile_range = (0, 90) for feature 1
        - percentile_range = (5, 95) for feature 2

    Examples
    --------

    Quick start with actual_plot_interact

    .. highlight:: python
    .. code-block:: python

        from pdpbox import info_plots, get_dataset

        test_titanic = get_dataset.titanic()
        titanic_data = test_titanic['data']
        titanic_features = test_titanic['features']
        titanic_target = test_titanic['target']
        titanic_model = test_titanic['xgb_model']

        fig, axes, summary_df = info_plots.actual_plot_interact(
            model=titanic_model, X=titanic_data[titanic_features],
            features=['Fare', ['Embarked_C', 'Embarked_Q', 'Embarked_S']],
            feature_names=['Fare', 'Embarked'])

    """

    # check model
    n_classes, predict = _check_model(model=model)
    check_results = _check_info_plot_interact_params(
        num_grid_points=num_grid_points, grid_types=grid_types, percentile_ranges=percentile_ranges,
        grid_ranges=grid_ranges, cust_grid_points=cust_grid_points, show_outliers=show_outliers,
        plot_params=plot_params, features=features, df=X)

    num_grid_points = check_results['num_grid_points']
    grid_types = check_results['grid_types']
    percentile_ranges = check_results['percentile_ranges']
    grid_ranges = check_results['grid_ranges']
    cust_grid_points = check_results['cust_grid_points']
    show_outliers = check_results['show_outliers']
    plot_params = check_results['plot_params']
    feature_types = check_results['feature_types']

    # prediction
    prediction = predict(X, **predict_kwds)

    info_df = X[_make_list(features[0]) + _make_list(features[1])]
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    agg_dict = {}
    actual_prediction_columns_qs = []
    for idx in range(len(actual_prediction_columns)):
        agg_dict[actual_prediction_columns[idx]] = [q1, q2, q3]
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
    agg_dict['fake_count'] = 'count'

    data_x, actual_plot_data, prepared_results = _prepare_info_plot_interact_data(
        data_input=info_df, features=features, feature_types=feature_types, num_grid_points=num_grid_points,
        grid_types=grid_types, percentile_ranges=percentile_ranges, grid_ranges=grid_ranges,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint, agg_dict=agg_dict)

    actual_plot_data.columns = ['_'.join(col) if col[1] != '' else col[0] for col in actual_plot_data.columns]
    actual_plot_data = actual_plot_data.rename(columns={'fake_count_count': 'fake_count'})

    # prepare summary data frame
    summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_interact_summary(
        data_x=data_x, plot_data=actual_plot_data, prepared_results=prepared_results, feature_types=feature_types)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]

    title = plot_params.get('title', 'Actual predictions plot for %s' % ' & '.join(feature_names))
    subtitle = plot_params.get('subtitle',
                               'Medium value of actual prediction through different feature value combinations.')

    fig, axes = _info_plot_interact(
        feature_names=feature_names, display_columns=display_columns,
        percentile_columns=percentile_columns, ys=[col + '_q2' for col in actual_prediction_columns],
        plot_data=actual_plot_data, title=title, subtitle=subtitle, figsize=figsize,
        ncols=ncols, annotate=annotate, plot_params=plot_params, is_target_plot=False)

    return fig, axes, summary_df

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


def onefact_actual_plot(prediction, X, feature, feature_name, num_grid_points=10, grid_type='percentile', percentile_range=None,
                grid_range=None, cust_grid_points=None, show_percentile=False, show_outliers=False, endpoint=True,
                which_classes=None, predict_kwds={}, ncols=2, figsize=None, plot_params=None, f2v=False):
    # prediction = model.predict_proba(X_test)            percentile: 百分比

    # check inputs
    # n_classes, predict = _check_model(model=model)
    n_classes = 2
    feature_type, show_outliers = _check_info_plot_params(
        df=X, feature=feature, grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_outliers=show_outliers)
    print('feature_type', feature_type)    # 数字 numeric
    #print('show_outliers', show_outliers)   False

    # make predictions
    # info_df only contains feature value and actual predictions
    # print('X.shape', X.shape)   (num_data,42)
    # print('feature', feature)
    # print('_make_list(feature)', _make_list(feature))   feature
    info_df = X[_make_list(feature)]
    # print(type(info_df))  # <class 'pandas.core.frame.DataFrame'>
    # print('info_df',info_df.shape)    (num_data,1)
    actual_prediction_columns = ['actual_prediction']
    if n_classes == 0:
        info_df['actual_prediction'] = prediction
    elif n_classes == 2:   # 默认执行这个
        info_df['actual_prediction'] = prediction[:, 1]
    else:
        plot_classes = range(n_classes)
        if which_classes is not None:
            _check_classes(classes_list=which_classes, n_classes=n_classes)
            plot_classes = sorted(which_classes)

        actual_prediction_columns = []
        for class_idx in plot_classes:
            info_df['actual_prediction_%d' % class_idx] = prediction[:, class_idx]
            actual_prediction_columns.append('actual_prediction_%d' % class_idx)

    info_df_x, bar_data, summary_df, info_cols, display_columns, percentile_columns = _prepare_info_plot_data(
        feature=feature, feature_type=feature_type, data=info_df, num_grid_points=num_grid_points,
        grid_type=grid_type, percentile_range=percentile_range, grid_range=grid_range,
        cust_grid_points=cust_grid_points, show_percentile=show_percentile,
        show_outliers=show_outliers, endpoint=endpoint)

    print("info_df_x:\n", info_df_x)
    print("bar_data:\n", bar_data)
    print("summary_df:\n", summary_df)
    print("info_cols:\n", info_cols)
    print("info_df_x:\n", info_df_x)
    print("display_columns:\n", display_columns)

    print("替换……")
    print(type(display_columns), len(display_columns))

    # 以下代码是FR转真实值的：
    # if f2v:   # 默认不执行
    #     pass
    # fr2values = fr[fr['fact'] == feature_name]
    # print(fr2values)
    # fr2values = dict(zip(fr2values["fr"], fr2values["values"]))
    # new_display_columns = []
    # for i in display_columns:
    #     if fr2values.get(i) is None:
    #         print(i)
    #     new_display_columns.append(fr2values.get(i).replace("^", "\n"))
    # display_columns = new_display_columns
    # print("display_columns:\n", display_columns)
    # print("new_display_columns:\n", new_display_columns)

    # exit(0)
    print("percentile_columns:\n", percentile_columns)

    # prepare data for box lines
    # each box line contains 'x' and actual prediction q1, q2, q3
    box_lines = []
    actual_prediction_columns_qs = []
    # print('len(actual_prediction_columns)',len(actual_prediction_columns))
    for idx in range(len(actual_prediction_columns)):
        # print('actual_prediction_columns[idx]',actual_prediction_columns[idx])
        box_line = info_df_x.groupby('x', as_index=False).agg(
            {actual_prediction_columns[idx]: [q1, q2, q3]}).sort_values('x', ascending=True)
        # q1, q2, q3分别指的是当前区间所有预测结果的0.25,0.5， 0.75 分位数
        print('box_line\n', box_line)
        print('box_line.columns', box_line.columns)
        box_line.columns = ['_'.join(col) if col[1] != '' else col[0] for col in box_line.columns]
        print('box_line\n', box_line)
        box_lines.append(box_line)
        actual_prediction_columns_qs += [actual_prediction_columns[idx] + '_%s' % q for q in ['q1', 'q2', 'q3']]
        print('actual_prediction_columns_qs',actual_prediction_columns_qs)

        summary_df = summary_df.merge(box_line, on='x', how='outer').fillna(0)
        print("summary_df:\n", summary_df)
        print(summary_df.columns)
    summary_df = summary_df[info_cols + ['count'] + actual_prediction_columns_qs]  # 调换位置
    print("summary_df:\n", summary_df)
    print('summary_df.columns',summary_df.columns)   # Index(['x', 'display_column', 'value_lower', 'value_upper',
                                               # 'percentile_column', 'percentile_lower', 'percentile_upper', 'count',
                                               # 'actual_prediction_q1', 'actual_prediction_q2', 'actual_prediction_q3']
    # figsize = (16, 9)
    figsize = None
    # dpi = 600
    print('info_df_x\n', info_df_x)
    print('bar_data\n', bar_data)
    print('box_lines\n', box_lines)
    print('actual_prediction_columns\n', actual_prediction_columns)
    print('feature\n', feature)
    print('display_columns\n', display_columns)
    print('percentile_columns\n', percentile_columns)
    print('figsize\n', figsize)
    print('ncols\n', ncols)
    print('plot_params\n', plot_params)
    fig, axes = _actual_plot(plot_data=info_df_x, bar_data=bar_data, box_lines=box_lines,
                             actual_prediction_columns=actual_prediction_columns, feature_name=feature_name,
                             display_columns=display_columns, percentile_columns=percentile_columns, figsize=figsize,
                             ncols=ncols, plot_params=plot_params)
    print("figsize:\n", figsize)
    print("ncols:\n", ncols)
    print("plot_params:\n", plot_params)
    print("actual_prediction_columns:\n", actual_prediction_columns)
    print("box_lines:\n", box_lines)
    print("bar_data:\n", bar_data)
    return fig, axes, summary_df

