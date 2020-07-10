import csv
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly as py
import squarify
from random import randint
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.offline as py


# py.init_notebook_mode(connected=False)


def create_metadata_file(filename, header):
    filename = '%s_%s.csv' % (filename, time.strftime("%Y%m%d%H%M%S"))
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(header.split(','))
    return filename


def save_metadata(filename, fixed_part, silhouette, ari, elems_qtt, time_clust, time_total):
    fixed_part = '%s,%s,%s,%s,%s' % (fixed_part, str(silhouette), str(ari), time_clust, time_total)
    with open(filename, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        for i in range(len(elems_qtt)):
            new = fixed_part.split(',') + [i, elems_qtt[str(i)]]
            spamwriter.writerow(new)


def save_metadata_incidents(filename, fixed_part, time_clust, time_total, elems_qtt,
                            silhouette_means, silhouette_stds, durations_means, durations_stds):
    fixed_part = '%s,%s,%s' % (fixed_part, time_clust, time_total)
    with open(filename, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        for i in range(len(elems_qtt)):
            new = fixed_part.split(',') + ([i, elems_qtt[i], silhouette_means[i], silhouette_stds[i],
                                            durations_means[i], durations_stds[i]])
            spamwriter.writerow(new)


def plot_incidents_tests_statistics(filename, path):
    def get_combination(df):
        return '-'.join([df[0], df[1], df[2], str(df[4])])

    def get_dfs(df):
        df['test combination'] = df.apply(lambda x: get_combination(x), axis=1)
        df['k'] = df['k'].astype(int)
        df = df.loc[df['k'] != 0]
        df['duration_mean'] = df['duration_mean'].round(2)
        df['duration_std'] = df['duration_std'].round(2)
        df = df.sort_values(by=['feature_selection', 'counting'])

        df_grouped = (df[['test combination', 'k', 'duration_std', 'elem_qtt', 'silhouette_mean', 'silhouette_std']]
                      .groupby(['test combination', 'k'])
                      .agg([np.mean, np.std, np.sum]))
        df_grouped['stds_std'] = df_grouped['duration_std']['std'].round(2)
        df_grouped['stds_mean'] = df_grouped['duration_std']['mean'].round(2)
        df_grouped['silhouette_group_mean'] = df_grouped['silhouette_mean']['mean'].round(2)
        df_grouped['silhouette_group_std'] = df_grouped['silhouette_std']['std'].round(2)
        df_grouped['elem_qtt_mean'] = (df_grouped['elem_qtt']['sum'])
        df_grouped['elem_qtt_std'] = (df_grouped['elem_qtt']['std'])  # /df_grouped['elem_qtt']['sum']).round(2)
        df_grouped = df_grouped.reset_index().sort_values(['test combination', 'k'])
        df_grouped['test combination'] = df_grouped['test combination'].apply(lambda x: '-'.join(x.split('-')[:-1]))

        return df, df_grouped

    def plot_durations_means(df):
        ''' Plots incidentes' resolution times in each resulting group,
            generating one image for each (feature construction mode (single or combined), k) pair,
            with the counting_strategy in the rows and attributes selection in the columns'''
        for act_repres in list(df['activity_represent'].unique()):
            for k in [3, 5, 10]:
                select = df.loc[df['activity_represent'] == act_repres].loc[df['k'] == k]
                fig = px.bar(select, x='group', y='duration_mean', error_y='duration_std',  # text = 'duration_mean',
                             facet_col='feature_selection', facet_row='counting', template="plotly_white",
                             title='Média do tempo de resolução dos incidentes em cada grupo - representação = %s, k = %s' % (
                             act_repres, str(k)))
                fig.write_image('%sgroups_means/%s-k%s.png' % (path, act_repres, str(k)))
                fig.write_html('%sgroups_means/%s-k%s.html' % (path, act_repres, str(k)))

    def plot_durations_means_stds(df_grouped):
        fig = (px.bar(df_grouped, x='test combination', y="stds_mean",  # text = 'stds_mean',
                      facet_row='k', barmode='group',
                      error_y='stds_std', template="plotly_white",
                      title="Média do desvio padrão nos grupos",
                      width=700, height=800))
        fig.update_xaxes(tickangle=90)
        fig.write_image('%sstds.png' % path)
        fig.write_html('%sstds.html' % path)

    def plot_silhouettes(df_grouped):
        fig = (px.bar(df_grouped, x='test combination', y="silhouette_group_mean", text='silhouette_group_mean',
                      error_y='silhouette_group_std', facet_row='k', template="plotly_white",
                      width=1000, height=1200, title='Silhouette médio dos grupos'))
        fig.update_xaxes(tickangle=90)
        fig.write_image('%ssilhouettes.png' % path)
        fig.write_html('%ssilhouettes.html' % path)

    def plot_elems(df_grouped):
        fig = (px.bar(df_grouped, x='test combination', y="elem_qtt_mean", text='elem_qtt_mean',
                      error_y='elem_qtt_std', facet_row='k', template="plotly_white",
                      width=1000, height=1200,
                      title='Desvio padrão da quantidade de elementos por grupo (relativo à quantidade total de elementos)'))
        fig.update_xaxes(tickangle=90)
        fig.write_image('%selems.png' % path)
        fig.write_html('%selems.html' % path)

    df = pd.read_csv(filename)
    df, df_grouped = get_dfs(df)

    plot_durations_means(df)
    plot_durations_means_stds(df_grouped)
    plot_silhouettes(df_grouped)
    plot_elems(df_grouped)


def get_groups_count(labels):
    groups = labels.tolist()
    groups_count = {}
    for group in groups:
        items_count = groups.count(group)
        groups_count[str(group)] = items_count
    return groups_count


def plot_elems_per_group(groups_count, filename, plot_mean=False):
    fig, ax = plt.subplots()
    rects1 = ax.bar(np.arange(len(groups_count)), groups_count, 0.5, color=cm.Set2(.3))
    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False);
    ax.tick_params(color='gray', bottom=False, labelbottom=False, grid_alpha=0.5)
    if plot_mean:
        mean = np.mean(groups_count)
        plt.axhline(y=np.mean(groups_count), color='red', linestyle='--', linewidth=2)
        ax.text(len(groups_count) - 1.5, mean + 10, 'Mean: %s' % (mean.round(2)), size=15, color='red')
    plt.ylabel('Items in the group')
    plt.xlabel('Groups')
    fig.savefig(filename)
    plt.close(fig)
    return plt


def create_groups_wordcloud(kmeans_res, filename=None):
    wc_groups = wordcloud.WordCloud()
    wc_groups.background_color = 'white'
    wc_groups.fit_words(kmeans_res)
    if filename:
        wc_groups.to_file(filename)
    return wc_groups


def define_filename(path, n, ext='.csv'):
    filename = None
    if path:
        filename = path + str(n) + ext
    return filename


def get_groups_data(original_data, df, K, group_col, data_id_name='case',
                    df_id_name=None, path_groups_data=None, path_sublogs=None):
    def get_group_data(n):  # matrix format ()
        def filter_group(group_number):
            group = df.loc[df[group_col] == group_number]
            return group.drop(columns=group_col)

        df_group = filter_group(n)
        if path_groups_data:
            filename_group = define_filename(path_groups_data, n)
            df_group.to_csv(filename_group)
        return df_group

    def get_group_original_data(df_group, n):  # log format

        def get_traces_ids():
            traces_ids = df_group.index.unique()
            if df_id_name:
                traces_ids = df_group[df_id_name].unique()
            traces_ids = list(np.array(traces_ids) + 1)
            return traces_ids

        if path_sublogs:
            traces_ids = get_traces_ids()
            original_data_group = original_data.loc[original_data[data_id_name].isin(traces_ids)]
            filename_group = define_filename(path_sublogs, n)
            original_data_group.to_csv(filename_group)
            return original_data_group

    groups_data = {}
    for group_id in range(0, K):
        df_group = get_group_data(group_id)
        groups_data[group_id] = df_group
        if df_group.empty == True:
            continue
        get_group_original_data(df_group, group_id)
    return groups_data


def create_groups_visualizations(groups_data, heur_font_size=500,
                                 path_wc=None, path_barh=None, path_treemap=None, path_feat_list=None):
    def columns_frequencies(df_group):
        freqs = pd.to_numeric(df_group.sum(), errors='coerce')
        freqs = freqs[freqs > 0].sort_values(ascending=False)
        return freqs

    def create_and_save_group_wc(freqs, n):
        if path_wc:
            filename_wc = define_filename(path_wc, n, ext='.png')
            create_wordcloud(freqs.to_dict(), filename=filename_wc, heur_font_size=heur_font_size)

    def create_and_save_bars(freqs, n):
        if path_barh:
            filename_barh = define_filename(path_barh, n, ext='.png')
            freq_df = pd.DataFrame(freqs).reset_index()
            freq_df.columns = ['feature', 'frequency']
            fig = px.bar(freq_df[-100:], y='feature', x='frequency', orientation='h', template="plotly_white")
            fig.write_image(filename_barh)

    def create_and_save_treemap(freqs, n):
        if path_treemap:
            filename_treemap = define_filename(path_treemap, n, ext='.png')
            freq_df = pd.DataFrame(freqs).reset_index()
            plt.figure(figsize=(40, 20), dpi=80)
            colors = [plt.cm.Spectral(i / float(len(freq_df['index'].unique()))) for i in
                      range(len(freq_df['index'].unique()))]
            squarify.plot(sizes=freq_df[0], label=freq_df['index'], alpha=.8, color=colors)
            plt.savefig(filename_treemap)

    def create_and_save_features_list(groups_feat_list, n):
        if path_feat_list:
            pd.DataFrame(groups_feat_list).to_csv(define_filename(path_feat_list, n, ext='.csv'))

    groups_feat_list = {}
    for group_id in groups_data:
        group_freqs = columns_frequencies(groups_data[group_id])
        groups_feat_list[group_id] = pd.Series(group_freqs.index)
        create_and_save_group_wc(group_freqs, group_id)
        create_and_save_bars(group_freqs, group_id)
        create_and_save_treemap(group_freqs, group_id)
        create_and_save_features_list(groups_feat_list, group_id)
    return None


def create_labels_wordclouds(df, labeled_data, K, group_col, path_wc=None, colors={}):
    def group_wc(df_group, n):

        def get_filename():
            filename = None
            if path_wc:
                filename = path_wc + str(n)
            return filename

        def insert_color(word):
            if word not in colors:
                colors[word] = 'hsl(%s,60%%,60%%)' % (randint(0, 360))

        def columns_frequencies():
            bpc_freq = {};
            loop_freq = {};
            act_freq = {}
            for column in df_group.columns:
                col_sum = df_group[column].sum()
                if col_sum > 0:
                    col_str = column.replace('label_', '')
                    insert_color(col_str)
                    if 'BPC' in column:
                        col_str = col_str.replace('BPC-', '')
                        insert_color(col_str)
                        bpc_freq[col_str] = col_sum
                    elif 'ACT' in column:
                        col_str = col_str.replace('ACT-', '')
                        insert_color(col_str)
                        act_freq[col_str] = col_sum
                    else:
                        loop_freq[col_str] = col_sum

            return bpc_freq, loop_freq, act_freq

        filename_wc = get_filename()
        bpc_freq, loop_freq, act_freq = columns_frequencies()
        bpc_wc = create_wordcloud(bpc_freq, filename=filename_wc + '_bpc.png', my_colors=colors)
        loop_wc = None
        if loop_freq:
            loop_wc = create_wordcloud(loop_freq, filename=filename_wc + '_loop.png', my_colors=colors)
        act_wc = None
        if act_freq:
            act_wc = create_wordcloud(act_freq, filename=filename_wc + '_act.png', my_colors=colors)
        return bpc_wc, loop_wc, act_wc

    wordclouds = []
    labels = [col for col in labeled_data.columns if 'label' in col]
    df = df.join(labeled_data.loc[:, labels])
    for n in range(0, K):
        df_group = df.loc[df[group_col] == n].loc[:, labels]
        wcs = group_wc(df_group, n)
        wordclouds.append(wcs)
    return wordclouds


def calc_silhouette(df, labels, k):
    silhouette_avg = silhouette_score(df, labels)
    sample_silhouette_values = silhouette_samples(df, labels)
    return silhouette_avg, sample_silhouette_values


def plot_silhouette(labels, k, silhouette_avg, sample_silhouette_values, filename, plot_mean=True):
    groups_avg = []
    groups_std = []

    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)

    ax1.set_xlim([-1, 1])
    ax1.xaxis.label.set_size(40)

    y_upper = 0
    y_lower = 0
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        groups_avg.append(ith_cluster_silhouette_values.mean())
        groups_std.append(ith_cluster_silhouette_values.std())

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Set2(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.8)

        y_lower = y_upper + 10

    if plot_mean:
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.text(silhouette_avg + 0.03, y_upper / 2, silhouette_avg.round(2), size=25, color='red')

    ax1.set_ylim(bottom=0, top=y_upper + 20)
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.set_xticks(np.arange(-1, 2, 0.5))
    ax1.set_yticks(np.arange(0, y_upper + 1, y_upper / 5))
    ax1.tick_params(color='gray', labelsize=30, labelcolor='gray', labelright=False, labelleft=False, grid_alpha=0.5)
    ax1.spines['top'].set_visible(False);
    ax1.spines['right'].set_visible(False);
    ax1.spines['left'].set_visible(False)

    if filename:
        fig.savefig(filename)

    plt.close(fig)
    return plt, groups_avg, groups_std


def create_wordcloud(frequencies_dict, filename=None, my_colors=None, heur_font_size=500):
    def my_color_func(word, **kwargs):
        return my_colors[word]

    if (frequencies_dict):
        wc = wordcloud.WordCloud(background_color='white', max_font_size=heur_font_size / len(frequencies_dict))
        if my_colors:
            wc = wordcloud.WordCloud(background_color='white', max_font_size=heur_font_size / len(frequencies_dict),
                                     color_func=my_color_func)
        wc.fit_words(frequencies_dict)
        if filename:
            wc.to_file(filename)
        wc.to_image()
        return wc
    return None


def save_clustered_df(df, filename):
    df.to_csv(filename + '.csv')


def save_centers(centers, filename):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(centers, handle)


def plot_and_save_shifts(shifts, filename, graph_filename):
    shifts = pd.DataFrame({'shifts': shifts})

    fig = px.line(shifts, y='shifts', template='plotly_white')
    fig.write_image(graph_filename + '.png')
    fig.write_html(graph_filename + '.html')

    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(shifts, handle)