import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfTransformer


# function to transform event-log to 2-size-sequences log
def find_transitions(data, groupby_col, sortby_col, replaced_col, insert_artif_beginend=True):
    # function to find 2-size-sequences of activities
    def add_sequence(x):
        x['pseudotransition'] = x[replaced_col].shift(1) + '-' + x[replaced_col]
        return x

    aux_data = data.copy()
    if insert_artif_beginend:
        aux_data = insert_artif_acts(aux_data, groupby_col, sortby_col, replaced_col)
    data_sequences = aux_data.groupby(by=groupby_col, sort=sortby_col).apply(add_sequence)
    # data_sequences.sequence.fillna('no_transition', inplace = True)   # if you want to use the NaN as no_transition
    data_sequences.drop(columns=[replaced_col, sortby_col], inplace=True)
    data_sequences.dropna(subset=['pseudotransition'], inplace=True)
    return data_sequences


def insert_artif_acts(data, index_col, ts_col, act_col):
    one_second = timedelta(seconds=1)

    # to each trace, a pre and a pos activity will be added
    def create_pre_pos_act(x):
        index_val = x[index_col].unique()[0]  # get the value of this group index
        pre_ts = x[ts_col].min() - one_second
        pos_ts = x[ts_col].max() + one_second
        new = pd.DataFrame({ts_col: [pre_ts, pos_ts],
                            index_col: [index_val, index_val],
                            act_col: ['Artificial_begin', 'Artificial_end']})
        return x.append(new, ignore_index=True)

    return data.groupby(index_col, sort=[ts_col]).apply(create_pre_pos_act) \
        .sort_values([index_col, ts_col]) \
        .reset_index(drop=True)


def create_binary_matrix(data, index_column, pivoted_column):
    # create binary representation
    drop_cols = [col for col in data.columns if col != index_column and col != pivoted_column]
    data = data.drop(columns=drop_cols)
    binary = pd.pivot_table(data,
                            columns=pivoted_column,
                            index=index_column,
                            aggfunc=lambda x: 1,
                            fill_value=0)
    return binary


def create_tf_matrix(data, index_column, pivoted_column):
    # create tf representation
    drop_cols = [col for col in data.columns if col != index_column and col != pivoted_column]
    data = data.drop(columns=drop_cols)
    tf = pd.pivot_table(data,
                        columns=pivoted_column,
                        index=index_column,
                        aggfunc=len,
                        fill_value=0)
    return tf


def create_tfidf_matrix(tf_repres):
    # create tfidf representation
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_res = transformer.fit_transform(tf_repres.values.tolist())
    tfidf = pd.DataFrame(tfidf_res.toarray())
    tfidf.columns = tf_repres.columns
    tfidf.set_index(tf_repres.index.copy(), inplace=True)

    return tfidf


def create_performance_matrix(data, groupby_col, ts_col, drop_cols, rename_cols):
    def calc_duration(x):
        x['duration'] = x[ts_col].max() - x[ts_col].min()
        x['duration'] = x['duration'].astype('timedelta64[m]')
        return x

    def calc_time_diff(x):
        x['diff'] = x[ts_col] - x[ts_col].shift(1)
        x['diff'] = x['diff'].astype('timedelta64[m]')
        return x

    res = data.groupby(by=groupby_col).count().drop(columns=drop_cols).rename(columns=rename_cols)

    tduration = data.sort_values(ts_col).groupby(groupby_col)[ts_col].agg(['first', 'last'])
    res['duration'] = (tduration['last'] - tduration['first']).astype('timedelta64[m]')

    events_durations = data.groupby(by=groupby_col, sort=[groupby_col, ts_col]).apply(calc_time_diff)
    res['min_diff'] = events_durations.groupby(by=groupby_col).min()['diff']
    res['max_diff'] = events_durations.groupby(by=groupby_col).max()['diff']
    res['mean_diff'] = events_durations.groupby(by=groupby_col).mean()['diff']
    res['median_diff'] = events_durations.groupby(by=groupby_col).median()['diff']

    return res