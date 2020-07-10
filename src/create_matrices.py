import json
import pandas as pd
import sys

from scipy import sparse
from src.utils.log2matrix import create_binary_matrix, create_tf_matrix, create_tfidf_matrix

sys.path.append('../')

def get_and_save_matrices(representations, selections, log):
    for representation in representations:
        for selection in selections:
            print(
                '\n------\nCriando matrizes para composicao de feature por atributo "%s", usando o conjunto de atributos "/%s":' % (
                representation, selection))
            # Às vezes, o arquivo csv possui valores nulos, que são posteriormente exibidos como NaN no Data Frame. O método                 # dropna () do Pandas permite ao usuário analisar e eliminar linhas / colunas com valores nulos de diferentes maneiras.
            current = (log[['number'] + selections[selection]]).dropna()
            representations_matrices = create_and_save_matrices(current, representation, selections[selection],
                                                                'number', 'feature',
                                                                'matrices/%s_%s_' % (representation, selection))


def create_and_save_matrices(df, representation, cols, index_col, feature_col, filename):
    def get_filename(counting):
        return '%s%s.csv' % (filename, counting)

    if representation == 'individual':
        df = get_individual_val_repres(df, cols, index_col)
    elif representation == 'combined':
        df = get_combined_val_repres(df, cols, index_col)

    matrix = create_binary_matrix(df, index_col, feature_col)
    print('Matriz com contagem binaria criada (shape %s)!' % str(matrix.shape))
    matrix.to_csv(get_filename('binary'))
    matrix = create_tf_matrix(df, index_col, feature_col)
    print('Matriz com contagem tf criada (shape %s)!' % str(matrix.shape))
    matrix.to_csv(get_filename('tf'))
    matrix = create_tfidf_matrix(matrix)
    print('Matriz com contagem  tfidf criada (shape %s)!' % str(matrix.shape))
    matrix.to_csv(get_filename('tfidf'))
    print('Done!\n-------\n')


def get_combined_val_repres(df, cols, index_col):
    def add_col_name(x):
        aux = x.index + '-' + x.astype(str)
        x['feature'] = '--'.join(aux)
        return x

    df = (df.set_index(index_col)
          .apply(lambda x: add_col_name(x), axis=1)
          .reset_index())
    return df[[index_col, 'feature']]


def get_individual_val_repres(df, cols, index_col):
    df_melt = pd.melt(df, id_vars=index_col, value_vars=cols)  # .dropna()
    df_melt['feature'] = df_melt[['variable', 'value']].astype(str).apply(lambda x: '-'.join(x), axis=1)
    return df_melt.drop(columns=['variable', 'value'])


# Get matrices to all combination
def create_matrices(log):
    log.head()
    representations = ['individual'] #['individual', 'combined']
    selections={'specialist': ['incident_state', 'category', 'priority']}
    get_and_save_matrices(representations, selections, log)