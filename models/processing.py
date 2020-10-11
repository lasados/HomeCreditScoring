import pandas as pd
import numpy as np


def get_table_description(filename: str,
                          table_name: str):
    """
    Get description of each column in particular table.

    Arguments:
        filename - (str) csv file with columns:
                        ['Table' - (str) filename of table,
                         'Row' - (str) column name,
                         'Description' - (str) description of column,
                         'Special' - (str) comments]

        table_name - string in 'Table', filename of table
    Returns:
        table_description - pd.DataFrame
    """
    # Read file with descriptions
    description_df = pd.read_csv(filename, encoding="ISO-8859-1")
    assert description_df.columns == ['Table', 'Row', 'Description', 'Special'], 'Wrong file'
    # Extraction info
    mask_table = description_df['Table'] == table_name
    table_description_raw = description_df[mask_table][['Row', 'Description']]
    dict_description = {k: v for k, v in table_description_raw.values}

    table_description = pd.DataFrame(dict_description, index=['Description'])
    return table_description


def get_column_types(data: pd.DataFrame,
                     exclude_col=None,
                     include_col=None):
    """
    Finds type of data for each column.

    Arguments:
        data - pd.DataFrame
        exclude_col - list of str, which columns exclude. Default - None, exclude 0 columns.
        include_col - list of str, which columns include. Default - None, include all columns.
    Returns:
        column_types - dict with keys:
                        ['Continuous' - decimal (Numerical),
                         'Binary' - Yes/No or 1/0 (Categorical),
                         'Nominal' - string (Categorical),
                         'Unknown' - columns that could be both]
    """
    column_types = dict()
    assert (exclude_col is not None) or (include_col is not None), 'Could set only one param'

    if include_col is not None:
        all_columns = include_col
    elif exclude_col is not None:
        all_columns = data.columns
        for col in exclude_col:
            all_columns.pop(col)
    else:
        all_columns = data.columns

    continuous_columns = []
    binary_columns = []
    nominal_columns = []
    unknown_columns = []

    for column in all_columns:
        dtype = data.dtypes[column]
        n_unique = len(data[column].unique())

        if dtype == 'object':
            if n_unique == 2:
                binary_columns.append(column)
            else:
                nominal_columns.append(column)
        else:
            if n_unique == 2:
                binary_columns.append(column)
            elif 3 <= n_unique <= 15:
                unknown_columns.append(column)
            else:
                continuous_columns.append(column)

    column_types['Continuous'] = continuous_columns
    column_types['Binary'] = binary_columns
    column_types['Nominal'] = nominal_columns
    column_types['Unknown'] = unknown_columns

    return column_types

