import pandas as pd


def get_table_description(filename: str,
                          table_name: str):
    """
    Get description of each column in particular table.

    Arguments:
        filename: (str) csv file with columns:
                         'Table' - (str) filename of table;
                         'Row' - (str) column name;
                         'Description' - (str) description of column;
                         'Special' - (str) comments.

        table_name: one of string in 'Table' column, filename of table for description.
    Returns:
        table_description: pd.DataFrame
    """
    # Read file with descriptions
    description_df = pd.read_csv(filename, encoding="ISO-8859-1", index_col=None)
    # Extraction info
    mask_table = description_df['Table'] == table_name
    table_description_raw = description_df[mask_table][['Row', 'Description']]
    dict_description = {k: v for k, v in table_description_raw.values}

    table_description = pd.DataFrame(dict_description, index=['Description'])
    return table_description


class DataProcess:

    def __init__(self,
                 data: pd.DataFrame,
                 table_description=None):
        self.data = data
        self.table_description = table_description
        self.column_types = None

    def get_column_types(self,
                         exclude_col=None,
                         include_col=None):
        """
        Finds type of data for each column.
            exclude_col: list of str, which columns exclude. Default None, exclude 0 columns.
            include_col: list of str, which columns include. Default None, include all columns.
        Returns:
            column_types: dict with columns of each data type:
                             'Continuous' -- key of decimal columns (Numerical);
                             'Binary' -- key of binary columns, Yes/No or 1/0 (Categorical);
                             'Nominal' -- key of nominal columns, string (Categorical);
                             'Unknown' -- key of columns that could be both (Numerical,Categorical).
        """
        data = self.data
        column_types = dict()
        assert (exclude_col is None) or (include_col is None), 'Could set only one param'

        if include_col is not None:
            all_columns = include_col
        elif exclude_col is not None:
            all_columns = list(data.columns)
            for col in exclude_col:
                all_columns.remove(col)
        else:
            all_columns = list(data.columns)

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

        self.column_types = column_types
        return column_types

    def get_dtype_info(self,
                       display_type: str):
        """
        Show information of particular data type.
        Format of display - Data Type, Column Name, Description(Optional), Distribution.

        Arguments:
            display_type: str, one of ('Continuous', 'Binary', 'Nominal', 'Unknown').

        """
        data = self.data
        table_description = self.table_description

        if self.column_types is None:
            self.get_column_types()
        column_types = self.column_types

        display_columns = column_types[display_type]
        print(display_type, ' Data')
        for column in display_columns:
            col_description = table_description[column].values[0]
            col_distribution = data[column].value_counts(normalize=True)

            print("\n==========\n",
                  column, ' - ', col_description,
                  "\n==========",
                  col_distribution)


application_train = pd.read_csv('../data/application_train.csv')
application_train_description = get_table_description(filename='../data/HomeCredit_columns_description.csv',
                                                      table_name='application_{train|test}.csv')


process_data = DataProcess(data=application_train,
                           table_description=application_train_description)

print(process_data.get_column_types())