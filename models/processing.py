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


class ProcessEDA:
    """
    Class for Exploratory Data Analysis.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame for analysis. Any format, any columns.

    table_description : pd.DataFrame, default=None
        DataFrame with explanation of each column in main data.
        columns = {"Table", "Row", "Description", "Special"}

    target_column : str, default=None
        Name of target column in data.

    id_column : str, default=None
        Name of column with row id.

    date_column : str, default=None
        Name of column with dates.

    Attributes:
    ----------
    data : pd.DataFrame, from constructor.

    table_description :  pd.DataFrame, from constructor.

    target_column : str, from constructor.

    id_column : str, from constructor.

    date_column : str, from constructor.

    column_types : dict,
        column_types.keys() = dict_keys(['Continuous', 'Binary', 'Nominal', 'Unknown']).
    """

    def __init__(self,
                 data: pd.DataFrame,
                 table_description=None,
                 target_column=None,
                 id_column=None,
                 date_column=None):

        self.data = data
        self.table_description = table_description
        self.target_column = target_column
        self.id_column = id_column
        self.date_column = date_column

        self.column_types = self._get_column_types(exclude_col=[target_column, id_column])

    def _get_column_types(self,
                          exclude_col=None,
                          include_col=None):
        """
        Finds type of data for each column.

        Parameters:
        ----------
        exclude_col : list of str, default=None.
            Which columns excluded. If None -> exclude 0 columns.

        include_col : list of str, default=None.
            Which columns included. If None -> include all columns.

        Returns:
        --------
        column_types: dict,
            Columns associated with dtypes in the form "{dtype: [columns]}"
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
            col_dtype = data.dtypes[column]
            n_unique = len(data[column].unique())

            if col_dtype == 'datetime64':
                continue
            elif col_dtype == 'bool':
                binary_columns.append(column)
            elif col_dtype == 'object':
                if n_unique == 2:
                    binary_columns.append(column)
                else:
                    nominal_columns.append(column)
            elif col_dtype in ['float64', 'int64']:
                if n_unique == 2:
                    binary_columns.append(column)
                elif 3 <= n_unique <= 15:
                    unknown_columns.append(column)
                else:
                    continuous_columns.append(column)
            else:
                unknown_columns.append(column)

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

        Parameters:
        ----------
        display_type: str,
            What type of data to display. One of {'Continuous', 'Binary', 'Nominal', 'Unknown'}.
        """
        data = self.data
        table_description = self.table_description

        if self.column_types is None:
            self._get_column_types()
        column_types = self.column_types

        display_columns = column_types[display_type]
        print(display_type, ' Data')
        for column in display_columns:
            col_distribution = data[column].value_counts(normalize=True)

            if table_description is not None:
                col_description = table_description[column].values[0]
            else:
                col_description = ''

            print("\n==========\n",
                  column, ' - ', col_description,
                  "\n==========",
                  col_distribution)


application_train = pd.read_csv('../data/application_train.csv')
application_train_description = get_table_description(filename='../data/HomeCredit_columns_description.csv',
                                                      table_name='application_{train|test}.csv')


process_data = ProcessEDA(data=application_train,
                          table_description=application_train_description,
                          target_column='TARGET',
                          id_column='SK_ID_CURR')

print(process_data.column_types)