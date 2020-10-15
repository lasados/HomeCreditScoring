import pandas as pd
import matplotlib.pyplot as plt


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


class BinaryClassEDA:
    """
    Class for Exploratory Data Analysis in Binary Classification task.

    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame for analysis. Any format, any columns.

    target_column : str
        Name of target column in data. Contain 1 and 0.

    table_description : pd.DataFrame, default=None
        DataFrame with explanation of each column in main data.
        columns = {"Table", "Row", "Description", "Special"}

    id_column : str, default=None
        Name of column with row id.

    date_column : str, default=None
        Name of column with dates.

    Attributes:
    ----------
    data : pd.DataFrame, from constructor.

    target_column : str, from constructor.

    table_description :  pd.DataFrame, from constructor.

    id_column : str, from constructor.

    date_column : str, from constructor.

    column_types : dict,
        column_types.keys() = dict_keys(['Continuous', 'Binary', 'Nominal', 'Unknown']).
    """

    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str,
                 table_description=None,
                 id_column=None,
                 date_column=None):

        self.data = data
        self.target_column = target_column
        assert self._is_valid_target(), 'Target column must contain 1 and 0'

        self.table_description = table_description
        self.id_column = id_column
        self.date_column = date_column

        self.column_types = self._get_column_types(exclude_col=[target_column, id_column])

    def _is_valid_target(self):
        data = self.data
        target_column = self.target_column
        return set(data[target_column].unique()) == {1, 0}

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
            n_unique = pd.notna(data[column].unique()).sum()

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

    def count_column_dtypes(self):
        """
        Returns number of columns for each dtype in attribute column_types.
        """
        count_dict = {key: len(value) for key, value in self.column_types.items()}
        count_df = pd.DataFrame(count_dict, index=['N columns'])
        return count_df

    def change_dtype_group(self,
                           new_dtype: str,
                           old_dtype=None,
                           change_columns=None,
                           on_delete=True):
        """
        Changes dtype of columns.

        Parameters:
        ----------
        new_dtype : str, new type of columns.
        One of 'Continuous', 'Binary', 'Nominal', 'Unknown'.

        old_dtype : str, type of data to change.

        columns : list of strings. columns to change.

        Note:
        ----
        Only one of params (old_dtype, columns) should be set.
        """
        column_types = self.column_types
        if change_columns is None:
            assert old_dtype is not None
            column_types[new_dtype].extend(column_types[old_dtype])
            if on_delete:
                column_types[old_dtype] = []

        elif old_dtype is None:
            assert change_columns is not None
            assert type(change_columns) == list

            if on_delete:
                for key in column_types:
                    column_types[key] = [c for c in column_types[key] if c not in change_columns]

            # Add columns after deletion
            column_types[new_dtype].extend(change_columns)
        else:
            raise AttributeError('Set old_dtype or columns')

        self.column_types = column_types

    def create_dtype_group(self, group_name: str):
        """
        Creates new group in columns_type.

        Parameters:
        -----------
        group_name : str, name of new group.
        """
        self.column_types[group_name] = []

    def get_dtype_info(self, display_type: str):
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
            col_distribution = data[column].value_counts(normalize=True, dropna=False)

            if table_description is not None:
                col_description = table_description[column].values[0]
            else:
                col_description = ''

            print("\n==========\n",
                  column, ' - ', col_description,
                  "\n==========\n", col_distribution)

    def plot_categorical_dist_by_target(self,
                                        plot_dtype=None,
                                        plot_columns=None,
                                        positive_name='default',
                                        negative_name='good'):
        """
        Plots distribution bars of selected categorical columns in different target groups.
        Columns must contain categorical features - ordinal, nominal, binary.

        Parameters:
        ----------
        plot_dtype : str, default=None.
            Key from {'Continuous', 'Binary', 'Nominal', 'Unknown'}.
            If None -> plot_columns will be used.

        plot_columns : list of strings, default=None.
            Which columns to plot. If None -> plot_dtype will be used.

        positive_name : str, default='default'
            Name of group with positive target.

        negative_name : str, default='good'
            Name of group with positive target.
        """
        data = self.data
        column_types = self.column_types
        target_column = self.target_column

        if plot_dtype is None:
            assert plot_columns is not None, 'At least one of the parameters must be set'
            assert type(plot_columns) == list, 'plot_columns must be list of strings'
        elif plot_columns is None:
            assert plot_dtype is not None, 'At least one of the parameters must be set'
            plot_columns = column_types[plot_dtype]
        else:
            raise AssertionError('Only one of parameters must be set')

        positive_mask = data[target_column] == 1
        negative_mask = data[target_column] == 0

        n_charts = len(plot_columns)
        fig = plt.figure(figsize=(16, 5 * n_charts))
        gs = fig.add_gridspec(n_charts, 2, hspace=0.4)

        # Plot chart for each column
        for i, column in enumerate(plot_columns):

            # Make numeric values string
            if data[column].dtype != object:
                data[column] = data[column].astype('str')

            # Make nan values string
            data[column] = data[column].fillna('NaN')

            # Find classes
            positive_cls = data[positive_mask][column]
            negative_cls = data[negative_mask][column]

            positive_dist = positive_cls.value_counts(normalize=True, dropna=False)
            negative_dist = negative_cls.value_counts(normalize=True, dropna=False)
            full_dist = data[column].value_counts(normalize=True, dropna=False)

            # Labels of bins
            all_labels = full_dist.index
            positive_labels = positive_dist.index
            negative_labels = negative_dist.index

            # Plot two charts in case of bad scale.
            # If frequency of most common a lot more than least one.
            if max(full_dist) > 10 * min(full_dist):
                # Find most common names and others.
                major_labels = [label for label in all_labels if
                                (full_dist[label] > 0.1 * max(full_dist))]
                # Labels sorted by values
                minor_labels = all_labels[len(major_labels):]

                major_values = {'positive': positive_dist[major_labels],
                                'negative': negative_dist[major_labels]}

                minor_values = {'positive': positive_dist[minor_labels],
                                'negative': negative_dist[minor_labels]}

                # Create subtitle for two charts
                ax_title = fig.add_subplot(gs[i, :], frameon=False)
                ax_title.set_xticks([])
                ax_title.set_yticks([])
                ax_title.set_title(column)

                # Create axes
                ax_maj = fig.add_subplot(gs[i, 0])
                ax_min = fig.add_subplot(gs[i, 1])

                # Plot most common labels
                plt.sca(ax_maj)
                plt.xticks(rotation=30)
                ax_maj.bar(major_labels, major_values['positive'],
                           label=positive_name, color='r', width=0.6)
                ax_maj.bar(major_labels, major_values['negative'],
                           label=negative_name, color='g', width=0.6, align='edge')
                plt.legend()

                # Plot least common labels
                plt.sca(ax_min)
                plt.xticks(rotation=30)
                ax_min.bar(minor_labels, minor_values['positive'],
                           label=positive_name, color='r', width=0.6)
                ax_min.bar(minor_labels, minor_values['negative'],
                           label=negative_name, color='g', width=0.6, align='edge')
                plt.legend()

            # Plot one chart in case of good scale.
            else:
                ax = fig.add_subplot(gs[i, :])
                plt.sca(ax)
                plt.title(column)
                ax.bar(positive_labels, positive_dist,
                       label=positive_name, color='r', width=0.6)
                ax.bar(negative_labels, negative_dist,
                       label=negative_name, color='g', width=0.6, align='edge')
                plt.xticks(rotation=30)
                plt.legend()
        plt.show()


if __name__ == '__main__':
    application_train = pd.read_csv('../data/application_train.csv')
    application_train_description = get_table_description(filename='../data/HomeCredit_columns_description.csv',
                                                          table_name='application_{train|test}.csv')

    process_data = BinaryClassEDA(data=application_train,
                                  table_description=application_train_description,
                                  target_column='TARGET',
                                  id_column='SK_ID_CURR')

    process_data.plot_categorical_dist_by_target('Binary')