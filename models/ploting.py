import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_categorical_bars(column_names,
                          data: pd.DataFrame):
    """
    Plots distribution bars of selected columns from data.
    Columns must contain categorical features - ordinal, nominal, binary.

    Arguments:
        column_names - list of strings, column names of categorical data
        data - pd.DataFrame
    """
    # Default dataset
    if data is None:
        data = application_train.copy()

    # Default columns
    if column_names is None:
        column_names = cat_columns
    elif type(column_names) != list:
        column_names = [column_names]
    else:
        pass

    # Find number of charts
    n_charts = len(column_names)
    plt_nrows = n_charts

    # Create pyplot figure
    fig = plt.figure(figsize=(16, 5 * plt_nrows))
    gs = fig.add_gridspec(plt_nrows, 2, hspace=0.4)

    # Plot chart for each column
    for i, column in enumerate(column_names):

        # Make labels string
        if data[column].dtype != object:
            data[column] = data[column].astype('str')

        # Find distribution
        def_vals = data[data.TARGET == 1][column].value_counts(normalize=True)
        undef_vals = data[data.TARGET == 0][column].value_counts(normalize=True)
        all_vals = data[column].value_counts(normalize=True)

        # Labels of bins
        all_labels = all_vals.index
        def_labels = def_vals.index
        undef_labels = undef_vals.index

        # Plot two charts if frequency of most common a lot more than least one (for zoom).
        if max(all_vals) > 10 * min(all_vals):
            # Find most common names and others
            major_labels = [label for label in all_labels if
                            (all_vals[label] > 0.1 * max(all_vals))]
            minor_labels = all_labels[len(major_labels):]

            # Dicts with values of distribution
            major_values = {'def': def_vals[major_labels],
                            'undef': undef_vals[major_labels]}

            minor_values = {'def': def_vals[minor_labels],
                            'undef': undef_vals[minor_labels]}

            # Create suptitle for two charts
            ax_title = fig.add_subplot(gs[i, :], frameon=False)
            ax_title.set_xticks([])
            ax_title.set_yticks([])
            ax_title.set_title(column)

            # Create axes
            ax_maj = fig.add_subplot(gs[i, 0])
            ax_min = fig.add_subplot(gs[i, 1])

            plt.sca(ax_maj)
            plt.xticks(rotation=30)
            ax_maj.bar(major_labels, major_values['def'], label='default', color='r', width=0.6)
            ax_maj.bar(major_labels, major_values['undef'], label='good', color='g', width=0.6,
                       align='edge')
            plt.legend()

            plt.sca(ax_min)
            plt.xticks(rotation=30)
            ax_min.bar(minor_labels, minor_values['def'], label='default', color='r', width=0.6)
            ax_min.bar(minor_labels, minor_values['undef'], label='good', color='g', width=0.6,
                       align='edge')
            plt.legend()

        # Plot one chart.
        else:
            ax = fig.add_subplot(gs[i, :])
            plt.sca(ax)
            plt.title(column)
            ax.bar(def_labels, def_vals, label='default', color='r', width=0.6)
            ax.bar(undef_labels, undef_vals, label='good', color='g', width=0.6, align='edge')
            plt.xticks(rotation=30)
            plt.legend()
    plt.show()