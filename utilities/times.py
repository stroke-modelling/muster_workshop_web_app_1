"""
Calculations to do with time results.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_time_diff_admissions_grid(df_times, r=5):
    cols_index = df_times.index.names
    col_admissions = df_times.columns[0]
    df_times = df_times.reset_index()
    # Calculate time differences due to redir:
    df_times['diff_ivt'] = (df_times['redirection_approved_ivt'] -
                            df_times['usual_care_ivt'])
    df_times['diff_mt'] = (df_times['redirection_approved_mt'] -
                            df_times['usual_care_mt'])
    # Round to nearest "r" minutes:
    df_times['diff_ivt'] = (np.round(1e-4 + df_times['diff_ivt'] / r, 0)) * r
    df_times['diff_mt'] = (np.round(1e-4 + df_times['diff_mt'] / r, 0)) * r
    # Group time diffs:
    df_times = df_times.drop(cols_index, axis='columns')
    df_times = df_times.groupby(['diff_ivt', 'diff_mt']).sum().reset_index()
    # Convert to 2D grid:
    df_times = df_times.pivot(index='diff_ivt', columns='diff_mt', values=col_admissions)
    # Fill in any missing rows/columns:
    times_index = np.arange(df_times.index.values.min(), df_times.index.values.max() + r, r)
    times_index = times_index[times_index not in df_times.index.values]
    df_times.loc[times_index] = np.nan
    times_columns = np.arange(df_times.columns.min(), df_times.columns.max() + r, r)
    times_columns = [c for c in times_columns if c not in df_times.columns]
    df_times[times_columns] = np.nan
    df_times = df_times[sorted(df_times.columns)].sort_index()
    return df_times


def plot_time_diff_admissions_grid(df_times):
    ht = ''.join([
        'IVT %{y:+}min, MT %{x:+}min:<br>',
        '%{z} patients',
        '<extra></extra>'
    ])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=df_times.values,
        y=df_times.index,
        x=df_times.columns,
        colorbar={'title': 'Number of patients', 'title_side': 'right'},
        hovertemplate=ht,
        hoverongaps=False,
        ))
    t = 15
    lowest_tick_ivt = int(df_times.index.min() / t) * t
    lowest_tick_mt = int(min(df_times.columns) / t) * t
    fig.update_yaxes(
        title_text='Change in time to IVT (minutes)',
        tickmode='linear',
        tick0=lowest_tick_ivt,
        dtick=t,
        )
    fig.update_xaxes(
        title_text='Change in time to MT (minutes)',
        tickmode='linear',
        tick0=lowest_tick_mt,
        dtick=t,
        )
    title_text = ''.join([
        'Numbers of patients by change in treatment times<br>',
        'due to redirection.'
        ])
    fig.update_layout(title_text=title_text)
    st.plotly_chart(fig)


def calculate_quantiles(df_times, time_cols, region, quants, r=5):
    """
    """
    # Store results in here:
    list_quants = []
    for col in time_cols:
        # Make sure the times column is sorted:
        df_times = df_times.sort_values(col)
        # Calculate results.
        # Cumulative sum of numbers of patients:
        col_cumsum = f'{region}_cumsum'
        df_times[col_cumsum] = df_times[region].cumsum()
        # Results for this region will go in here:
        s_quants = pd.Series()
        for q in quants:
            # Convert fraction to number of patients:
            n_target = round(q * df_times[col_cumsum].max(), r)
            # Where is this condition met?
            m = df_times[col_cumsum] >= n_target
            # Pick out first time where condition met:
            t = df_times.loc[m, col].values[0]
            # Store:
            s_quants[q] = t
        # Store results for this region:
        s_quants.name = col
        list_quants.append(s_quants)
    # Convert results into dataframe:
    df_q = pd.concat(list_quants, axis='columns')
    df_q.index.name = 'Quantile'
    df_q.columns.name = 'Treatment time'
    return df_q
