"""
Build up patient populations.
"""
import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utilities.utils import print_progress_loc, update_plotly_font_sizes


def select_onion_population():
    """
    Resulting series keys: population, label, prop_nlvo, prop_lvo,
    prop_other, prop_redir_considered, redir_sensitivity,
    redir_specificity.
    """
    # Load in pathway timings from file:
    f = './data/speedy_populations.csv'
    df_pops = pd.read_csv(f)

    number_cols = [c for c in df_pops.columns if is_numeric_dtype(df_pops[c])]

    # # Convert to percent:
    df_pops[number_cols] = df_pops[number_cols] * 100.0

    # Set up display:
    conf = dict()
    for col in number_cols:
        conf[col] = st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=1.0,
            # format='percent',
            format='%.0f%%'
        )

    df_pops = st.data_editor(
        df_pops,
        column_order=['label'] + number_cols,
        # hide_index=True,
        column_config=conf,
        num_rows='dynamic',
    )

    # Convert to proportions:
    df_pops[number_cols] = df_pops[number_cols] / 100.0
    # Sanity check added row labels:
    if len(df_pops['label'].unique()) != len(df_pops):
        st.error('Please rename any rows with duplicate names.')
    else:
        pass
    # Create index names for any added rows:
    mask = df_pops['population'].isna()
    df_pops.loc[mask, 'population'] = (
        df_pops.loc[mask, 'label'].str.lower()
        .replace(' ', '_', regex=True)
        )

    # Set up for layer selection:
    try:
        ind_default = int(np.where(
            df_pops['population'] == 'green_ischaemic')[0][0])
    except IndexError:
        # The value we're looking for has been deleted.
        ind_default = 0
    dict_labels = dict(zip(df_pops['population'], df_pops['label']))
    if 'null' in dict_labels.keys():
        # No label given to a custom row.
        # Change label from default null type so that f() labelling
        # function doesn't return a typeerror later.
        dict_labels['null'] = 'null'

    def f(label):
        """Display layer with nice name instead of key."""
        return dict_labels[label]
    # Pick a layer to use for calculating population results:
    layer_key = st.selectbox(
        'Choose a population for the outcome results.',
        options=df_pops['population'],
        format_func=f,
        index=ind_default,
        )
    # Pick out variables for this layer:
    series_chosen_pops = df_pops.loc[df_pops['population'] == layer_key].squeeze()

    # Sanity check added row props:
    # (do this now because it's irritating to do it sooner before the
    # new row has been completed.)
    cols_occ = ['prop_nlvo', 'prop_lvo', 'prop_other']
    if (series_chosen_pops[cols_occ].sum() == 1):
        pass
    else:
        st.error('Please check that stroke type proportions sum to 1.')
    return series_chosen_pops


def calculate_population_subgroups(d):
    """
    Combine the existing proportions to find subgroups.
    """
    # Proportions redirected:
    d['prop_redir_lvo'] = d['redir_sensitivity']
    d['prop_redir_nlvo'] = (1.0 - d['redir_specificity'])

    s = 0.0  # checksum
    for occ in ['nlvo', 'lvo']:
        d[f'prop_{occ}_redir_usual_care'] = (
            d[f'prop_{occ}'] * (1.0 - d['prop_redir_considered']))
        d[f'prop_{occ}_redir_accepted'] = (
            d[f'prop_{occ}'] *
            d['prop_redir_considered'] *
            (d[f'prop_redir_{occ}'])
            )
        d[f'prop_{occ}_redir_rejected'] = (
            d[f'prop_{occ}'] *
            d['prop_redir_considered'] *
            (1.0 - d[f'prop_redir_{occ}'])
            )
        s += d[f'prop_{occ}_redir_usual_care']
        s += d[f'prop_{occ}_redir_accepted']
        s += d[f'prop_{occ}_redir_rejected']

    if not round(s, 7) == 1.0:
        st.error(f'Check proportions for {occ}: {s}.')

    return d


def select_subgroups_for_results():

    # Load in subgroup names and labels from file:
    f = './data/subgroup_names.csv'
    df_subgroups = pd.read_csv(f, index_col='label')
    df_subgroups = df_subgroups.drop('subgroup', axis='columns')

    # Make index values:
    label_cols = []
    for c in df_subgroups.columns:
        lc = f'{c}_label'
        label_cols.append(lc)
        c_dict = {1: c, 0: ''}
        df_subgroups[lc] = df_subgroups[c].map(c_dict)
    df_subgroups['subgroup'] = (
        df_subgroups[label_cols].apply(lambda x: '_'.join(x), axis=1))
    # Remove leading and trailing _ and repeated _:
    df_subgroups['subgroup'] = df_subgroups['subgroup'].str.strip('_')
    for i in range(len(label_cols), 1, -1):
        df_subgroups['subgroup'] = (
            df_subgroups['subgroup'].str.replace('_'*i, '_'))
    df_subgroups = df_subgroups.drop(label_cols, axis='columns')

    # Set up which occlusion/treatment combinations are included:
    occs_tres = ['nlvo_ivt', 'nlvo_no_treatment',
                 'lvo_ivt', 'lvo_mt', 'lvo_ivt_mt', 'lvo_no_treatment']
    for ot in occs_tres:
        o = ot.split('_')[0]
        t = '_'.join(ot.split('_')[1:])
        df_subgroups[f'{o}_{t}'] = df_subgroups[o] * df_subgroups[t]

    # Select which groups to use:
    df_subgroups = df_subgroups.reset_index().set_index('subgroup')
    dict_labels = df_subgroups['label'].to_dict()

    def f(label):
        """Display layer with nice name instead of key."""
        return dict_labels[label]
    list_selected_subgroups = st.multiselect(
        'Subgroups to calculate outcomes for',
        dict_labels.keys(),
        format_func=f,
        default='nlvo_lvo_ivt_mt_ivt_mt'
    )
    df_subgroups = df_subgroups.loc[list_selected_subgroups]

    return df_subgroups


def calculate_population_subgroup_grid(d, df_subgroups=None):
    """

    There are two sets of "usual care". One is for the current
    real-life setup, and the other is for the redirection scenario
    when some patients are not considered for redirection.
    """
    # Each individual subgroup.
    treatment_groups = [
        'nlvo_ivt', 'nlvo_no_treatment',
        'lvo_ivt', 'lvo_mt', 'lvo_ivt_mt', 'lvo_no_treatment',
        ]

    # Extra proportions to let all props be looked up in the
    # same way:
    d = d.copy()
    d['prop_nlvo_no_treatment'] = (1.0 - d['prop_nlvo_ivt'])
    d['prop_lvo_no_treatment'] = (1.0 - (
        d['prop_lvo_ivt'] + d['prop_lvo_mt'] + d['prop_lvo_ivt_mt']))

    # Full population, usual care:
    s = pd.Series()
    s.name = 'full_population'
    for tre in treatment_groups:
        occ = tre.split('_')[0]
        s[tre] = d[f'prop_{occ}'] * d[f'prop_{tre}']
    # Store separate results for "usual care" by itself:
    df_pop_usual_care = pd.DataFrame(s)
    df_pop_usual_care.index.name = 'treatment_group'
    # Scenario label for consistency with redir df:
    df_pop_usual_care.insert(0, 'scenario', 'usual_care')

    # Full population, redir options:
    redir_scenarios = [
        'redir_usual_care', 'redir_accepted', 'redir_rejected']
    props = {}
    for scen in redir_scenarios:
        r = pd.Series()
        r.name = 'full_population'
        for tre in treatment_groups:
            occ = tre.split('_')[0]
            p = d[f'prop_{occ}_{scen}']
            r[f'{tre}'] = p * d[f'prop_{tre}']
        props[scen] = r
    # Gather this full population into one dataframe:
    df_pop_redir = pd.DataFrame.from_dict(props)
    df_pop_redir.index.name = 'treatment_group'
    df_pop_redir.columns.name = 'scenario'
    df_pop_redir = df_pop_redir.unstack()
    df_pop_redir = (pd.DataFrame(df_pop_redir).rename(
        columns={0: 'full_population'})
        .reset_index().set_index('treatment_group'))

    # Full population, only redirection approved:
    scen = 'redir_accepted'
    r = pd.Series()
    r.name = 'full_population'
    for tre in treatment_groups:
        occ = tre.split('_')[0]
        p = d[f'prop_{occ}_{scen}']
        r[f'{tre}'] = p * d[f'prop_{tre}']
    # Normalise:
    r = r / r.sum()
    # Store separate results for "redir approved only" by itself:
    df_pop_redir_approved_only = pd.DataFrame(s)
    df_pop_redir_approved_only.index.name = 'treatment_group'
    # Scenario label for consistency with redir df:
    df_pop_redir_approved_only.insert(0, 'scenario', 'redir_accepted_only')

    # Gather the new props dataframes:
    list_of_dfs = [df_pop_usual_care, df_pop_redir,
                   df_pop_redir_approved_only]

    # Split "treatment group" column into occlusion / treatment
    # columns. To match df_subgroups.
    cols_bits = ['nlvo', 'lvo', 'ivt', 'mt', 'ivt_mt', 'no_treatment']
    for df in list_of_dfs:
        # Set treatment group as index to make loc easier:
        i_col = df.index.name
        df.reset_index(inplace=True)
        df.set_index('treatment_group', inplace=True)
        # Start with all columns 0...
        df[cols_bits] = 0
        # ... then update the values for each subgroup:
        for t in df.index:
            if t.startswith('nlvo'):
                df.loc[t, 'nlvo'] = 1
            elif t.startswith('lvo'):
                df.loc[t, 'lvo'] = 1
            if 'no_treatment' in t:
                df.loc[t, 'no_treatment'] = 1
            elif 'ivt_mt' in t:
                df.loc[t, 'ivt_mt'] = 1
            elif 'ivt' in t:
                df.loc[t, 'ivt'] = 1
            elif 'mt' in t:
                df.loc[t, 'mt'] = 1
        # Undo earlier index change:
        df.reset_index(inplace=True)
        df.set_index(i_col, inplace=True)

    # Calculate proportions for the selected subgroups.
    for sub_name in df_subgroups.index:
        s = df_subgroups.loc[sub_name]
        for df in list_of_dfs:
            # Start with all rows allowed...
            mask = np.full(len(df), True)
            # ... then remove rows that don't match setup:
            for c in cols_bits:
                if s[c] == 0:
                    mask[df[c] == 1] = 0
            # Copy over values from only allowed rows:
            df[sub_name] = df['full_population'] * mask
            # Normalise:
            df[sub_name] = df[sub_name] / df[sub_name].sum()

    return tuple(list_of_dfs)


def plot_population_props(
        props_usual_care,
        props_redir,
        s,
        subgroup_setup
        ):
    titles = ['Usual care', 'Redirection available']
    fig = make_subplots(rows=2, cols=1, subplot_titles=titles)

    fig.add_trace(go.Bar(
        x=props_usual_care.index,
        y=100.0 * props_usual_care[s],
        name='Usual care'
    ), row=1, col=1)

    redir_scenarios = props_redir['scenario'].unique()
    redir_labels = {
        'redir_usual_care': 'Usual care',
        'redir_accepted': 'Accept redirection',
        'redir_rejected': 'Reject redirection',
    }
    for scenario in redir_scenarios:
        m = props_redir['scenario'] == scenario
        fig.add_trace(go.Bar(
            x=props_redir.loc[m].index,
            y=100.0 * props_redir.loc[m, s],
            name=redir_labels[scenario],
        ), row=2, col=1)

    fig.update_layout(barmode='group')
    fig.update_layout(height=600, width=600,
                      title_text=subgroup_setup['label'])
    fig.update_xaxes(title_text='Treatment group', row=2, col=1)
    for i in [1, 2]:
        fig.update_yaxes(
            title_text='Percentage of subgroup patients', row=i, col=1)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.3,
        xanchor="center",
        x=0.5,
    ))
    fig = update_plotly_font_sizes(fig)
    st.plotly_chart(fig)


def calculate_unique_outcomes_onion(
        dict_base_outcomes,
        df_pop_usual_care,
        df_pop_redir,
        df_pop_redir_accepted_only,
        df_subgroups,
        df_treat_times_sets_unique,
        s,
        check_mrs_noncum=False,
        _log=True, _log_loc=None,
        ):
    """
    Combine the base outcomes into nLVO+LVO and combined scenarios
    (mix of usual care, redirection rejected, redirection approved).

    Population dict keys: population, label, prop_nlvo, prop_lvo,
    prop_other, prop_redir_considered, redir_sensitivity,
    redir_specificity, prop_redir_lvo, prop_redir_nlvo,
    prop_X_redir_usual_care, prop_X_redir_accepted,
    prop_X_redir_rejected for X in nlvo, lvo.
    Each base outcome df has the keys:
    time_to_ivt or time_to_mt, mrs_0-2, mrs_shift,
    utility_shift, mrs_dists_i for i in 0 to 6

    Note: checked the mRS distribution combination.
    The result is the same for combining non-cumulative scores
    as for combining cumulative scores and then taking diff.
    """
    # Which outcome data do we need?
    base_outcome_keys = list(dict_base_outcomes.keys())
    base_outcome_keys_here = [
        k for k in base_outcome_keys if df_subgroups[k] == 1]

    # Match scenario with simpler proportions dataframes:
    dict_scenario_props = {
        'usual_care': df_pop_usual_care,
        'redir_usual_care': df_pop_redir.loc[
            df_pop_redir['scenario'] == 'redir_usual_care'],
        'redir_accepted': df_pop_redir.loc[
            df_pop_redir['scenario'] == 'redir_accepted'],
        'redir_rejected': df_pop_redir.loc[
            df_pop_redir['scenario'] == 'redir_rejected'],
        'redir_accepted_only': df_pop_redir_accepted_only,
    }

    # Match the scenario names with the treatment times
    # in the outcomes data.
    time_lookup = {
        'usual_care': 'usual_care',
        'redir_usual_care': 'usual_care',
        'redir_accepted': 'redirection_approved',
        'redir_rejected': 'redirection_rejected',
        'redir_accepted_only': 'redirection_approved',
    }
    time_cols = list(df_treat_times_sets_unique.columns)

    def gather_outcomes_and_props(
            scenarios,
            dict_scenario_props,
            df_treat_times_sets_unique,
            base_outcome_keys_here,
            time_lookup
            ):
        """
        """
        # Take a copy of each set of treatment times
        # and make a new version of the relevant outcomes
        # with repeated rows where necessary.
        # The indices of the resulting dataframes must match.
        df = df_treat_times_sets_unique.copy()
        dfs_to_combine = {}
        for scenario in scenarios:
            props = dict_scenario_props[scenario]
            for base_scen in base_outcome_keys_here:
                occ = base_scen.split('_')[0]
                treat = '_'.join(base_scen.split('_')[1:])

                if treat == 'no_treatment':
                    # Make as many copies of the no-treatment data
                    # as there are rows of treatment times.
                    df_out = dict_base_outcomes[base_scen]
                    # Set up values:
                    arr = np.tile(df_out.values, len(df)).reshape(
                        len(df), len(df_out.columns))
                    # Set up index:
                    df_idx = df.copy()
                    df_idx = df_idx.set_index(list(df_idx.columns))
                    # Make new dataframe.
                    # Reset index because later we set it again (!).
                    df_outcomes = pd.DataFrame(
                        arr,
                        index=df_idx.index,
                        columns=df_out.columns
                    ).reset_index()
                elif treat == 'ivt_mt':
                    # Have to check against the time to IVT and the
                    # time to MT.
                    treat_times = [f'{time_lookup[scenario]}_ivt',
                                   f'{time_lookup[scenario]}_mt']
                    time_tos = ['time_to_ivt', 'time_to_mt']

                    rename_dict = dict(zip(time_tos, treat_times))
                    df_out = (dict_base_outcomes[base_scen].reset_index()
                              .rename(columns=rename_dict))
                    df_outcomes = pd.merge(
                        df,
                        df_out,
                        on=treat_times,
                        how='left'
                    )
                    # Drop unwanted columns with no counterparts in the
                    # other outcome data.
                    cols_ivt_mt = ['ivt_better', 'mrs_0-2_ivt', 'mrs_0-2_mt']
                    df_outcomes = df_outcomes.drop(cols_ivt_mt, axis='columns')
                else:
                    treat_time = f'{time_lookup[scenario]}_{treat}'
                    df_outcomes = pd.merge(
                        df,
                        dict_base_outcomes[base_scen],
                        left_on=treat_time, right_on=f'time_to_{treat}',
                        how='left'
                    )
                    df_outcomes = df_outcomes.drop(
                        f'time_to_{treat}', axis='columns')
                if check_mrs_noncum:
                    # Sanity check:
                    # Convert mRS distributions to non-cumulative:
                    mrs_cols = [f'mrs_dists_{i}' for i in range(7)]
                    mrs_noncum_cols = [c.replace('dists_', 'dists_noncum_')
                                    for c in mrs_cols]
                    df_outcomes[mrs_noncum_cols] = np.diff(
                        df_outcomes[mrs_cols], prepend=0.0)

                # All outcome dfs must share these time columns:
                df_outcomes = df_outcomes.set_index(time_cols)
                # Store for later combination:
                k = f'{scenario}_{base_scen}'
                dfs_to_combine[k] = {}
                dfs_to_combine[k]['prop'] = props.loc[base_scen, s]
                dfs_to_combine[k]['df'] = df_outcomes

        df = sum([dfs_to_combine[k]['prop'] * dfs_to_combine[k]['df']
                  for k in dfs_to_combine.keys()])
        # Sanity check:
        prop_sum = sum([dfs_to_combine[k]['prop']
                        for k in dfs_to_combine.keys()])
        if np.round(prop_sum, 5) != 1.0:
            st.error('Check outcome combination proportions.')
        return df

    df_usual_care = gather_outcomes_and_props(
        ['usual_care'],
        dict_scenario_props,
        df_treat_times_sets_unique,
        base_outcome_keys_here,
        time_lookup
        )
    df_redir = gather_outcomes_and_props(
        ['redir_usual_care', 'redir_accepted', 'redir_rejected'],
        dict_scenario_props,
        df_treat_times_sets_unique,
        base_outcome_keys_here,
        time_lookup
        )
    df_redir_accepted_only = gather_outcomes_and_props(
        ['redir_accepted_only'],
        dict_scenario_props,
        df_treat_times_sets_unique,
        base_outcome_keys_here,
        time_lookup
        )

    if check_mrs_noncum:
        mrs_cols = [f'mrs_dists_{i}' for i in range(7)]
        mrs_noncum_cols = [c.replace('dists_', 'dists_noncum_')
                           for c in mrs_cols]

        st.write('eg')
        st.write(df_usual_care)
        c = pd.DataFrame(
            np.diff(df_usual_care[mrs_cols], prepend=0.0),
            columns=mrs_noncum_cols,
            index=df_usual_care.index
        )
        st.write(c)
        st.write(df_usual_care[mrs_noncum_cols])
        df_check = (np.round(c, 5) ==
                    np.round(df_usual_care[mrs_noncum_cols], 5))
        st.write(df_check)
        st.write(df_check.sum(axis='columns'))
        st.write('egg')

    if _log:
        p = 'Calculated unique outcomes for this population.'
        print_progress_loc(p, _log_loc)
    return {
        'usual_care': df_usual_care,
        'redir_allowed': df_redir,
        'redir_accepted_only': df_redir_accepted_only,
        }


def gather_lsoa_level_outcomes(
        dict_outcomes,
        df_lsoa_units_times,
        _log=True, _log_loc=None
        ):
    """

    Don't calculate the transfer-only subset
    because the "transfer_required" column is in
    df_lsoa_units_times.
    """
    redir_scens = ['usual_care', 'redirection_approved',
                   'redirection_rejected']
    treats = ['ivt', 'mt']
    cols_times = [f'{s}_{t}' for s in redir_scens for t in treats]
    df_lsoa = df_lsoa_units_times.set_index(cols_times)

    dict_lsoa = {}
    for subgroup_name, dict_subgroup in dict_outcomes.items():
        dfs_here = [df_lsoa['LSOA']]
        for scenario, df_outcomes in dict_subgroup.items():
            cols_outcomes = list(df_outcomes.columns)
            rename_dict = dict([(c, f'{c}_{scenario}') for c in cols_outcomes])
            df = df_outcomes.rename(columns=rename_dict)
            dfs_here.append(df)

        df = pd.concat(dfs_here, axis='columns')
        df = df.reset_index().set_index('LSOA')
        df = df.drop(cols_times, axis='columns')
        dict_lsoa[subgroup_name] = df
    if _log:
        p = 'Gathered full LSOA-level results.'
        print_progress_loc(p, _log_loc)
    return dict_lsoa
