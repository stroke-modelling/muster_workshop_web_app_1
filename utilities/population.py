"""
Build up patient populations.
"""
import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from utilities.outcomes import combine_lvo_ivt_mt_outcomes
from utilities.utils import print_progress_loc


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

    # Convert to percent:
    df_pops[number_cols] = df_pops[number_cols] * 100.0

    # Set up display:
    conf = dict()
    for col in number_cols:
        conf[col] = st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=1, format='%.0f%%'
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
            d[f'prop_{occ}'] * (1.0 - d[f'prop_redir_considered']))
        d[f'prop_{occ}_redir_accepted'] = (
            d[f'prop_{occ}'] *
            d[f'prop_redir_considered'] *
            (d[f'prop_redir_{occ}'])
            )
        d[f'prop_{occ}_redir_rejected'] = (
            d[f'prop_{occ}'] *
            d[f'prop_redir_considered'] *
            (1.0 - d[f'prop_redir_{occ}'])
            )
        s += d[f'prop_{occ}_redir_usual_care']
        s += d[f'prop_{occ}_redir_accepted']
        s += d[f'prop_{occ}_redir_rejected']

    if not round(s, 7) == 1.0:
        st.error(f'Check proportions for {occ}: {s}.')

    return d


def calculate_unique_outcomes_onion(
        dict_base_outcomes,
        df_base_lvo_ivt_mt_better,
        dict_onion,
        df_treat_times_sets_unique,
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
    utility_shift, mrs_dists_i for i in 0 to 6,
    mrs_dists_noncum_i for i in 0 to 6.
    """
    dict_outcomes = {}

    df_lvo_ivt_mt = combine_lvo_ivt_mt_outcomes(
        dict_base_outcomes['lvo_ivt'],
        dict_base_outcomes['lvo_mt'],
        df_base_lvo_ivt_mt_better,
        _log=False
    )

    # Combo nLVO and LVO outcomes for IVT only:
    dict_outcomes['usual_care_combo_ivt'] = (
        dict_base_outcomes['nlvo_ivt'] * dict_onion['prop_nlvo'] +
        dict_base_outcomes['lvo_ivt'] * dict_onion['prop_lvo']
    )
    # mRS distributions won't sum to 1 - CHECK THIS -------------------------------------------------
    dict_outcomes['usual_care_combo_mt'] = (
        dict_base_outcomes['lvo_mt'] * dict_onion['prop_lvo']
    )
    dict_outcomes['usual_care_combo_ivt_mt'] = (
        df_lvo_ivt_mt * dict_onion['prop_lvo']
    )
    # Usual care / redirection considered outcomes.
    # Take admissions-weighted average of the outcomes in the
    # base scenarios.
    scenarios = ['usual_care', 'redirection_approved',
                 'redirection_rejected']
    treats = ['ivt', 'mt']
    occs = ['nlvo', 'lvo']
    cols_treat_scen = [f'{s}_{t}' for s in scenarios for t in treats]
    base_scens = ['nlvo_ivt', 'lvo_ivt', 'lvo_mt']
    prop_lookup = {
        'usual_care': 'redir_usual_care',
        'redirection_approved': 'redir_accepted',
        'redirection_rejected': 'redir_rejected',
    }

    # Take a copy of each set of treatment times
    # and make a new version of the relevant outcomes
    # with repeated rows where necessary.
    # The indices of the resulting dataframes must match.
    df = df_treat_times_sets_unique.copy()

    # mRS distributions won't sum to 1 - CHECK THIS -------------------------------------------------
    dfs_to_combine = {}
    for scenario in scenarios:
        for base_scen in base_scens:
            occ, treat = base_scen.split('_')
            treat_time = f'{scenario}_{treat}'
            df_outcomes = pd.merge(
                df,
                dict_base_outcomes[base_scen],
                left_on=treat_time, right_on=f'time_to_{treat}',
                how='left'
            )
            df_outcomes = df_outcomes.drop(f'time_to_{treat}',
                                           axis='columns')
            df_outcomes = df_outcomes.set_index(cols_treat_scen)
            prop_key = f'prop_{occ}_{prop_lookup[scenario]}'
            # CHECK THIS - what about proportions who receive IVT, MT?
            # LVO is currently being counted twice so proportions don't
            # sum to 1. Adding the full LVO population instead of just
            # IVT plus just MT plus both IVT and MT.
            # Maybe in props need to include prop of LVO/nLVO who
            # receive IVT/MT/both... --------------------------------------------------TO DO
            k = f'{scenario}_{base_scen}'
            dfs_to_combine[k] = {}
            dfs_to_combine[k]['prop'] = dict_onion[prop_key]
            dfs_to_combine[k]['df'] = df_outcomes
    df = sum([dfs_to_combine[k]['prop'] * dfs_to_combine[k]['df']
              for k in dfs_to_combine.keys()])
    prop_sum = sum([dfs_to_combine[k]['prop']
                    for k in dfs_to_combine.keys()])
    st.write('eg')
    st.write(prop_sum)
    st.write(df)
    st.write('egg')



    if _log:
        p = 'Calculated unique outcomes for this population.'
        print_progress_loc(p, _log_loc)
    return dict_outcomes