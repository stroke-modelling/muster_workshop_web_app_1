"""
Streamlit app template.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

# Custom functions:
from utilities.fixed_params import page_setup
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Title:
st.markdown('# MUSTER')


# ###########################
# ########## SETUP ##########
# ###########################

# Draw the input selection boxes in this function:
with st.sidebar:
    st.header('Inputs')
    input_dict = inputs.select_parameters()

# Which scenario ID has this combination of parameters?
scenario_id = inputs.find_scenario_id(input_dict)

# Pick out results for this scenario ID:
results_dict = inputs.find_scenario_results(scenario_id)

pathway_dicts = results.split_results_dict_by_pathway(results_dict)

results_all = pathway_dicts['all']
results_drip_ship = pathway_dicts['drip_ship']
results_mothership = pathway_dicts['mothership']
results_msu = pathway_dicts['msu']

# Parameters shared across all pathway types:
df_results_all = pd.DataFrame.from_dict(
    [results_all], orient='columns').T

container_shared_data = st.container()

# DataFrame with one row per pathway type:
df_results = pd.DataFrame.from_dict(
    [results_drip_ship, results_mothership, results_msu],
    orient='columns',
)
df_results.index = ['Drip & ship', 'Mothership', 'MSU']

group_by = st.radio(
    'Group results by:',
    ['Stroke type', 'Outcome type']
    )

if group_by == 'Stroke type':
    # Pick out the occlusion and treatment types and stick them
    # in a MultiIndex header.
    split_list = ['lvo_ivt_mt_', 'nlvo_ivt_', 'lvo_ivt_', 'lvo_mt_']
    new_cols = results.make_multiindex_stroke_type(
        df_results.columns, split_list)
    tuples = list(zip(*new_cols))
    df_results.columns = pd.MultiIndex.from_tuples(tuples)

    # Pick out the occlusion and treatment types and stick them
    # into separate DataFrames.
    df_results_nlvo_ivt = df_results['nlvo_ivt']
    df_results_lvo_ivt = df_results['lvo_ivt']
    df_results_lvo_mt = df_results['lvo_mt']
    df_results_lvo_ivt_mt = df_results['lvo_ivt_mt']
    df_results_other = df_results['']

    st.markdown('### nLVO IVT')
    style_dict = results.make_column_style_dict(
        df_results_nlvo_ivt.index, format='%.3f')
    st.dataframe(
        df_results_nlvo_ivt.T,
        column_config=style_dict
        )

    st.markdown('### LVO IVT')
    style_dict = results.make_column_style_dict(
        df_results_lvo_ivt.index, format='%.3f')
    st.dataframe(
        df_results_lvo_ivt.T,
        column_config=style_dict
        )

    st.markdown('### LVO MT')
    style_dict = results.make_column_style_dict(
        df_results_lvo_mt.index, format='%.3f')
    st.dataframe(
        df_results_lvo_mt.T,
        column_config=style_dict
        )

    st.markdown('### LVO IVT & MT')
    style_dict = results.make_column_style_dict(
        df_results_lvo_ivt_mt.index, format='%.3f')
    st.dataframe(
        df_results_lvo_ivt_mt.T,
        column_config=style_dict
        )
else:
    # Pick out the outcome types and stick them
    # in a MultiIndex header.
    split_list = ['utility_shift', 'mrs_0-2', 'mrs_shift', 'utility']
    new_cols = results.make_multiindex_stroke_type(
        df_results.columns, split_list)
    tuples = list(zip(*new_cols))
    df_results.columns = pd.MultiIndex.from_tuples(tuples)

    # Pick out the occlusion and treatment types and stick them
    # into separate DataFrames.
    df_results_mrs_02 = df_results['mrs_0-2']
    df_results_mrs_shift = df_results['mrs_shift']
    df_results_utility = df_results['utility']
    df_results_utility_shift = df_results['utility_shift']
    df_results_other = df_results['']

    st.markdown('### mRS 0-2')
    style_dict = results.make_column_style_dict(
        df_results_mrs_02.index, format='%.3f')
    st.dataframe(
        df_results_mrs_02.T,
        column_config=style_dict
        )

    st.markdown('### mrs_shift')
    style_dict = results.make_column_style_dict(
        df_results_mrs_shift.index, format='%.3f')
    st.dataframe(
        df_results_mrs_shift.T,
        column_config=style_dict
        )

    st.markdown('### utility')
    style_dict = results.make_column_style_dict(
        df_results_utility.index, format='%.3f')
    st.dataframe(
        df_results_utility.T,
        column_config=style_dict
        )

    st.markdown('### utility_shift')
    style_dict = results.make_column_style_dict(
        df_results_utility_shift.index, format='%.3f')
    st.dataframe(
        df_results_utility_shift.T,
        column_config=style_dict
        )

with container_shared_data:
    st.markdown('## Fixed values')
    cols = ['nlvo_no_treatment_mrs_0-2', 'nlvo_no_treatment_utility']
    df_outcomes = df_results_all.loc[cols].T
    style_dict = results.make_column_style_dict(
        df_outcomes.columns, format='%.3f')
    st.dataframe(
        df_outcomes,
        column_config=style_dict,
        hide_index=True
        )

    # Travel times:
    cols = ['nearest_ivt_time', 'nearest_mt_time', 'transfer_time']
    df_travel = df_results_all.loc[cols].T
    style_dict = results.make_column_style_dict(
        df_travel.columns, format='%d')
    st.dataframe(
        df_travel,
        column_config=style_dict,
        hide_index=True
        )

    st.markdown('## This scenario:')
    # Times to treatment:
    cols = ['ivt_time', 'mt_time']
    df_times = df_results_other[cols]
    style_dict = results.make_column_style_dict(
        df_times.index, format='%d')
    st.dataframe(
        df_times.T,
        column_config=style_dict,
        # hide_index=True
        )

    # MSU bits:
    cols = ['nearest_time', 'occupied_treatment', 'occupied_no_treatment']
    # Extra pd.DataFrame() here otherwise streamlit sees it's a Series
    # and overrides the style dict.
    df_msu = pd.DataFrame(df_results_other.loc['MSU', cols])
    style_dict = results.make_column_style_dict(
        df_msu.index, format='%d')
    st.dataframe(
        df_msu.T,
        column_config=style_dict,
        # hide_index=True
        )


# ----- The end! -----
