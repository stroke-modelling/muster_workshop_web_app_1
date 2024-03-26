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

# Separate the fixed parameters
# (currently in results data for some reason):
fixed_keys = [
    'nearest_ivt_time', 'nearest_mt_time', 'transfer_time',
    'nearest_msu_time', 'Admissions', 'England',
    'nlvo_no_treatment_mrs_0-2', 'nlvo_no_treatment_utility',
    'lvo_no_treatment_mrs_0-2', 'lvo_no_treatment_utility',
    ]
fixed_dict = dict([(k, results_dict[k]) for k in fixed_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys()) if k not in fixed_keys])

# Separate times and outcomes:
time_keys = [
    'drip_ship_ivt_time', 'drip_ship_mt_time',
    'mothership_ivt_time', 'mothership_mt_time',
    'msu_ivt_time', 'msu_mt_time'
]
treatment_time_dict = dict([(k, results_dict[k]) for k in time_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys()) if k not in time_keys])

# Separate pathways:
pathway_dicts = results.split_results_dict_by_pathway(results_dict)
results_all = pathway_dicts['all']
results_drip_ship = pathway_dicts['drip_ship']
results_mothership = pathway_dicts['mothership']
results_msu = pathway_dicts['msu']

# Make a container for parameters shared across all pathway types.
# The results will be drawn first but this container appears higher.
container_shared_data = st.container()

# DataFrame with one row per pathway type:
df_results = pd.DataFrame.from_dict(
    [results_drip_ship, results_mothership, results_msu],
    orient='columns',
)

 st.markdown('### Outcomes ###')
 
df_results.index = ['Drip & ship', 'Mothership', 'MSU']

# User inputs for how to display the data:
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
    st.markdown('Baseline outcomes')

    cols = ['nlvo_no_treatment_mrs_0-2', 'nlvo_no_treatment_utility',
        'lvo_no_treatment_mrs_0-2', 'lvo_no_treatment_utility',]
    df_outcomes = pd.Series(dict([(k, fixed_dict[k]) for k in cols]))
    style_dict = results.make_column_style_dict(
        df_outcomes.index, format='%.3f')
    st.dataframe(
        pd.DataFrame(df_outcomes).T,
        column_config=style_dict,
        hide_index=True
        )

    # Travel times:
    st.markdown('Average travel times (minutes) to closest units')
    cols = ['nearest_ivt_time', 'nearest_mt_time',
            'transfer_time', 'nearest_msu_time']
    df_travel = pd.Series(dict([(k, fixed_dict[k]) for k in cols]))
    style_dict = results.make_column_style_dict(
        df_travel.index, format='%d')
    st.dataframe(
        pd.DataFrame(df_travel).T,
        column_config=style_dict,
        hide_index=True
        )

    st.markdown('## This scenario')
    
    st.markdown('### Treatment times ###')
    
    st.markdown('Average times (minutes) to treatment')
    # Times to treatment:
    columns = ['drip_ship', 'mothership', 'msu']
    index = ['ivt', 'mt']
    table = [[0, 0, 0], [0, 0, 0]]
    for c, col in enumerate(columns):
        for i, ind in enumerate(index):
            key = f'{col}_{ind}_time'
            table[i][c] = int(round(treatment_time_dict[key], 0))
    df_times = pd.DataFrame(table, columns=columns, index=index)
    style_dict = results.make_column_style_dict(
        df_times.columns, format='%d')
    st.dataframe(
        df_times,
        column_config=style_dict,
        # hide_index=True
        )

    # MSU bits:
    
    st.markdown('### MSU Use ###')
     
    st.markdown('MSU use time (minutes) per patient')
    cols = ['occupied_treatment', 'occupied_no_treatment']
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
