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
from utilities.plot_timeline import build_data_for_timeline, draw_timeline
import utilities.utils as utils
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Title:
st.markdown('# MUSTER')


import streamlit as st

st.markdown('''
This model shows predicted outcomes for non-large vessel occlusion (nLVO) and large vessel occlusion 
stroke. Outcomes are calculated for 34,000 small areas (LSOAs) across England based on expected 
travel times, and other timing parameters chosen by the slider bars on the right.

More detail may be found on estimation of stroke outcomes [here](https://samuel-book.github.io/stroke_outcome/intro.html). 
The reported outcomes are for treated patients (they do not include patients unsuitable for treatment, 
haemorrhagic strokes, or mimics)

Three pathways are modelled, through to thrombectomy (note: thrombectomy is only applied to large 
vessel occlusions; non-large vessel occlusions are treated with thrombolysis only). The three pathways are:

1) *Drip-and-ship*: All patients are taken to their closest emergency stroke unit, all of which 
provide thrombolysis. For patients who receive thrombectomy there is a transfer to a thrombectomy-capable 
if the patient has first attended a hopsital that provides thrombolysis only.

2) *Mothership*: All patients are taken to a comprehensive stroke centre that can provide both 
thrombolysis and thrombectomy.

3) *Mobile stroke unit (MSU)*: MSUs are dispatched, from comprehensive stroke centres, to stroke patients. 
Head scans and thrombolysis are provided on-scene, where the patient is. For patients who have been 
treated with thrombolysis or considered suitable for thrombectomy, the MSU takes the patient to the 
comprehensive stroke centre. Where a patient does not receive thrombolysis, and is not considered 
a candidate for thrombectomy, the MSU becomes available for another stroke patient, and a standard 
ambulance conveys the patient to the closest emergency stroke unit. In this particular model there 
are no capacity limits for the MSU, and it is assumed all strokes are triaged correctly with the 
emergency call - the model shows outcomes if all patients were seen by a MSU.
''')

st.image('./pages/images/stroke_treatment.jpg')


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
    'nearest_ivt_time',
    'nearest_mt_time',
    'transfer_time',
    'nearest_msu_time',
    'Admissions',
    'England',
    'nlvo_no_treatment_mrs_0-2',
    'nlvo_no_treatment_utility',
    'lvo_no_treatment_mrs_0-2',
    'lvo_no_treatment_utility',
    ]
fixed_dict = dict([(k, results_dict[k]) for k in fixed_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys())
                     if k not in fixed_keys])

# Separate times and outcomes:
time_keys = [
    'drip_ship_ivt_time',
    'drip_ship_mt_time',
    'mothership_ivt_time',
    'mothership_mt_time',
    'msu_ivt_time',
    'msu_mt_time'
]
treatment_time_dict = dict([(k, results_dict[k]) for k in time_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys())
                     if k not in time_keys])

# Gather cumulative times and nicer-formatted cumulative time labels:
(times_dicts, times_cum_dicts, times_cum_label_dicts
 ) = build_data_for_timeline(fixed_dict | treatment_time_dict | input_dict)

# Column header names: occlusion, pathway, treatment, outcome.
df_results = utils.convert_results_dict_to_multiindex_df(results_dict)

# Make a container for parameters shared across all pathway types.
# The results will be drawn first but this container appears higher.
container_shared_data = st.container()

draw_timeline(times_cum_dicts, times_cum_label_dicts)

st.markdown('''
### Outcomes ###

* **mrs_0-2**: Proportion patients modified Rankin Scale 0-2 (higher is better)
* **mrs_shift**: Average shift in modified Rankin Scale (negative is better)
* **utility**: Average utility (higher is better)
* **utility_shift**: Average improvement in (higher is better)
''')

# User inputs for how to display the data:
group_by = st.radio(
    'Group results by:',
    ['Treatment type', 'Outcome type']
    )

if group_by == 'Treatment type':
    for stroke_type in ['ivt', 'mt', 'ivt_mt']:
        df_here = utils.take_subset_by_column_level_values(
            df_results.copy(), treatment=[stroke_type])
        df_here = utils.convert_row_to_table(
            df_here, ['occlusion', 'outcome'])
        st.markdown(f'### {stroke_type}')
        style_dict = results.make_column_style_dict(
            df_here.columns, format='%.3f')
        st.dataframe(
            df_here,
            column_config=style_dict
            )
else:
    for outcome in ['mrs_shift', 'mrs_0-2', 'utility', 'utility_shift']:
        df_here = utils.take_subset_by_column_level_values(
            df_results, outcome=[outcome])
        df_here = utils.convert_row_to_table(
            df_here, ['occlusion', 'treatment'])
        st.markdown(f'### {outcome}')
        style_dict = results.make_column_style_dict(
            df_here.columns, format='%.3f')
        st.dataframe(
            df_here,
            column_config=style_dict
            )


with container_shared_data:
    st.markdown('## Fixed values')
    st.markdown('Baseline outcomes')

    cols = [
        'nlvo_no_treatment_mrs_0-2',
        'nlvo_no_treatment_utility',
        'lvo_no_treatment_mrs_0-2',
        'lvo_no_treatment_utility',
        ]
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
    cols = ['msu_occupied_treatment', 'msu_occupied_no_treatment']
    # Extra pd.DataFrame() here otherwise streamlit sees it's a Series
    # and overrides the style dict.
    dict_msu = dict(zip(cols, [results_dict[k] for k in cols]))
    df_msu = pd.DataFrame(pd.Series(dict_msu))
    style_dict = results.make_column_style_dict(
        df_msu.index, format='%d')
    st.dataframe(
        df_msu.T,
        column_config=style_dict,
        # hide_index=True
        )


# ----- The end! -----
