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

df_results = pd.DataFrame.from_dict(
    [results_drip_ship, results_mothership, results_msu],
    orient='columns',
)
df_results.index = ['Drip & ship', 'Mothership', 'MSU']

# Pick out the occlusion and treatment types and stick them
# in a MultiIndex header.
new_cols = results.make_multiindex_stroke_type(df_results.columns)
tuples = list(zip(*new_cols))
df_results.columns = pd.MultiIndex.from_tuples(tuples)

# Pick out the occlusion and treatment types and stick them
# into separate DataFrames.
df_results_nlvo_ivt = df_results['nlvo_ivt']
df_results_lvo_ivt = df_results['lvo_ivt']
df_results_lvo_mt = df_results['lvo_mt']
df_results_lvo_ivt_mt = df_results['lvo_ivt_mt']
df_results_other = df_results['']

st.markdown('### Shared results')

df_results_all = pd.DataFrame.from_dict(
    [results_all], orient='columns').T
st.table(df_results_all)
st.table(df_results_other.T)

st.markdown('### nLVO IVT')
st.table(df_results_nlvo_ivt.T)

st.markdown('### LVO IVT')
st.table(df_results_lvo_ivt.T)

st.markdown('### LVO MT')
st.table(df_results_lvo_mt.T)

st.markdown('### LVO IVT & MT')
st.table(df_results_lvo_ivt_mt.T)



# ----- The end! -----
