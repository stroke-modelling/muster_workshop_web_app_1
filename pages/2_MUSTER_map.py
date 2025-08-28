"""
MUSTER results display app.

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
import utilities.calculations as calc
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.plot_mrs_dists as mrs
from utilities.utils import load_reference_mrs_dists
# import classes.model_module as model
# Containers:
import utilities.inputs as inputs
import utilities.colour_setup as colour_setup
import utilities.plot_timeline as timeline


# @st.cache_data
def calculate_treatment_times_each_lsoa(df_unit_services, treatment_time_dict):

    # ---- Travel times -----
    # One row per LSOA:
    df_travel_times = calc.calculate_geography(df_unit_services).copy()
    df_travel_times = df_travel_times.set_index('LSOA')

    # Copy stroke unit names over. Currently has only postcodes.
    cols_postcode = ['nearest_ivt_unit', 'nearest_mt_unit',
                     'transfer_unit', 'nearest_msu_unit']
    cols_postcode = [c for c in cols_postcode if c in df_travel_times.columns]
    for col in cols_postcode:
        df_travel_times = pd.merge(
            df_travel_times, df_unit_services['ssnap_name'],
            left_on=col, right_index=True, how='left'
            )
        df_travel_times = df_travel_times.rename(columns={
            'ssnap_name': f'{col}_name'})
        # Reorder columns so name appears next to postcode.
        i = df_travel_times.columns.tolist().index(col)
        cols_reorder = [
            *df_travel_times.columns[:i],
            f'{col}_name',
            *df_travel_times.columns[i:-1]
            ]
        df_travel_times = df_travel_times[cols_reorder]

    # ---- Treatment times -----
    # Find the times to IVT and MT in each scenario and for each
    # LSOA. The times are rounded to the nearest minute.
    # One row per LSOA:
    df_travel_times = calc.calculate_treatment_times_each_lsoa(
        df_travel_times, treatment_time_dict)
    return df_travel_times


def main_calculations(df_times):
    # ----- Unique treatment times -----
    # Find the complete set of times to IVT and times to MT.
    # Don't need the combinations of IVT and MT times yet.
    # Find set of treatment times:
    times_to_ivt = sorted(list(set(
        df_times[['usual_care_ivt', 'msu_ivt']].values.flatten())))
    times_to_mt = sorted(list(set(
        df_times[['usual_care_mt', 'msu_mt_with_ivt', 'msu_mt_no_ivt']]
        .values.flatten())))

    # ----- Outcomes for unique treatment times -----
    # Run the outcome model for only the unique treatment times
    # instead of one row per LSOA.
    # Run results for IVT and for MT separately.
    outcomes_by_stroke_type_ivt_only = (
        calc.run_outcome_model_for_unique_times_ivt(times_to_ivt))
    outcomes_by_stroke_type_mt_only = (
        calc.run_outcome_model_for_unique_times_mt(times_to_mt))

    # Convert these results dictionaries into dataframes:
    df_outcomes_ivt, df_outcomes_mt = (
        calc.convert_outcome_dicts_to_df_outcomes(
            times_to_ivt,
            times_to_mt,
            outcomes_by_stroke_type_ivt_only,
            outcomes_by_stroke_type_mt_only
            )
        )
    df_mrs_ivt, df_mrs_mt = (
        calc.convert_outcome_dicts_to_df_mrs(
            times_to_ivt,
            times_to_mt,
            outcomes_by_stroke_type_ivt_only,
            outcomes_by_stroke_type_mt_only
            )
        )
    return (
        df_outcomes_ivt,
        df_outcomes_mt,
        df_mrs_ivt,
        df_mrs_mt
    )


# @st.cache_data
def calculate_outcomes_for_combo_groups(
        df_lsoa,
        input_dict
        ):
    df_lsoa = df_lsoa.rename(columns={'LSOA': 'lsoa'})
    df_lsoa = df_lsoa.set_index('lsoa')

    # Calculate diff - msu minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['msu', 'usual_care']
        )

    return df_lsoa


def calculate_pdeath_for_combo_groups(
        df_pdeath, scenarios, input_dict,
        treatment_types=['ivt', 'mt', 'ivt_mt']
        ):
    """
    input_dict must contain keys 'prop_nlvo' and 'prop_lvo'.

    f'{s}_probdeath_nlvo_ivt',
    f'{s}_probdeath_lvo_ivt',
    f'{s}_probdeath_lvo_mt',
    f'{s}_probdeath_lvo_ivt_mt',
    """
    # Combine nLVO and LVO groups.
    # Set up data for no treatment:
    dist_dict = load_reference_mrs_dists()
    df_pdeath['probdeath_nlvo_no_treatment'] = (
        dist_dict['nlvo_no_treatment_noncum'][-1])
    # Gather the column names here:
    cols_nlvo = []
    cols_lvo = []
    cols_combo = []
    for s in scenarios:
        for t in treatment_types:
            # Set up existing column names.
            col_nlvo = f'{s}_probdeath_nlvo_{t}'
            col_lvo = col_nlvo.replace('nlvo', 'lvo')
            # New column name:
            col_combo = col_nlvo.replace('nlvo', 'combo')
            # Change nLVO column for special cases where the treatment
            # option is invalid:
            if t == 'mt':
                # Use no-treatment data.
                col_nlvo = 'probdeath_nlvo_no_treatment'
            elif t == 'ivt_mt':
                # Use IVT-only data.
                col_nlvo = col_nlvo.replace('ivt_mt', 'ivt')
            else:
                pass
            cols_nlvo.append(col_nlvo)
            cols_lvo.append(col_lvo)
            cols_combo.append(col_combo)
    # Combine the data:
    props_list = [input_dict['prop_nlvo'], input_dict['prop_lvo']]
    df_pdeath = calc.combine_results(
        df_pdeath, cols_nlvo, cols_lvo, cols_combo, props_list)
    return df_pdeath


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )

try:
    page_last_run = st.session_state['page_last_run']
    if page_last_run != 'MUSTER':
        # Clear the OPTIMIST results.
        keys_to_del = list(st.session_state.keys())
        for key in keys_to_del:
            del st.session_state[key]
except KeyError:
    # No page has been run yet.
    pass
st.session_state['page_last_run'] = 'MUSTER'


# #####################################
# ########## CONTAINER SETUP ##########
# #####################################
# Both tabs:
# +-----------------------------------------------+
# |                container_intro                |
# +-------------------------+---------------------+
#
# Inputs tab:
# +-----------------------------------------------+
# |                container_inputs               |
# |  +-----------------------------------------+  |
# |  |         container_inputs_summary        |  |
# |  +-----------------------------------------+  |
# |  |           container_inputs_top          |  |
# |  +-----------------------------------------+  |
# |  +---------+----------+---------+----------+  |
# |  |    c1   |    c2    |    c3   |    c4    |  |
# |  +---------+----------+---------+----------+  |
# +-----------------------------------------------+
# |             container_unit_services           |
# |  +-----------------------------------------+  |
# |  |         cu1        |         cu2        |  |
# |  +-----------------------------------------+  |
# |  |       container_services_buttons        |  |
# |  +-----------------------------------------+  |
# |  |      container_services_dataeditor      |  |
# |  +-----------------------------------------+  |
# +-----------------------------------------------+
#
# Results tab:
# +-----------------------------------------------+
# |                 container_rerun               |
# +-----------------------------------------------+
# |            container_results_tables           |
# +-----------------------------------------------+
# |              container_map_inputs             |
# |  +------------+--------------+-------------+  |
# |  |    cm0     |     cm1      |     cm2     |  |
# |  +------------+--------------+-------------+  |
# +-----------------------------------------------+
# |                 container_map                 |
# +-----------------------------------------------+
# |          container_input_region_type          |
# +-----------------------------------------------+
# |              container_actual_vlim            |
# +-----------------------------------------------+
# +-----------------------------------------------+
# |             container_mrs_dists_etc           |
# |  +-----------------------------------------+  |
# |  |           container_mrs_dists           |  |
# |  +--------------------+--------------------+  |
# |  |   container_bars   | container_mrs_input|  |
# |  +--------------------+--------------------+  |
# +-----------------------------------------------+
#
# Sidebar:
# v Accessibility & advanced options
#   +--------------------------+
#   | container_select_outcome |
#   +--------------------------+
#   |  container_select_cmap   |
#   +--------------------------+
#   |  container_select_vlim   |
#   +--------------------------+

container_intro = st.container()
tab_inputs, tab_results = st.tabs(['Inputs', 'Results'])
with tab_inputs:
    container_inputs = st.container()
    # container_timeline_plot = st.container()
    container_unit_services = st.container()

with container_inputs:
    st.markdown('## Pathway timings')
    container_inputs_summary = st.container()
    container_inputs_top = st.container()
    (
        container_inputs_standard,     # c1
        container_timeline_standard,   # c2
        container_inputs_msu,          # c3
        container_timeline_msu         # c4
    ) = st.columns(4)
with container_unit_services:
    (
        container_unit_services_top,  # cu1
        container_services_map        # cu2
    ) = st.columns([1, 2])
    with container_unit_services_top:
        st.header('Stroke unit services')
    container_services_buttons = st.container()
    container_services_dataeditor = st.container()
with container_services_buttons:
    st.info(''.join([
        'To update the services, use the following buttons ',
        'and/or click the tick-boxes in the table.'
        ]), icon='➡️')

with tab_results:
    container_rerun = st.container()
    st.markdown('## Full results')
    with st.expander('Full data tables'):
        container_results_tables = st.container()
    container_map_inputs = st.container()  # border=True)
    with container_map_inputs:
        st.markdown('## Subgroup results')
        (
            container_input_words,        # cm0
            container_input_treatment,    # cm1
            container_input_stroke_type,  # cm2
        ) = st.columns(3)
        with container_input_words:
            st.markdown('Pick the subgroup using these buttons:')

    # Convert the map container to empty so that the placeholder map
    # is replaced once the real map is ready.
    st.markdown('### Maps of average outcomes')
    container_map = st.empty()
    container_input_region_type = st.container()
    container_actual_vlim = st.container()
    container_mrs_dists_etc = st.container()
    # Convert mRS dists to empty so that re-running a fragment replaces
    # the bars rather than displays the new plot in addition.
    with container_mrs_dists_etc:
        st.markdown('### Distributions of mRS scores')
        container_mrs_dists = st.empty()

with st.sidebar:
    with st.expander('Accessibility & advanced options'):
        container_select_outcome = st.container()
        container_select_cmap = st.container()
        container_select_vlim = st.container()

with container_intro:
    st.markdown('# Benefit in outcomes from Mobile Stroke Units')


# #################################
# ########## USER INPUTS ##########
# #################################

# These affect the data in all tables and all plots.

# ----- Pathway timings and stroke units -----
input_dict = {}
with container_inputs_top:
    st.info('To update the timings, use the number box options below.',
            icon='➡️')
    st.markdown(''.join([
        'These timings affect the times to treatment for all patients ',
        'excluding the travel times to their chosen stroke units or MSU.'
    ]))
    input_dict['process_time_call_ambulance'] = st.number_input(
        'Time to call ambulance',
        value=60,
        help=f"Reference value: {60}",
        # key=key
        )
with container_inputs_standard:
    st.markdown('### Standard pathway')
    input_dict = inputs.select_parameters_map(input_dict)
with container_inputs_msu:
    st.markdown('### Mobile Stroke Unit')
    input_dict = inputs.select_parameters_msu(input_dict)

df_unit_services, df_unit_services_full = (
    inputs.select_stroke_unit_services_broad(
        container_services_buttons,
        container_services_dataeditor,
    ))


# Calculate times to treatment without travel.
# These are used in the main_calculations and also displayed on
# the inputs page as a sanity check.
treatment_times_without_travel = (
    calc.calculate_times_to_treatment_without_travel_usual_care(input_dict))
treatment_times_without_travel_msu = (
    calc.calculate_times_to_treatment_without_travel_msu(input_dict))
# Combine these:
treatment_times_without_travel = (
    treatment_times_without_travel | treatment_times_without_travel_msu)

# These do not change the underlying data,
# but do change what is shown in the plots.

# ----- Stroke type, treatment, outcome -----
with container_select_outcome:
    st.markdown('### Alternative outcome measures')
    outcome_type, outcome_type_str = inputs.select_outcome_type()
with container_input_treatment:
    treatment_type, treatment_type_str = inputs.select_treatment_type()
with container_input_stroke_type:
    stroke_type, stroke_type_str = (
        inputs.select_stroke_type(use_combo_stroke_types=False))


# ----- Regions to draw -----
# Name of the column in the geojson that labels the shapes:
with container_input_region_type:
    outline_name = st.radio(
        'Region type to draw on maps',
        ['None', 'ISDN', 'ICB', 'Nearest service', 'Ambulance service'],
        horizontal=True
        )

# ----- Colourmap selection -----
cmap_names = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
# Add the reverse option after each entry. Remove any double reverse
# reverse _r_r. Result is flat list.
cmap_names = sum([[c, (c + '_r').replace('_r_r', '')]
                  for c in cmap_names], [])
with container_select_cmap:
    st.markdown('### Colour schemes')
    cmap_name = inputs.select_colour_maps(cmap_names)
cmap_diff_name = cmap_name
cmap_pop_name = cmap_name
# If we're showing mRS scores then flip the colour maps:
if outcome_type == 'mrs_shift':
    cmap_name = (cmap_name + '_r').replace('_r_r', '')
    cmap_diff_name = (cmap_diff_name + '_r').replace('_r_r', '')


# ----- Demographic data -----
# For population map. Load in LSOA-level demographic data:
try:
    df_demog = st.session_state['df_demog']
except KeyError:
    df_demog = inputs.load_lsoa_demog()
# Set the column of this data that we want...
column_pop = 'population_density'
# ... and how the name should be displayed in the app:
column_pop_pretty = 'Population density (people per square kilometre)'


# ----- Colour limits -----
# Load initial colour limits info (vmin, vmax):
dict_colours, dict_colours_diff = (
    colour_setup.load_colour_limits(outcome_type))
dict_colours_pop = {'vmin': 0.0, 'vmax': 100.0}
# User inputs for vmin and vmax with loaded values as defaults:
with container_select_vlim:
    st.markdown('### Colour limits')
    dict_colours['vmin'], dict_colours['vmax'] = (
        inputs.select_map_colour_limits(dict_colours, outcome_type_str))
    dict_colours_diff['vmin'], dict_colours_diff['vmax'] = (
        inputs.select_map_colour_limits(
            dict_colours_diff, f'{outcome_type_str} benefit of MSU'))
    dict_colours_pop['vmin'], dict_colours_pop['vmax'] = (
        inputs.select_map_colour_limits(dict_colours_pop, column_pop_pretty))


# ######################################
# ########## PLOT USER INPUTS ##########
# ######################################

# ----- Timeline -----
# Load emoji and labels:
timeline_display_dict = timeline.get_timeline_display_dict()
# Create timelines:
time_dicts = timeline.build_time_dicts_muster(input_dict)
time_offsets, tmax = timeline.build_time_dicts_for_plot_msu(
    time_dicts, gap_between_chunks=45)

# Make subsets of the dictionaries to be displayed:
time_keys_standard = ['prehosp_usual_care', 'ivt_only_unit',
                      'mt_transfer_unit', 'ivt_mt_unit']
time_keys_msu = ['msu_dispatch', 'prehosp_msu_ivt', 'prehosp_msu_no_ivt',
                 'mt_transfer_from_msu']
time_dicts_standard, time_offsets_standard = (
    timeline.subset_time_dicts(time_dicts, time_offsets, time_keys_standard))
time_dicts_msu, time_offsets_msu = (
    timeline.subset_time_dicts(time_dicts, time_offsets, time_keys_msu))

# Draw the timelines:
with container_timeline_standard:
    timeline.plot_timeline(
        time_dicts_standard,
        timeline_display_dict,
        y_vals=[0.5, 1, 1, 0],  # timeline fragment centres
        time_offsets=time_offsets_standard,
        tmax=tmax,
        tmin=-10.0
        )
with container_timeline_msu:
    timeline.plot_timeline(
        time_dicts_msu,
        timeline_display_dict,
        y_vals=[0.5, 0, 1, 0.5],  # timeline fragment centres
        time_offsets=time_offsets_msu,
        tmax=tmax,
        tmin=-10.0
    )


# ----- Treatment times summary -----
df_treatment_times = (
    timeline.make_treatment_time_df_msu(treatment_times_without_travel))
# Display the times:
times_explanation_usual_str = ('''
For the standard pathway:
+ The "fastest" time to MT is when the first stroke unit provides MT.
+ The "slowest" time to MT is when a transfer to the MT unit is needed.
''')
times_explanation_msu_str = ('''
For the MSU:
+ The "fastest" time to MT is when thrombolysis was not given in the MSU.
+ The "slowest" time to MT is when thrombolysis has been given in the MSU.
''')
with container_inputs_summary:
    st.markdown('Summary of treatment times:')
    st.table(df_treatment_times)
    cols = st.columns(2)
    with cols[0]:
        st.markdown(times_explanation_usual_str)
    with cols[1]:
        st.markdown(times_explanation_msu_str)


# ----- Unit maps -----
# Stroke unit scatter markers:
traces_units = plot_maps.create_stroke_team_markers(
    df_unit_services_full)
# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'msu': [[1, 2]],        # second map only
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1], [1, 2]]  # both maps
}
# Regions to draw
# Name of the column in the geojson that labels the shapes:
with container_unit_services_top:
    st.markdown(''.join([
        'These maps show which services are provided ',
        'by each stroke unit in England.'
        ]))
    outline_name_for_unit_map = st.radio(
        'Region type to draw on maps',
        ['None', 'ISDN', 'ICB', 'Nearest service', 'Ambulance service'],
        key='outlines_for_unit_map'
        )

# Region outlines:
if outline_name_for_unit_map == 'None':
    outline_names_col_for_unit_map = None
    gdf_catchment_lhs_for_unit_map = None
    gdf_catchment_rhs_for_unit_map = None
else:
    if outline_name_for_unit_map == 'Nearest service':
        # Times to treatment:
        geo = calc.calculate_geography(df_unit_services)
        # Put columns in the format expected by the region outline
        # function:
        geo = geo.rename(columns={
            'LSOA': 'lsoa',
            'nearest_ivt_unit': 'nearest_ivt_unit_name',
            'nearest_mt_unit': 'nearest_mt_unit_name',
            'nearest_msu_unit': 'nearest_msu_unit_name',
            })
        geo = geo.set_index(['lsoa', 'nearest_ivt_unit_name'])
    else:
        # Don't need catchment areas.
        geo = None
    (
        outline_names_col_for_unit_map,
        gdf_catchment_lhs_for_unit_map,
        gdf_catchment_rhs_for_unit_map,
        gdf_catchment_pop_for_unit_map
    ) = calc.load_or_calculate_region_outlines(
        outline_name_for_unit_map, geo, col_rhs='nearest_msu_unit_name')

# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [
    gdf_catchment_lhs_for_unit_map,
    gdf_catchment_rhs_for_unit_map
    ]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        gdf['x'], gdf['y'] = maps.convert_shapely_polys_into_xy(gdf)

with container_services_map:
    plot_maps.plotly_unit_maps(
        traces_units,
        unit_subplot_dict,
        gdf_catchment_lhs_for_unit_map,
        gdf_catchment_rhs_for_unit_map,
        outline_names_col_for_unit_map,
        outline_name_for_unit_map,
        subplot_titles=['Usual care', 'Mobile Stroke Unit']
        )


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_map:
    plot_maps.plotly_blank_maps(['', '', ''], n_blank=3)

try:
    df_lsoa_regions = st.session_state['df_lsoa_regions']
except KeyError:
    df_lsoa_regions = inputs.load_lsoa_region_lookups()
    st.session_state['df_lsoa_regions'] = df_lsoa_regions


with container_rerun:
    if st.button('Calculate results', type='primary'):

        st.session_state['input_dict'] = input_dict
        st.session_state['df_unit_services_on_last_run'] = df_unit_services
        st.session_state['df_unit_services_full_on_last_run'] = (
            df_unit_services_full)

        df_times = calculate_treatment_times_each_lsoa(
            df_unit_services, treatment_times_without_travel)

        (
            df_outcomes_ivt,
            df_outcomes_mt,
            st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'],
        ) = main_calculations(df_times)



        scenarios = ['usual_care', 'msu']
        r = {'msu_mt_no_ivt': 'msu_mt', 'msu_mt_with_ivt': 'msu_ivt_mt'}
        st.session_state['df_lsoa'] = (
            calc.build_full_lsoa_outcomes_from_unique_time_results(
                df_times.rename(columns=r), df_outcomes_ivt, df_outcomes_mt, scenarios)
        )

        # Calculate outcomes:
        st.session_state['df_lsoa'] = calculate_outcomes_for_combo_groups(
            st.session_state['df_lsoa'],
            input_dict
            )

        # Calculate probabilities of death.
        # Pick out the masks where IVT is better than MT:
        cols_ivt_better = [f'{s}_lvo_ivt_better_than_mt' for s in scenarios]
        # Place these masks in the travel time data:
        df_pdeath = pd.merge(
            df_times.copy().rename(columns=r).reset_index().rename(columns={'LSOA': 'lsoa'}).set_index('lsoa'),
            st.session_state['df_lsoa'][cols_ivt_better],
            left_index=True, right_index=True, how='left'
            )
        # Now gather P(death):
        df_pdeath = calc.gather_pdeath_from_unique_time_results(
            df_pdeath.reset_index(), st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'], scenarios,
        )
        df_pdeath = df_pdeath.set_index('lsoa')
        # note: cannot currently run the below as we haven't defined
        # the proportions of nLVO and LVO in the input_dict.
        # # Calculate P(death) for combined groups, mix of nLVO and LVO.
        # df_pdeath = calculate_pdeath_for_combo_groups(
        #     df_pdeath, scenarios, input_dict)

        # Combine outcome and P(death) data:
        cols_pdeath = [c for c in df_pdeath.columns if 'probdeath' in c]
        st.session_state['df_lsoa'] = pd.merge(
            st.session_state['df_lsoa'], df_pdeath[cols_pdeath],
            left_index=True, right_index=True, how='left'
        )

        # Gather outcomes and P(death) into regions:
        (
            st.session_state['df_icb'],
            st.session_state['df_isdn'],
            st.session_state['df_nearest_ivt'],
            st.session_state['df_ambo']
        ) = calc.group_results_by_region(
            st.session_state['df_lsoa'].reset_index().rename(columns={'LSOA': 'lsoa'}),
            df_unit_services,
            df_lsoa_regions
            )
        new_results_run = True
    else:
        new_results_run = False
        # Check whether the inputs have changed from last run:
        try:
            # Define s to shorten the following lines:
            s = st.session_state['df_unit_services_on_last_run']
            # Conditions that mean inputs have changed:
            c1 = (st.session_state['input_dict'] != input_dict)
            c2 = (s['Use_IVT'] != df_unit_services['Use_IVT']).any()
            c3 = (s['Use_MT'] != df_unit_services['Use_MT']).any()
            c4 = (s['Use_MSU'] != df_unit_services['Use_MSU']).any()
            # Check for any of these changing:
            inputs_changed = (c1 | c2 | c3 | c4)
        except KeyError:
            # First run of the app.
            inputs_changed = False
        # If the inputs have changed, print a warning:
        if inputs_changed:
            with container_rerun:
                st.warning(''.join([
                    'Inputs have changed! The results currently being shown ',
                    'are for the previous set of inputs. ',
                    'Use the "calculate results" button ',
                    'to update the results.'
                    ]), icon='⚠️')


# Check if any results have been calculated before.
if 'df_lsoa' in st.session_state.keys():
    pass
else:
    # This hasn't been created yet and so the results cannot be drawn.
    st.stop()


# #########################################
# ########## RESULTS - FULL DATA ##########
# #########################################
with container_results_tables:
    table_choice = st.selectbox(
        'Display the following results table:',
        options = [
            'Results by IVT unit catchment',
            'Results by ISDN',
            'Results by ICB',
            'Results by ambulance service',
            'Full results by LSOA'
            ]
        )

    if 'IVT unit' in table_choice:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each IVT unit catchment area.'
            ]))
        st.dataframe(
            st.session_state['df_nearest_ivt'],
            # Set some columns to bool for nicer display:
            column_config={
                'transfer_required': st.column_config.CheckboxColumn()
                }
            )
    elif 'ISDN' in table_choice:
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(st.session_state['df_isdn'])

    elif 'ICB' in table_choice:
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(st.session_state['df_icb'])

    elif 'ambulance' in table_choice:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each ambulance service.'
            ]))
        st.dataframe(st.session_state['df_ambo'])

    else:
        st.dataframe(
            st.session_state['df_lsoa'],
            # Set some columns to bool for nicer display:
            column_config={
                'transfer_required': st.column_config.CheckboxColumn()
                }
            )


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

# st.session_state['df_mrs'] column names are in the format:
# `usual_care_lvo_ivt_mt_mrs_dists_X`, for X from 0 to 6, i.e.
# '{scenario}_{occlusion}_{treatment}_{dist}_{X}' with these options:
#
# +---------------------------+------------+------------+------------------+
# | Scenarios                 | Occlusions | Treatments | Dist types       |
# +---------------------------+------------+------------+------------------+
# | usual_care                | nlvo       | ivt        | mrs_dists        |
# | msu                       | lvo        | mt         | mrs_dists_noncum |
# | diff_msu_minus_usual_care |            | ivt_mt     |                  |
# +---------------------------+------------+------------+------------------+
#
# There is not a separate column for "no treatment" to save space.

# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)
bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]

# Which mRS distributions will be shown on the bars:
scenario_mrs = ['usual_care', 'msu']

# Limit the mRS data to only LSOA that benefit from an MSU,
# i.e. remove anything where the added utility of MSU is not better
# than the added utility of usual care.
d_str = 'diff_msu_minus_usual_care'

if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # This data doesn't exist so show no LSOAs.
    # lsoa_to_keep = []
    col_to_mask_mrs = ''
elif ((stroke_type == 'nlvo') & ('mt' in treatment_type)):
    # Use IVT-only data:
    col_to_mask_mrs = f'{d_str}_{stroke_type}_ivt_{outcome_type}'
    # lsoa_to_keep = st.session_state['df_lsoa'].index[
    #     (st.session_state['df_lsoa'][c1] > 0.0)]
else:
    # Look up the data normally.
    col_to_mask_mrs = f'{d_str}_{stroke_type}_{treatment_type}_{outcome_type}'
    # lsoa_to_keep = st.session_state['df_lsoa'].index[
    #     (st.session_state['df_lsoa'][c1] > 0.0)]

# if inputs_changed:
#     pass
# else:
# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.

# Pick out useful bits from the full outcome results:
cols_to_copy = [
    'Admissions',
    'usual_care_ivt',
    'usual_care_mt',
    'usual_care_lvo_ivt_better_than_mt',
    'nearest_ivt_unit_name'
    ]
if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
    cols_to_copy.append(col_to_mask_mrs)
df_mrs_usual_care = st.session_state['df_lsoa'][cols_to_copy].copy()
df_mrs_usual_care = df_mrs_usual_care.rename(columns={
    'usual_care_ivt': 'time_to_ivt',
    'usual_care_mt': 'time_to_mt',
    'usual_care_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
})

cols_to_copy_msu = [
    'Admissions',
    'msu_ivt',
    'msu_ivt_mt',  # 'msu_mt_with_ivt',
    'msu_mt',  # 'msu_mt_no_ivt',
    'msu_lvo_ivt_better_than_mt',
    'nearest_ivt_unit_name'
    ]
if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
    cols_to_copy_msu.append(col_to_mask_mrs)
df_mrs_msu = st.session_state['df_lsoa'][cols_to_copy_msu].copy()
if 'ivt' in treatment_type:
    df_mrs_msu['time_to_mt'] = df_mrs_msu['msu_ivt_mt']  # 'msu_mt_with_ivt']
else:
    df_mrs_msu['time_to_mt'] = df_mrs_msu['msu_mt']  # 'msu_mt_no_ivt']
# df_mrs_msu = df_mrs_msu.drop(['msu_mt_with_ivt', 'msu_mt_no_ivt'], axis='columns')
df_mrs_msu = df_mrs_msu.drop(['msu_mt', 'msu_ivt_mt'], axis='columns')
df_mrs_msu = df_mrs_msu.rename(columns={
    'msu_ivt': 'time_to_ivt',
    'msu_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
})

# Merge in region info:
df_mrs_usual_care = pd.merge(
    df_mrs_usual_care.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')
df_mrs_msu = pd.merge(
    df_mrs_msu.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')

dict_of_dfs = {
    'usual_care': df_mrs_usual_care,
    'msu': df_mrs_msu,
}


@st.fragment
def display_mrs_dists():
    (
        container_bars,
        container_mrs_input,
    ) = st.columns(2)

    with container_mrs_input:
        # User input:
        bar_option = st.selectbox('Region for mRS distributions', bar_options)
        st.markdown(''.join([
            'mRS distributions are shown for only LSOA who would benefit ',
            'from an MSU. These are LSOA where the "added utility" ',
            'for the "MSU" scenario ',
            'is better than for the "usual care" scenario.'
            ]))
    # Set up where the data should come from -
    # which region type was selected, and which region name.
    region_selected, col_region = mrs.pick_out_region_name(bar_option)

    # if inputs_changed:
    #     if 'fig_mrs' in st.session_state.keys():
    #         pass
    #     else:
    #         st.stop()
    # else:

    # Prettier formatting for the plot title:
    col_pretty = ''.join([
        f'{stroke_type_str}, ',
        f'{treatment_type_str}'
        ])

    # mrs_lists_dict = mrs.calculate_average_mrs(
    #     stroke_type,
    #     treatment_type,
    #     col_region,
    #     region_selected,
    #     col_to_mask_mrs,
    #     # Setup for mRS dists:
    #     dict_of_dfs,
    #     # The actual mRS dists:
    #     st.session_state['df_mrs_ivt'],
    #     st.session_state['df_mrs_mt'],
    #     input_dict
    #     )

    # Find reference mRS distributions (no treatment).
    # If occ_type is nLVO or LVO, this returns the normal dists.
    # Otherwise it returns a scaled sum of the nLVO and LVO dists.
    dist_ref_cum, dist_ref_noncum = mrs.load_no_treatment_mrs_dists(
        stroke_type)
    # Store no-treatment data:
    dict_no_treatment = {
        'noncum': dist_ref_noncum,
        'cum': dist_ref_cum,
        'std': None
    }

    # Store results in here:
    keys = ['no_treatment'] + scenario_mrs

    # Decide whether to use no-treatment dists or to
    # fish dists out of the big mRS lists.
    use_ref_data = (True if
                    ((stroke_type == 'nlvo') & (treatment_type == 'mt'))
                    else False)
    # Use nLVO IVT data instead of nLVO IVT & MT.
    # (Getting UnboundLocalError if attempting this while changing
    # value of treatment_type)
    # if ((stroke_type == 'nlvo') & ('mt' in treatment_type)):
    #     treat_type = 'ivt'
    # else:
    treat_type = treatment_type
    # Calculate mRS for both nLVO and LVO so that we can find the mRS
    # for "redirection considered" and for combo nLVO+LVO group.
    stroke_types = ['nlvo', 'lvo']

    mrs_dfs_dict = {}
    if use_ref_data:
        mrs_lists_dict = {}
        mrs_lists_dict['no_treatment'] = dict_no_treatment
        for key in keys:
            mrs_lists_dict[key] = dict_no_treatment
    else:
        for key in scenario_mrs:
            mrs_dfs_dict[key] = {}
        lsoa_names = mrs.find_lsoa_names_to_keep(
            dict_of_dfs['usual_care'],
            col_to_mask_mrs,
            col_region,
            region_selected
            )
        mrs_dfs_dict, dist_cols = mrs.find_total_mrs_for_unique_times(
            dict_of_dfs,
            lsoa_names,
            treat_type,
            stroke_types,
            st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'],
            )

        if stroke_type == 'nlvo':
            dist_cols = [c for c in dist_cols if 'nlvo' in c]
        elif stroke_type == 'lvo':
            dist_cols = [c for c in dist_cols if
                         (('lvo' in c) & ('nlvo' not in c))]
        else:
            pass

        # Average these mRS dists:
        mrs_lists_dict = {}
        mrs_lists_dict['no_treatment'] = dict_no_treatment
        for key, df in mrs_dfs_dict.items():
            dist_noncum, dist_cum, dist_std = (
                mrs.calculate_average_mrs_dists(df, dist_cols))
            # Store in results dict:
            mrs_lists_dict[key] = {}
            mrs_lists_dict[key]['noncum'] = dist_noncum
            mrs_lists_dict[key]['cum'] = dist_cum
            mrs_lists_dict[key]['std'] = dist_std

    mrs_format_dicts = (
        mrs.setup_for_mrs_dist_bars(mrs_lists_dict))

    st.session_state['fig_mrs'] = mrs.plot_mrs_bars(
        mrs_format_dicts, title_text=f'{region_selected}<br>{col_pretty}')


    with container_bars:
        # Options for the mode bar.
        # (which doesn't appear on touch devices.)
        plotly_config = {
            # Mode bar always visible:
            # 'displayModeBar': True,
            # Plotly logo in the mode bar:
            'displaylogo': False,
            # Remove the following from the mode bar:
            'modeBarButtonsToRemove': [
                # 'zoom',
                # 'pan',
                'select',
                # 'zoomIn',
                # 'zoomOut',
                'autoScale',
                'lasso2d'
                ],
            # Options when the image is saved:
            'toImageButtonOptions': {'height': None, 'width': None},
            }
        st.plotly_chart(
            st.session_state['fig_mrs'],
            use_container_width=True,
            config=plotly_config
            )


with container_mrs_dists:
    display_mrs_dists()


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

# Display names:
subplot_titles = [
    'Usual care',
    'Benefit of MSU over usual care',
    column_pop_pretty
]


# ----- Find data for colours -----

# st.session_state['df_lsoa'] column names are in the format:
# `usual_care_lvo_ivt_mt_utility_shift`, i.e.
# '{scenario}_{occlusion}_{treatment}_{outcome}' with these options:
#
# +---------------------------+------------+------------+---------------+
# | Scenarios                 | Occlusions | Treatments | Outcomes      |
# +---------------------------+------------+------------+---------------+
# | usual_care                | nlvo       | ivt        | utility_shift |
# | msu                       | lvo        | mt         | mrs_shift     |
# | diff_msu_minus_usual_care |            | ivt_mt     | mrs_0-2       |
# +---------------------------+------------+------------+---------------+
#
# There is not a separate column for "no treatment" to save space.

# Find the names of the columns that contain the data
# that will be shown in the colour maps.
if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # Use no-treatment data.
    # Set this to something that doesn't exist so it fails the try.
    column_colours = None
    column_colours_diff = None
else:
    # If this is nLVO with IVT and MT, look up the data for
    # nLVO with IVT only.
    using_nlvo_ivt_mt = ((stroke_type == 'nlvo') & ('mt' in treatment_type))
    t = 'ivt' if using_nlvo_ivt_mt else treatment_type

    column_colours = '_'.join([
        'usual_care', stroke_type, t, outcome_type])
    column_colours_diff = '_'.join([
        'diff_msu_minus_usual_care', stroke_type, t, outcome_type])


# ----- Set up map data -----
try:
    df_raster = st.session_state['df_raster']
    transform_dict = st.session_state['transform_dict']
except KeyError:
    # Load LSOA geometry:
    df_raster, transform_dict = maps.load_lsoa_raster_lookup()
    # Store:
    st.session_state['df_raster'] = df_raster
    st.session_state['df_raster_cols'] = df_raster.columns
    st.session_state['transform_dict'] = transform_dict

if new_results_run:
    # Remove results from last run:
    df_raster = df_raster[st.session_state['df_raster_cols']]
    # Merge in outcomes data:
    df_raster = pd.merge(df_raster, st.session_state['df_lsoa'],
                         left_on='LSOA11NM', right_on='lsoa', how='left')
    # Merge demographic data:
    df_raster = pd.merge(df_raster, df_demog[['LSOA', column_pop]],
                         left_on='LSOA11NM', right_on='LSOA', how='left')
st.session_state['df_raster'] = df_raster

# Make raster arrays out of the chosen data:
burned_lhs = maps.convert_df_to_2darray(df_raster, column_colours,
                                        transform_dict)
burned_rhs = maps.convert_df_to_2darray(df_raster, column_colours_diff,
                                        transform_dict)
burned_pop = maps.convert_df_to_2darray(df_raster, column_pop,
                                        transform_dict)


# ----- Set up colours -----
# Pick out the data for the colours:
try:
    vals_for_colours = df_raster[column_colours]
    vals_for_colours_diff = df_raster[column_colours_diff]
except KeyError:
    # Those columns don't exist in the data.
    # This should only happen for nLVO treated with MT only.
    vals_for_colours = [0] * len(df_raster)
    vals_for_colours_diff = [0] * len(df_raster)
    # Note: this works for now because expect always no change
    # for added utility and added mrs<=2 with no treatment.
# Pick out values:
vals_for_colours_pop = df_raster[column_pop]
vals_lists = [vals_for_colours, vals_for_colours_diff, vals_for_colours_pop]

# Colour limits.
# Record actual highest and lowest values in a DataFrame:
df_actual_vlim = pd.DataFrame(
    [[min(v) for v in vals_lists], [max(v) for v in vals_lists]],
    columns=subplot_titles, index=['Minimum', 'Maximum']
)
with container_actual_vlim:
    st.markdown('Ranges of the plotted data:')
    st.dataframe(df_actual_vlim)
    st.markdown(''.join([
        'The range of the colour scales in the maps can be changed ',
        'using the options in the sidebar.'
        ]))

# Load colour map colours:
dict_colours['cmap'] = colour_setup.make_colour_list(
    cmap_name,
    vmin=dict_colours['vmin'],
    vmax=dict_colours['vmax']
    )
dict_colours_diff['cmap'] = colour_setup.make_colour_list(
    cmap_diff_name,
    vmin=dict_colours_diff['vmin'],
    vmax=dict_colours_diff['vmax']
    )
dict_colours_pop['cmap'] = colour_setup.make_colour_list(
    cmap_pop_name,
    vmin=dict_colours_pop['vmin'],
    vmax=dict_colours_pop['vmax']
    )
# Colour bar titles:
dict_colours['title'] = f'{outcome_type_str}'
dict_colours_diff['title'] = (
    f'{outcome_type_str}: Benefit of MSU over usual care')
dict_colours_pop['title'] = column_pop_pretty


# ----- Region outlines -----
if outline_name == 'None':
    outline_names_col = None
    gdf_catchment_pop = None
    gdf_catchment_lhs = None
    gdf_catchment_rhs = None
else:
    (
        outline_names_col,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        gdf_catchment_pop,
    ) = calc.load_or_calculate_region_outlines(
            outline_name, st.session_state['df_lsoa'], use_msu=True)


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_catchment_pop, gdf_catchment_lhs, gdf_catchment_rhs]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        gdf['x'], gdf['y'] = maps.convert_shapely_polys_into_xy(gdf)


# ----- Stroke units -----
# Stroke unit scatter markers:
traces_units = plot_maps.create_stroke_team_markers(
    st.session_state['df_unit_services_full_on_last_run'])
# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'msu': [[1, 2]],        # second map only
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1], [1, 2]]  # both maps
}


# ----- Plot -----
st.session_state['fig_maps'] = plot_maps.plotly_many_heatmaps(
    burned_lhs,
    burned_rhs,
    burned_pop,
    gdf_catchment_lhs,
    gdf_catchment_rhs,
    gdf_catchment_pop,
    outline_names_col,
    outline_name,
    traces_units,
    unit_subplot_dict,
    subplot_titles=subplot_titles,
    dict_colours=dict_colours,
    dict_colours_diff=dict_colours_diff,
    dict_colours_pop=dict_colours_pop,
    transform_dict=transform_dict,
    cmaps=cmap_names,
    )


with container_map:
    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            # 'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    st.plotly_chart(
            st.session_state['fig_maps'],
            use_container_width=True,
            config=plotly_config
            )
