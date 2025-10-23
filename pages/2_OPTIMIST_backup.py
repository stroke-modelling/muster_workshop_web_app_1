"""
OPTIMIST results display app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.plot_mrs_dists as mrs
import utilities.colour_setup as colour_setup
import utilities.inputs as inputs
import utilities.plot_timeline as timeline
from utilities.utils import load_reference_mrs_dists


# @st.cache_data
def calculate_treatment_times_each_lsoa(df_unit_services, treatment_time_dict):

    # ---- Travel times -----
    # One row per LSOA:
    df_travel_times = calc.calculate_geography(df_unit_services).copy()
    df_travel_times = df_travel_times.set_index('LSOA')

    # Copy stroke unit names over. Currently has only postcodes.
    cols_postcode = ['nearest_ivt_unit', 'nearest_mt_unit',
                     'transfer_unit']
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
    df_travel_times = calc.calculate_treatment_times_each_lsoa_prehospdiag(
        df_travel_times, treatment_time_dict)
    return df_travel_times


def main_calculations(df_times):
    # ----- Unique treatment times -----
    # Find the complete set of times to IVT and times to MT.
    # Don't need the combinations of IVT and MT times yet.
    # Find set of treatment times:
    scenarios = ['usual_care', 'redirection_approved', 'redirection_rejected']
    cols_ivt = [f'{s}_ivt' for s in scenarios]
    cols_mt = [f'{s}_mt' for s in scenarios]

    times_to_ivt = sorted(list(set(df_times[cols_ivt].values.flatten())))
    times_to_mt = sorted(list(set(df_times[cols_mt].values.flatten())))

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
    # Make a new dummy zero column available:
    col_zero = 'zero'
    df_lsoa[col_zero] = 0.0

    # Extra calculations for redirection:
    # Combine redirection rejected and approved results in
    # proportions given by specificity and sensitivity.
    # This creates columns labelled "redirection_considered".
    # nLVO:
    cols_rr, cols_ra, cols_rc = calc.build_columns_combine_redir(
        scenario_list=['redirection_rejected', 'redirection_approved'],
        combo_name='redirection_considered',
        treatment_list=['ivt'],
        occlusion_list=['nlvo'],
        dummy_col=col_zero
        )
    cols_rr, cols_ra, cols_rc = calc.keep_existing_cols(
        df_lsoa, cols_rr, cols_ra, cols_rc)
    prop_nlvo_redirected = (1.0 - input_dict['specificity'])
    props_list = [1.0 - prop_nlvo_redirected, prop_nlvo_redirected]
    df_lsoa = calc.combine_results(
        df_lsoa, cols_rr, cols_ra, cols_rc, props_list)
    # LVO:
    cols_rr, cols_ra, cols_rc = calc.build_columns_combine_redir(
        scenario_list=['redirection_rejected', 'redirection_approved'],
        combo_name='redirection_considered',
        occlusion_list=['lvo'],
        dummy_col=col_zero
        )
    cols_rr, cols_ra, cols_rc = calc.keep_existing_cols(
        df_lsoa, cols_rr, cols_ra, cols_rc)
    prop_lvo_redirected = input_dict['sensitivity']
    props_list = [1.0 - prop_lvo_redirected, prop_lvo_redirected]
    df_lsoa = calc.combine_results(
        df_lsoa, cols_rr, cols_ra, cols_rc, props_list)

    # # Make combined nLVO + LVO data in the proportions given.
    props_list = [input_dict['prop_nlvo'], input_dict['prop_lvo']]
    # # Don't calculate the separate redirection approved/rejected bits.
    # Usual care:
    cols_nlvo_usual, cols_lvo_usual, cols_combo_usual = (
        calc.build_columns_combine_occlusions(
            scenario_list=['usual_care'], dummy_col=col_zero))
    # Keep a copy of these for later:
    cols_usual = [cols_nlvo_usual, cols_lvo_usual, cols_combo_usual]
    # Only keep existing columns:
    cols_nlvo_usual, cols_lvo_usual, cols_combo_usual = (
        calc.keep_existing_cols(df_lsoa, cols_nlvo_usual, cols_lvo_usual,
                                cols_combo_usual))
    # Make the new data:
    df_lsoa = calc.combine_results(df_lsoa, cols_nlvo_usual, cols_lvo_usual,
                                   cols_combo_usual, props_list)

    # Redirection:
    cols_nlvo_redir, cols_lvo_redir, cols_combo_redir = (
        calc.build_columns_combine_occlusions(
            scenario_list=['redirection_considered'], dummy_col=col_zero))
    # Keep a copy of these for later:
    cols_redir = [cols_nlvo_redir, cols_lvo_redir, cols_combo_redir]
    # Only keep existing columns:
    cols_nlvo_redir, cols_lvo_redir, cols_combo_redir = (
        calc.keep_existing_cols(df_lsoa, cols_nlvo_redir, cols_lvo_redir,
                                cols_combo_redir))
    # Make the new data:
    df_lsoa = calc.combine_results(df_lsoa, cols_nlvo_redir, cols_lvo_redir,
                                   cols_combo_redir, props_list)
    # Remove the dummy column now it's no longer needed:
    df_lsoa = df_lsoa.drop(col_zero, axis='columns')

    # Calculate diff: redirect minus usual care:
    # Combine the three lists into one long list and remove repeats:
    cols_usual = sum([sorted(list(set(c))) for c in cols_usual], [])
    cols_redir = sum([sorted(list(set(c))) for c in cols_redir], [])
    # Only keep useful columns that exist in the data:
    cols_usual = [c for c in cols_usual if
                  ((c != col_zero) & (c in df_lsoa.columns))]
    cols_redir = [c for c in cols_redir if
                  ((c != col_zero) & (c in df_lsoa.columns))]
    # Make matching columns for the difference names:
    diff_str = 'diff_redirection_considered_minus_usual_care'
    cols_diff = [c.replace('usual_care', diff_str) for c in cols_usual]
    # Make new data:
    props_list = [1.0, -1.0]
    df_lsoa = calc.combine_results(
        df_lsoa, cols_redir, cols_usual, cols_diff, props_list)

    return df_lsoa


def calculate_pdeath_for_combo_groups(
        df_pdeath, scenarios, input_dict,
        treatment_types=['ivt', 'mt', 'ivt_mt']
        ):
    """
    f'{s}_probdeath_nlvo_ivt',
    f'{s}_probdeath_lvo_ivt',
    f'{s}_probdeath_lvo_mt',
    f'{s}_probdeath_lvo_ivt_mt',
    """

    # Make "redirection considered" group.
    # For nLVO with IVT:
    cols_rr = ['redirection_rejected_probdeath_nlvo_ivt']
    cols_ra = ['redirection_approved_probdeath_nlvo_ivt']
    cols_rc = ['redirection_considered_probdeath_nlvo_ivt']
    prop_nlvo_redirected = (1.0 - input_dict['specificity'])
    props_list = [1.0 - prop_nlvo_redirected, prop_nlvo_redirected]
    df_pdeath = calc.combine_results(
        df_pdeath, cols_rr, cols_ra, cols_rc, props_list)
    # For LVO groups:
    cols_rr = [f'redirection_rejected_probdeath_lvo_{t}'
               for t in treatment_types]
    cols_ra = [f'redirection_approved_probdeath_lvo_{t}'
               for t in treatment_types]
    cols_rc = [f'redirection_considered_probdeath_lvo_{t}'
               for t in treatment_types]
    prop_lvo_redirected = input_dict['sensitivity']
    props_list = [1.0 - prop_lvo_redirected, prop_lvo_redirected]
    df_pdeath = calc.combine_results(
        df_pdeath, cols_rr, cols_ra, cols_rc, props_list)

    # Combine nLVO and LVO groups.
    # Set up data for no treatment:
    dist_dict = load_reference_mrs_dists()
    df_pdeath['probdeath_nlvo_no_treatment'] = (
        dist_dict['nlvo_no_treatment_noncum'][-1])
    # Gather the column names here:
    cols_nlvo = []
    cols_lvo = []
    cols_combo = []
    for s in scenarios + ['redirection_considered']:
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
    page_title='OPTIMIST',
    page_icon=':ambulance:',
    layout='wide'
    )

try:
    page_last_run = st.session_state['page_last_run']
    if page_last_run != 'OPTIMIST':
        # Clear the MUSTER results.
        keys_to_del = list(st.session_state.keys())
        for key in keys_to_del:
            del st.session_state[key]
except KeyError:
    # No page has been run yet.
    pass
st.session_state['page_last_run'] = 'OPTIMIST'

# """
# TO DO - check geopandas version - updated with statsmodels?

# Conversion from lat/long to BNG isn't working
# returns inf coordinates, ends up with points instead of polygons

# requirements file says 0.14.2, current install is 0.14.3

# """

# # All msoa shapes:
# from utilities.maps import _import_geojson
# import os

# gdf_boundaries_msoa = geopandas.read_file(os.path.join('data', 'outline_msoa11cds.geojson'))

# st.write(gdf_boundaries_msoa.crs)

# # If crs is given in the file, geopandas automatically
# # pulls it through. Convert to National Grid coordinates:
# if gdf_boundaries_msoa.crs != 'EPSG:27700':
#     gdf_boundaries_msoa = gdf_boundaries_msoa.to_crs('EPSG:27700')#

# # gdf_boundaries_msoa = _import_geojson(
# #     'MSOA11NM',
# #     # path_to_file=os.path.join('data', 'MSOA_Dec_2011_Boundaries_Super_Generalised_Clipped_BSC_EW_V3_2022_7707677027087735278.geojson')# 'MSOA_V3_reduced_simplified.geojson')
# #     # path_to_file=os.path.join('data', 'MSOA_V3_reduced_simplified.geojson')
# #     path_to_file=os.path.join('data', 'outline_msoa11cds.geojson')
#     # )
# st.write(gdf_boundaries_msoa['geometry'])
# for col in gdf_boundaries_msoa.columns:
#     st.write(gdf_boundaries_msoa[col])
# st.stop()

# import utilities.utils as utils
# utils.make_outline_ambo()
# # utils.make_outline_lsoa_limit_to_england()
# # # utils.make_outline_msoa_from_lsoa()
# # # utils.make_outline_icbs('icb')
# # # utils.make_outline_icbs('isdn')
# # # utils.make_outline_england_wales()
# st.stop()


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
        container_inputs_population    # c3
    ) = st.columns(3)
with container_unit_services:
    st.header('Stroke unit services')
    (
        container_unit_services_top,  # cu1
        container_services_map,        # cu2
        container_services_dataeditor
    ) = st.columns([2, 4, 8])
with container_services_dataeditor:
    st.info(''.join([
        'To update the services, ',
        'click the tick-boxes in the table.'
        ]), icon='➡️')

with tab_results:
    container_rerun = st.container()

    st.markdown('## Summary results')
    # Places to put summary boxes:
    container_summary_results = st.container()

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
        container_mrs_limit = st.container()
        container_mrs_dists = st.empty()

with st.sidebar:
    with st.expander('Accessibility & advanced options'):
        container_select_outcome = st.container()
        container_select_cmap = st.container()
        container_select_vlim = st.container()

with container_intro:
    st.markdown('# Benefit in outcomes from redirection')


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
        'excluding the travel times to their chosen stroke units.'
    ]))
with container_inputs_standard:
    input_dict = inputs.select_parameters_pathway_optimist(input_dict)
with container_inputs_population:
    st.markdown('### Population inputs')
    population_dict = inputs.select_parameters_population_optimist()

# Combine the two input dicts:
input_dict = input_dict | population_dict

with container_services_dataeditor:
    df_unit_services, df_unit_services_full = (
        inputs.select_stroke_unit_services(use_msu=False))

# Calculate times to treatment without travel.
# These are used in the main_calculations and also displayed on
# the inputs page as a sanity check.
treatment_times_without_travel = (
    calc.calculate_times_to_treatment_without_travel_usual_care(input_dict))
treatment_times_with_prehospdiag = (
    calc.calculate_times_to_treatment_without_travel_prehospdiag(input_dict))
# Combine these:
treatment_times_without_travel = (
    treatment_times_without_travel | treatment_times_with_prehospdiag
    )

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
        inputs.select_stroke_type(use_combo_stroke_types=True))


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
            dict_colours_diff, f'{outcome_type_str} benefit of redirection'))
    dict_colours_pop['vmin'], dict_colours_pop['vmax'] = (
        inputs.select_map_colour_limits(dict_colours_pop, column_pop_pretty))


# ######################################
# ########## PLOT USER INPUTS ##########
# ######################################

# ----- Timeline -----
# Load emoji and labels:
timeline_display_dict = timeline.get_timeline_display_dict()
# Create timelines:
time_dicts = timeline.build_time_dicts_optimist(input_dict)
time_offsets, tmax = timeline.build_time_dicts_for_plot_optimist(
    time_dicts, gap_between_chunks=45)

# Make subsets of the dictionaries to be displayed:
time_keys_standard = ['prehosp_usual_care', 'prehosp_prehospdiag',
                      'ivt_only_unit', 'mt_transfer_unit', 'ivt_mt_unit']
time_dicts_standard, time_offsets_standard = (
    timeline.subset_time_dicts(time_dicts, time_offsets, time_keys_standard))

# Draw the timelines:
with container_timeline_standard:
    timeline.plot_timeline(
        time_dicts_standard,
        timeline_display_dict,
        y_vals=[0, 2, 1.5, 1.5, 0.5],  # timeline fragment centres
        time_offsets=time_offsets_standard,
        tmax=tmax,
        tmin=-10.0
        )


# ----- Treatment times summary -----
df_treatment_times = (
    timeline.make_treatment_time_df_optimist(treatment_times_without_travel))
# Display the times:
times_explanation_usual_str = ('''
+ The "fastest" time to MT is when the first stroke unit provides MT.
+ The "slowest" time to MT is when a transfer to the MT unit is needed.
''')
with container_inputs_summary:
    st.markdown('Summary of treatment times:')
    st.table(df_treatment_times)
    st.markdown(times_explanation_usual_str)


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
    'ivt': [[1, 1]],  # first map only
    'mt': [[1, 1]]    # both maps
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
        outline_name_for_unit_map, geo, use_msu=False)

# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [
    gdf_catchment_lhs_for_unit_map,
    ]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        gdf['x'], gdf['y'] = maps.convert_shapely_polys_into_xy(gdf)

with container_services_map:
    plot_maps.plotly_unit_map(
        traces_units,
        unit_subplot_dict,
        gdf_catchment_lhs_for_unit_map,
        outline_names_col_for_unit_map,
        outline_name_for_unit_map,
        subplot_titles=['']
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

        scenarios = [
            'usual_care', 'redirection_rejected', 'redirection_approved']
        st.session_state['df_lsoa'] = (
            calc.build_full_lsoa_outcomes_from_unique_time_results(
                df_times, df_outcomes_ivt, df_outcomes_mt, scenarios)
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
            df_times.copy().reset_index().rename(
                columns={'LSOA': 'lsoa'}).set_index('lsoa'),
            st.session_state['df_lsoa'][cols_ivt_better],
            left_index=True, right_index=True, how='left'
            )
        # Now gather P(death):
        df_pdeath = calc.gather_pdeath_from_unique_time_results(
            df_pdeath.reset_index(), st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'], scenarios,
        )
        df_pdeath = df_pdeath.set_index('lsoa')
        # Calculate P(death) for combined groups,
        # mix of nLVO and LVO and "redirection considered" groups.
        df_pdeath = calculate_pdeath_for_combo_groups(
            df_pdeath, scenarios, input_dict)

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
            st.session_state['df_ambo'],
            st.session_state['df_benefit_icb'],
            st.session_state['df_benefit_isdn'],
            st.session_state['df_benefit_nearest_ivt'],
            st.session_state['df_benefit_ambo'],
        ) = calc.group_results_by_region(
            st.session_state['df_lsoa'].reset_index().rename(
                columns={'LSOA': 'lsoa'}),
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
            # Check for any of these changing:
            inputs_changed = (c1 | c2 | c3)
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


if 'df_lsoa' in st.session_state.keys():
    pass
else:
    # This hasn't been created yet and so the results cannot be drawn.
    st.stop()


# #################################################
# ########## RESULTS - SUMMARY BY REGION ##########
# #################################################

@st.fragment
def display_summary_stats():
    # Select a region based on what's actually in the data,
    # not by guessing in advance which IVT units are included for example.
    region_options_dict = inputs.load_region_lists(df_unit_services_full)
    select_options = []  # ['National']
    for key, region_list in region_options_dict.items():
        select_options += [f'{key}: {v}' for v in region_list]
    selected_rows = st.multiselect('Highlighted regions', select_options)
    limit_summary_to_benefit = st.toggle('Limit summary stats to areas that benefit from redirection')
    container_summary_metrics = st.container(
        horizontal=True, horizontal_alignment='left', border=False)

    for r, row in enumerate(selected_rows):
        region_type = row.split(': ')[0]
        if row.startswith('ISDN'):
            df_here = st.session_state['df_isdn']
            df_ben = st.session_state['df_benefit_isdn']
        elif row.startswith('ICB'):
            df_here = st.session_state['df_icb']
            df_ben = st.session_state['df_benefit_icb']
        elif row.startswith('Nearest unit'):
            df_here = st.session_state['df_nearest_ivt']
            df_ben = st.session_state['df_benefit_nearest_ivt']
        elif row.startswith('Amb'):
            df_here = st.session_state['df_ambo']
            df_ben = st.session_state['df_benefit_ambo']
        else:
            st.write('meep')

        if row.startswith('Nearest'):
            series_row = df_here.loc[df_here['ssnap_name'] == row.replace(f'{region_type}: ', '')].squeeze()
            series_ben = df_ben.loc[df_ben['ssnap_name'] == row.replace(f'{region_type}: ', '')].squeeze()
        else:
            series_row = df_here.loc[row.replace(f'{region_type}: ', '')].squeeze()
            series_ben = df_ben.loc[row.replace(f'{region_type}: ', '')].squeeze()

        with container_summary_metrics:
            ch = st.container(border=True)
        with ch:
            title = f'__{row}__'

            # st.markdown(title.ljust(longest_str, '~'))
            st.markdown(title)

            s = series_ben if limit_summary_to_benefit else series_row
            if s.empty:
                st.write('No patients in this category.')
                pass
            else:
                # Only patients who benefit:
                t = s['transfer_required']
                v_uc = s['usual_care_combo_ivt_mt_mrs_0-2']
                v_rc = s['redirection_considered_combo_ivt_mt_mrs_0-2']
                v_di = s['diff_redirection_considered_minus_usual_care_combo_ivt_mt_mrs_0-2']
                d_uc = s['usual_care_probdeath_combo_ivt_mt']
                d_rc = s['redirection_considered_probdeath_combo_ivt_mt']
                d_di = d_rc - d_uc

                st.write('Proportion of patients with:')
                st.metric('Nearest unit offering MT', f'{1.0 - t:.1%}')
                st.metric('mRS<=2', f'{v_rc:.1%}', f'{v_di:.1%} from usual care')
                st.metric('Dead', f'{d_rc:.1%}', f'{d_di:.1%} from usual care', delta_color='inverse')


with container_summary_results:
    display_summary_stats()


# #########################################
# ########## RESULTS - FULL DATA ##########
# #########################################
with container_results_tables:

    table_choice = st.selectbox(
        'Display the following results table:',
        options=[
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
        df_here = st.session_state['df_nearest_ivt']
        column_config = {
            'transfer_required': st.column_config.CheckboxColumn(),
            }

    elif 'ISDN' in table_choice:
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        df_here = st.session_state['df_isdn']
        column_config = {}

    elif 'ICB' in table_choice:
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        df_here = st.session_state['df_icb']
        column_config= {}

    elif 'ambulance' in table_choice:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each ambulance service.'
            ]))
        df_here = st.session_state['df_ambo']
        column_config = {}
    else:
        df_here = st.session_state['df_lsoa']
        column_config = {
            'transfer_required': st.column_config.CheckboxColumn(),
            }

    # Main results table:
    st.dataframe(
        df_here,
        # Set some columns to bool for nicer display:
        column_config=column_config,
        )


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

with container_mrs_limit:
    limit_mrs_dists = st.toggle(
        'Limit mRS distributions to LSOA whose nearest stroke units do not offer MT.',
        value=True
    )
if limit_mrs_dists:
    # Limit the mRS data to only LSOA that are in the redirection zone,
    # i.e. remove anything that has the same nearest IVT unit and
    # nearest MT unit.
    col_to_mask_mrs = 'transfer_required'
else:
    col_to_mask_mrs = ''


# Which mRS distributions will be shown on the bars:
scenario_mrs = ['usual_care', 'redirection_considered']
# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)
bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]

# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.

# Pick out useful bits from the full outcome results:
scenarios = ['usual_care', 'redirection_rejected', 'redirection_approved']
all_mrs_scenarios = scenarios + ['redirection_considered']
dict_of_dfs = {}
for s in scenarios:
    cols_to_copy = [
        'Admissions',
        f'{s}_ivt',
        f'{s}_mt',
        f'{s}_lvo_ivt_better_than_mt',
        'nearest_ivt_unit_name'
        ]
    if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
        cols_to_copy.append(col_to_mask_mrs)

    df_mrs_s = st.session_state['df_lsoa'][cols_to_copy].copy()
    df_mrs_s = df_mrs_s.rename(columns={
        f'{s}_ivt': 'time_to_ivt',
        f'{s}_mt': 'time_to_mt',
        f'{s}_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
    })
    # Merge in region info:
    df_mrs_s = pd.merge(
        df_mrs_s.reset_index(), df_lsoa_regions,
        on='lsoa', how='left'
        ).set_index('lsoa')
    # Store result:
    dict_of_dfs[s] = df_mrs_s.copy()
# For redirection considered, need to account for every combo
# of time to treatment when redir approved and when redir
# rejected.
cols_to_copy = [
    'Admissions',
    'nearest_ivt_unit_name'
    ]
for s in ['redirection_rejected', 'redirection_approved']:
    cols_to_copy += [
        f'{s}_ivt',
        f'{s}_mt',
        f'{s}_lvo_ivt_better_than_mt',
    ]
if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
    cols_to_copy.append(col_to_mask_mrs)

df_mrs_s = st.session_state['df_lsoa'][cols_to_copy].copy()
for s in ['redirection_rejected', 'redirection_approved']:
    df_mrs_s = df_mrs_s.rename(columns={
        f'{s}_ivt': f'{s}_time_to_ivt',
        f'{s}_mt': f'{s}_time_to_mt',
        # f'{s}_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
    })
# Merge in region info:
df_mrs_s = pd.merge(
    df_mrs_s.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')
# Store result:
dict_of_dfs['redirection_considered'] = df_mrs_s.copy()


@st.fragment
def display_mrs_dists():
    (
        container_bars,
        container_mrs_input,
    ) = st.columns(2)

    with container_mrs_input:
        # User input:
        bar_option = st.selectbox('Region for mRS distributions', bar_options)

    # Set up where the data should come from -
    # which region type was selected, and which region name.
    region_selected, col_region = mrs.pick_out_region_name(bar_option)

    # Find reference mRS distributions (no treatment).
    # If occ_type is nLVO or LVO, this returns the normal dists.
    # Otherwise it returns a scaled sum of the nLVO and LVO dists.
    dist_ref_cum, dist_ref_noncum = mrs.load_no_treatment_mrs_dists(
        stroke_type, input_dict['prop_nlvo'], input_dict['prop_nlvo'])
    # Store no-treatment data:
    dict_no_treatment = {
        'noncum': dist_ref_noncum,
        'cum': dist_ref_cum,
        'std': None
    }

    # Store results in here:
    keys = ['no_treatment'] + all_mrs_scenarios

    # Decide whether to use no-treatment dists or to
    # fish dists out of the big mRS lists.
    use_ref_data = (True if
                    ((stroke_type == 'nlvo') & (treatment_type == 'mt'))
                    else False)
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
        for key in all_mrs_scenarios:
            mrs_dfs_dict[key] = {}
        lsoa_names = mrs.find_lsoa_names_to_keep(
            dict_of_dfs['usual_care'],
            col_to_mask_mrs,
            col_region,
            region_selected
            )
        mrs_dfs_dict, dist_cols = mrs.find_total_mrs_for_unique_times(
            dict([(k, dict_of_dfs[k]) for k in scenarios]),
            lsoa_names,
            treat_type,
            stroke_types,
            st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'],
            )
        dfs_dict_rc, dist_cols_rc = mrs.find_total_mrs_for_unique_times(
            dict([(k, dict_of_dfs[k]) for k in ['redirection_considered']]),
            lsoa_names,
            treat_type,
            stroke_types,
            st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'],
            multi_scens=['redirection_rejected', 'redirection_approved']
            )

        df = dfs_dict_rc['redirection_considered']
        dist_cols_to_combine = [c for c in df.columns if 'mrs' in c]
        dist_cols_to_combine = sorted(list(set(
            [d.replace('_redirection_rejected', '').replace('_redirection_approved', '')
             for d in dist_cols_to_combine
             ])))

        for d in dist_cols_to_combine:
            if 'nlvo' in d:
                prop_nlvo_redirected = (1.0 - input_dict['specificity'])
                props_list = [1.0 - prop_nlvo_redirected, prop_nlvo_redirected]
            else:
                prop_lvo_redirected = input_dict['sensitivity']
                props_list = [1.0 - prop_lvo_redirected, prop_lvo_redirected]

            col_rr = f'{d}_redirection_rejected'
            col_ra = f'{d}_redirection_approved'
            df[d] = (
                df[col_rr].values * props_list[0] +
                df[col_ra].values * props_list[1]
            )
            df = df.drop([col_rr, col_ra], axis='columns')

        # Place this data with the other scenarios:
        mrs_dfs_dict['redirection_considered'] = df

        # Calculate "redirection considered":
        nlvo_cols = [c for c in dist_cols if 'nlvo' in c]
        lvo_cols = [c for c in dist_cols if (('lvo' in c) & ('nlvo' not in c))]

        if stroke_type not in ['nlvo', 'lvo']:
            # Calculate combined nLVO + LVO data:
            combo_cols = [c.replace('lvo', 'combo') for c in lvo_cols]
            for key, df in mrs_dfs_dict.items():
                df[combo_cols] = (
                    (df[nlvo_cols] * input_dict['prop_nlvo']).values +
                    (df[lvo_cols] * input_dict['prop_lvo']).values
                )
                mrs_dfs_dict[key] = df

        # Pick out which columns should be displayed:
        if stroke_type == 'nlvo':
            dist_cols = nlvo_cols
        elif stroke_type == 'lvo':
            dist_cols = lvo_cols
        else:
            dist_cols = combo_cols

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

    # Prettier formatting for the plot title:
    col_pretty = ''.join([
        f'{stroke_type_str}, ',
        f'{treatment_type_str}'
        ])
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
    'Benefit of redirection over usual care',
    column_pop_pretty
]

# ----- Find data for colours -----

# df_lsoa column names are in the format:
# `usual_care_lvo_ivt_mt_utility_shift`, i.e.
# '{scenario}_{occlusion}_{treatment}_{outcome}' with these options:
#
# +---------------------------+------------+------------+---------------+
# | Scenarios                 | Occlusions | Treatments | Outcomes      |
# +---------------------------+------------+------------+---------------+
# | usual_care                | nlvo       | ivt        | utility_shift |
# | redirection_approved      | lvo        | mt         | mrs_shift     |
# | redirection_rejected      | combo      | ivt_mt     | mrs_0-2       |
# | redirection_considered    |            |            |               |
# | diff_redirection_considered_minus_usual_care        |               |
# +---------------------------+------------+------------+---------------+
#
# There is not a separate column for "no treatment" to save space.

# Find the names of the columns that contain the data
# that will be shown in the colour maps.
if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # Use no-treatment data.
    # Set this to something that doesn't exist so it fails a check later.
    # The check will set all values to zero.
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
        'diff_redirection_considered_minus_usual_care',
        stroke_type, t, outcome_type
        ])


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
    f'{outcome_type_str}: Benefit of redirection over usual care')
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
    ) = calc.load_or_calculate_region_outlines(outline_name, st.session_state['df_lsoa'])


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
    'ivt': [[1, 1]],          # left map only
    'mt': [[1, 1], [1, 2]]    # both maps
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
