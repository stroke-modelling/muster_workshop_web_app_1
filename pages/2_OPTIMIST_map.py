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
from utilities.maps_raster import make_raster_from_vectors, \
    set_up_raster_transform
import utilities.colour_setup as colour_setup
import utilities.inputs as inputs
import utilities.plot_timeline as timeline


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
    cols_ivt = [
        'usual_care_ivt',
        'redirection_approved_ivt',
        'redirection_rejected_ivt'
        ]
    cols_mt = [
        'usual_care_mt',
        'redirection_approved_mt',
        'redirection_rejected_mt'
        ]
    times_to_ivt = sorted(list(set(df_times[cols_ivt].values.flatten())))
    times_to_mt = sorted(list(set(df_times[cols_mt].values.flatten())))

    # ----- Outcomes for unique treatment times -----
    # Run the outcome model for only the unique treatment times
    # instead of one row per LSOA.
    # Run results for IVT and for MT separately.
    outcomes_by_stroke_type_ivt_only, outcomes_by_stroke_type_mt_only = (
        calc.run_outcome_model_for_unique_times(times_to_ivt, times_to_mt))

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
def gather_outcomes_by_region(
            df_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            df_mrs_ivt,
            df_mrs_mt,
            df_lsoa_regions,
            input_dict
            ):
    df_lsoa = calc.build_full_lsoa_outcomes_from_unique_time_results_optimist(
            df_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            df_mrs_ivt,
            df_mrs_mt,
    )

    df_lsoa = df_lsoa.rename(columns={'LSOA': 'lsoa'})
    df_lsoa = df_lsoa.set_index('lsoa')

    # Extra calculations for redirection:
    # Combine redirection rejected and approved results in
    # proportions given by specificity and sensitivity.
    # This creates columns labelled "redirection_considered".
    redirect_dict = {
        'sensitivity': input_dict['sensitivity'],
        'specificity': input_dict['specificity'],
    }
    df_lsoa = calc.combine_results_by_redirection(df_lsoa, redirect_dict)

    # Make combined nLVO + LVO data in the proportions given:
    # Combine for "usual care":
    prop_dict = {
        'nlvo': input_dict['prop_nlvo'],
        'lvo': input_dict['prop_lvo']
    }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['usual_care'])
    # Combine for redirection considered:
    # prop_dict = {
    #     'nlvo': input_dict['prop_redirection_considered_nlvo'],
    #     'lvo': input_dict['prop_redirection_considered_lvo']
    # }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['redirection_considered'])
    # Don't calculate the separate redirection approved/rejected bits.

    # Calculate diff - redirect minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['redirection_considered', 'usual_care']
        )

    df_icb, df_isdn, df_nearest_ivt, df_ambo = calc.group_results_by_region(
        df_lsoa.reset_index().rename(columns={'LSOA': 'lsoa'}),
        df_unit_services,
        df_lsoa_regions
        )

    return df_lsoa, df_icb, df_isdn, df_nearest_ivt, df_ambo


@st.cache_data
def old_main_calculations(input_dict, df_unit_services):
    # Times to treatment:
    df_travel_times = calc.calculate_geography(df_unit_services).copy()
    df_travel_times = df_travel_times.set_index('LSOA')

    # Add travel times to the pathway timings to get treatment times.
    df_outcome_uc = calc.make_outcome_inputs_usual_care(
        input_dict, df_travel_times)
    df_outcome_ra = calc.make_outcome_inputs_redirection_approved(
        input_dict, df_travel_times)
    df_outcome_rr = calc.make_outcome_inputs_redirection_rejected(
        input_dict, df_travel_times)
    dict_outcome_inputs = {
        'usual_care': df_outcome_uc,
        'redirection_approved': df_outcome_ra,
        'redirection_rejected': df_outcome_rr,
    }

    # Process LSOA and calculate outcomes:
    df_lsoa, df_mrs = calc.calculate_outcomes(
        dict_outcome_inputs, df_unit_services, df_travel_times)

    # Extra calculations for redirection:
    # Combine redirection rejected and approved results in
    # proportions given by specificity and sensitivity.
    # This creates columns labelled "redirection_considered".
    redirect_dict = {
        'sensitivity': input_dict['sensitivity'],
        'specificity': input_dict['specificity'],
    }
    df_lsoa = calc.combine_results_by_redirection(df_lsoa, redirect_dict)
    df_mrs = calc.combine_results_by_redirection(
        df_mrs, redirect_dict, combine_mrs_dists=True)

    # Make combined nLVO + LVO data in the proportions given:
    # Combine for "usual care":
    prop_dict = {
        'nlvo': input_dict['prop_nlvo'],
        'lvo': input_dict['prop_lvo']
    }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['usual_care'])
    df_mrs = calc.combine_results_by_occlusion_type(
        df_mrs, prop_dict, combine_mrs_dists=True,
        scenario_list=['usual_care'])
    # Combine for redirection considered:
    # prop_dict = {
    #     'nlvo': input_dict['prop_redirection_considered_nlvo'],
    #     'lvo': input_dict['prop_redirection_considered_lvo']
    # }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['redirection_considered'])
    df_mrs = calc.combine_results_by_occlusion_type(
        df_mrs, prop_dict, combine_mrs_dists=True,
        scenario_list=['redirection_considered'])
    # Don't calculate the separate redirection approved/rejected bits.

    # Calculate diff - redirect minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['redirection_considered', 'usual_care']
        )
    df_mrs = calc.combine_results_by_diff(
        df_mrs,
        scenario_types=['redirection_considered', 'usual_care'],
        combine_mrs_dists=True
        )

    df_icb, df_isdn, df_nearest_ivt, df_ambo = calc.group_results_by_region(
        df_lsoa, df_unit_services)

    return df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt, df_ambo


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
cmap_names = [
    'iceburn_r', 'seaweed', 'fusion', 'waterlily'
    ]
cmap_names += [c[:-2] if c.endswith('_r') else f'{c}_r'
               for c in cmap_names]
with container_select_cmap:
    st.markdown('### Colour schemes')
    cmap_name = inputs.select_colour_maps(cmap_names)
    cmap_diff_name = cmap_name
    cmap_pop_name = cmap_name
# If we're showing mRS scores then flip the colour maps:
if outcome_type == 'mrs_shift':
    cmap_name += '_r'
    cmap_diff_name += '_r'
    # Remove any double reverse reverse.
    if cmap_name.endswith('_r_r'):
        cmap_name = cmap_name[:-4]
    if cmap_diff_name.endswith('_r_r'):
        cmap_diff_name = cmap_diff_name[:-4]


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
# Set limits for the colour scale:
dict_colours_pop = {
    'vmin': 0.0,
    'vmax': 100.0,
    'step_size': 100.0,  # unused
}


# ----- Colour limits -----
# Load colour limits info (vmin, vmax, step_size):
dict_colours, dict_colours_diff = (
    colour_setup.load_colour_limits(outcome_type))
# User inputs for vmin and vmax with loaded values as defaults:
with container_select_vlim:
    st.markdown('### Colour limits')
    vmin = st.number_input(
        f'{outcome_type_str}: minimum value',
        value=dict_colours['vmin'],
        help=f'Default value: {dict_colours["vmin"]}',
    )
    vmax = st.number_input(
        f'{outcome_type_str}: maximum value',
        value=dict_colours['vmax'],
        help=f'Default value: {dict_colours["vmax"]}',
    )
    vmin_diff = st.number_input(
        f'{outcome_type_str} benefit of redirection: minimum value',
        value=dict_colours_diff['vmin'],
        help=f'Default value: {dict_colours_diff["vmin"]}',
    )
    vmax_diff = st.number_input(
        f'{outcome_type_str} benefit of redirection: maximum value',
        value=dict_colours_diff['vmax'],
        help=f'Default value: {dict_colours_diff["vmax"]}',
    )
    vmin_pop = st.number_input(
        f'{column_pop_pretty}: minimum value',
        value=dict_colours_pop['vmin'],
        help=f'Default value: {dict_colours_pop["vmin"]}',
    )
    vmax_pop = st.number_input(
        f'{column_pop_pretty}: maximum value',
        value=dict_colours_pop['vmax'],
        help=f'Default value: {dict_colours_pop["vmax"]}',
    )
    # Sanity checks:
    if (
        (vmax <= vmin) |
        (vmax_diff <= vmin_diff) |
        (vmax_pop <= vmin_pop)
            ):
        st.error(
            'Maximum value must be less than the minimum value.', icon='❗')
        st.stop()
# Overwrite default values:
dict_colours['vmin'] = vmin
dict_colours['vmax'] = vmax
dict_colours_diff['vmin'] = vmin_diff
dict_colours_diff['vmax'] = vmax_diff
dict_colours_pop['vmin'] = vmin_pop
dict_colours_pop['vmax'] = vmax_pop


# ######################################
# ########## PLOT USER INPUTS ##########
# ######################################

# ----- Timeline -----
time_dicts = timeline.build_time_dicts_optimist(input_dict)
timeline_display_dict = timeline.get_timeline_display_dict()

# Setup for timeline plot.
# Leave this gap in minutes between separate chunks of pathway:
gap_between_chunks = 45
# Start each chunk at these offsets:
time_offsets = {
    'prehosp_usual_care': 0,
    'prehosp_prehospdiag': 0,
    'ivt_only_unit': (
        gap_between_chunks + 
        max([
            sum(time_dicts['prehosp_usual_care'].values()),
            sum(time_dicts['prehosp_prehospdiag'].values()),
        ])
        ),
    'mt_transfer_unit': (
        gap_between_chunks * 2.0 +
        max([
            sum(time_dicts['prehosp_usual_care'].values()),
            sum(time_dicts['prehosp_prehospdiag'].values()),
        ]) +
        sum(time_dicts['ivt_only_unit'].values())
    ),
    'ivt_mt_unit': (
        gap_between_chunks + 
        max([
            sum(time_dicts['prehosp_usual_care'].values()),
            sum(time_dicts['prehosp_prehospdiag'].values()),
        ])
    ),
}
# Find shared max time for setting same size across multiple plots
# so that 1 minute always spans the same number of pixels.
tmax = max(
    [time_offsets[k] + sum(time_dicts[k].values()) for k in time_dicts.keys()]
) + gap_between_chunks

# Standard pathway data:
time_keys_standard = [
    'prehosp_usual_care',
    'prehosp_prehospdiag',
    'ivt_only_unit',
    'mt_transfer_unit',
    'ivt_mt_unit',
]
time_dicts_standard = dict([(k, time_dicts[k]) for k in time_keys_standard])
time_offsets_standard = dict([
    (k, time_offsets[k]) for k in time_keys_standard])

# Draw the timelines:
with container_timeline_standard:
    timeline.plot_timeline(
        time_dicts_standard,
        timeline_display_dict,
        y_vals=[0, 2, 1.5, 1.5, 0.5],
        time_offsets=time_offsets_standard,
        tmax=tmax,
        tmin=-10.0
        )

# ----- Treatment times -----
# Add strings to show travel times:
usual_care_time_to_ivt_str = ' '.join([
    f'{treatment_times_without_travel["usual_care_time_to_ivt"]}',
    '+ 🚑 travel to nearest unit'
    ])
usual_care_mt_no_transfer_str = ' '.join([
    f'{treatment_times_without_travel["usual_care_mt_no_transfer"]}',
    '+ 🚑 travel to nearest unit'
    ])
usual_care_mt_transfer_str = ' '.join([
    f'{treatment_times_without_travel["usual_care_mt_transfer"]}',
    '+ 🚑 travel to nearest unit',
    '+ 🚑 travel between units'
    ])

prehospdiag_rej_time_to_ivt_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_time_to_ivt"]}',
    '+ 🚑 travel to nearest unit'
    ])
prehospdiag_rej_mt_no_transfer_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_mt_no_transfer"]}',
    '+ 🚑 travel to nearest unit'
    ])
prehospdiag_rej_mt_transfer_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_mt_transfer"]}',
    '+ 🚑 travel to nearest unit',
    '+ 🚑 travel between units'
    ])

prehospdiag_app_time_to_ivt_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_time_to_ivt"]}',
    '+ 🚑 travel to MT unit'
    ])
prehospdiag_app_mt_no_transfer_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_mt_no_transfer"]}',
    '+ 🚑 travel to MT unit'
    ])
prehospdiag_app_mt_transfer_str = ' '.join([
    f'{treatment_times_without_travel["prehospdiag_mt_no_transfer"]}',
    '+ 🚑 travel to MT unit',
    ])

# Place these into a dataframe:
df_treatment_times = pd.DataFrame(
    [[usual_care_time_to_ivt_str, prehospdiag_rej_time_to_ivt_str, prehospdiag_app_time_to_ivt_str],
     [usual_care_mt_no_transfer_str, prehospdiag_rej_mt_no_transfer_str, prehospdiag_app_mt_no_transfer_str],
     [usual_care_mt_transfer_str, prehospdiag_rej_mt_transfer_str, prehospdiag_app_mt_transfer_str]],
    columns=['Standard pathway', 'Redirection rejected', 'Redirection approved'],
    index=['Time to IVT', 'Time to MT (fastest)', 'Time to MT (slowest)']
)
# Display the times:
times_explanation_usual_str = ('''
+ The "fastest" time to MT is when the first stroke unit provides MT.
+ The "slowest" time to MT is when a transfer to the MT unit is needed.
''')
with container_inputs_summary:
    st.markdown('Summary of treatment times:')
    st.table(df_treatment_times)
    st.markdown(times_explanation_usual_str)


# ----- Maps -----
# Stroke unit scatter markers:
traces_units = plot_maps.create_stroke_team_markers(
    df_unit_services_full)
# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1]]  # both maps
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
        [
            'None',
            'ISDN',
            'ICB',
            'Nearest service',  # hasn't been calculated yet
            'Ambulance service'
            ],
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
        x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
        gdf['x'] = x_list
        gdf['y'] = y_list

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
    inputs_changed = (
        (st.session_state['input_dict'] != input_dict) |
        (
            st.session_state['df_unit_services_on_last_run']['Use_IVT'] !=
            df_unit_services['Use_IVT']
        ).any() |
        (
            st.session_state['df_unit_services_on_last_run']['Use_MT'] !=
            df_unit_services['Use_MT']
        ).any()
    )
except KeyError:
    # First run of the app.
    inputs_changed = False

try:
    df_lsoa_regions = st.session_state['df_lsoa_regions']
except KeyError:
    df_lsoa_regions = inputs.load_lsoa_region_lookups()
    st.session_state['df_lsoa_regions'] = df_lsoa_regions

new_results_run = False

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

        (
            st.session_state['df_lsoa'],
            st.session_state['df_icb'],
            st.session_state['df_isdn'],
            st.session_state['df_nearest_ivt'],
            st.session_state['df_ambo']
        ) = gather_outcomes_by_region(
            df_times,
            df_outcomes_ivt,
            df_outcomes_mt,
            st.session_state['df_mrs_ivt'],
            st.session_state['df_mrs_mt'],
            df_lsoa_regions,
            input_dict
            )

        new_results_run = True

    else:
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

cols_to_copy_redirect_reject = [
    'Admissions',
    'redirection_rejected_ivt',
    'redirection_rejected_mt',
    'redirection_rejected_lvo_ivt_better_than_mt',
    'nearest_ivt_unit_name'
    ]
if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
    cols_to_copy_redirect_reject.append(col_to_mask_mrs)
df_mrs_redirect_reject = st.session_state['df_lsoa'][cols_to_copy_redirect_reject].copy()
df_mrs_redirect_reject = df_mrs_redirect_reject.rename(columns={
    'redirection_rejected_ivt': 'time_to_ivt',
    'redirection_rejected_mt': 'time_to_mt',
    'redirection_rejected_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
})

cols_to_copy_redirect_approve = [
    'Admissions',
    'redirection_approved_ivt',
    'redirection_approved_mt',
    'redirection_approved_lvo_ivt_better_than_mt',
    'nearest_ivt_unit_name'
    ]
if col_to_mask_mrs in st.session_state['df_lsoa'].columns:
    cols_to_copy_redirect_approve.append(col_to_mask_mrs)
df_mrs_redirect_approve = st.session_state['df_lsoa'][cols_to_copy_redirect_approve].copy()
df_mrs_redirect_approve = df_mrs_redirect_approve.rename(columns={
    'redirection_approved_ivt': 'time_to_ivt',
    'redirection_approved_mt': 'time_to_mt',
    'redirection_approved_lvo_ivt_better_than_mt': 'lvo_ivt_better_than_mt'
})

# Merge in region info:
df_mrs_usual_care = pd.merge(
    df_mrs_usual_care.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')
df_mrs_redirect_reject = pd.merge(
    df_mrs_redirect_reject.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')
df_mrs_redirect_approve = pd.merge(
    df_mrs_redirect_approve.reset_index(), df_lsoa_regions,
    on='lsoa', how='left'
    ).set_index('lsoa')

dict_of_dfs = {
    'usual_care': df_mrs_usual_care,
    'redirection_rejected': df_mrs_redirect_reject,
    'redirection_approved': df_mrs_redirect_approve
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

    mrs_lists_dict = mrs.calculate_average_mrs(
        stroke_type,
        treatment_type,
        col_region,
        region_selected,
        col_to_mask_mrs,
        # Setup for mRS dists:
        dict_of_dfs,
        # The actual mRS dists:
        st.session_state['df_mrs_ivt'],
        st.session_state['df_mrs_mt'],
        input_dict
        )

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
    'Benefit of redirection over usual care',
    column_pop_pretty
]


# ----- Set up geodataframe -----
try:
    gdf = st.session_state['gdf']
except KeyError:
    gdf = maps.load_lsoa_gdf()
    st.session_state['gdf_cols'] = gdf.columns

if new_results_run:
    # Remove results from last run:
    gdf = gdf[st.session_state['gdf_cols']]
    # Merge in outcomes data:
    gdf = pd.merge(
        gdf, st.session_state['df_lsoa'],
        left_on='LSOA11NM', right_on='lsoa', how='left'
        )
    # Merge demographic data into gdf:
    gdf = pd.merge(
        gdf, df_demog[['LSOA', column_pop]],
        left_on='LSOA11NM', right_on='LSOA', how='left'
        )
st.session_state['gdf'] = gdf

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


# Pick out the columns of data for the colours:
try:
    vals_for_colours = gdf[column_colours]
    vals_for_colours_diff = gdf[column_colours_diff]
except KeyError:
    # Those columns don't exist in the data.
    # This should only happen for nLVO treated with MT only.
    vals_for_colours = [0] * len(gdf)
    vals_for_colours_diff = [0] * len(gdf)
    # Note: this works for now because expect always no change
    # for added utility and added mrs<=2 with no treatment.

# Pick out values:
vals_for_colours_pop = gdf[column_pop]

# ----- Convert vectors to raster -----
# Set up parameters for conversion to raster:
transform_dict = set_up_raster_transform(gdf, pixel_size=2000)
# Burn geometries for left-hand map...
burned_lhs = make_raster_from_vectors(
    gdf['geometry'],
    vals_for_colours,
    transform_dict['height'],
    transform_dict['width'],
    transform_dict['transform']
)
# ... and right-hand map:
burned_rhs = make_raster_from_vectors(
    gdf['geometry'],
    vals_for_colours_diff,
    transform_dict['height'],
    transform_dict['width'],
    transform_dict['transform']
)
# ... and population map:
burned_pop = make_raster_from_vectors(
    gdf['geometry'],
    vals_for_colours_pop,
    transform_dict['height'],
    transform_dict['width'],
    transform_dict['transform']
)

# Record actual highest and lowest values:
actual_vmin = min(vals_for_colours)
actual_vmax = max(vals_for_colours)
actual_vmin_diff = min(vals_for_colours_diff)
actual_vmax_diff = max(vals_for_colours_diff)
actual_vmin_pop = min(vals_for_colours_pop)
actual_vmax_pop = max(vals_for_colours_pop)
# Put these into a DataFrame:
df_actual_vlim = pd.DataFrame(
    [[actual_vmin, actual_vmin_diff, actual_vmin_pop],
    [actual_vmax, actual_vmax_diff, actual_vmax_pop]],
    columns=subplot_titles,
    index=['Minimum', 'Maximum']
)
with container_actual_vlim:
    st.markdown('Ranges of the plotted data:')
    st.dataframe(df_actual_vlim)
    st.markdown(''.join([
        'The range of the colour scales in the maps can be changed ',
        'using the options in the sidebar.'
        ]))

# ----- Set up colours -----
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
    ) = calc.load_or_calculate_region_outlines(outline_name, st.session_state['df_lsoa'], use_msu=False)


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_catchment_pop, gdf_catchment_lhs, gdf_catchment_rhs]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
        gdf['x'] = x_list
        gdf['y'] = y_list

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
