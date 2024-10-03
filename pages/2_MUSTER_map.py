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

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.plot_mrs_dists as mrs
from utilities.maps_raster import make_raster_from_vectors, \
    set_up_raster_transform
# Containers:
import utilities.inputs as inputs
import utilities.colour_setup as colour_setup
import utilities.plot_timeline as timeline


@st.cache_data
def main_calculations(input_dict, df_unit_services):
    # Times to treatment:
    geo = calc.calculate_geography(df_unit_services)
    # Travel times for each LSOA:
    df_travel_times = geo.combined_data[
        [c for c in geo.combined_data.columns if 'time' in c] +
        ['transfer_required', 'LSOA']
        ]
    df_travel_times = df_travel_times.set_index('LSOA')

    # Add travel times to the pathway timings to get treatment times.
    df_outcome_uc = calc.make_outcome_inputs_usual_care(
        input_dict, df_travel_times)
    df_outcome_msu = calc.make_outcome_inputs_msu(
        input_dict, df_travel_times)
    dict_outcome_inputs = {
        'usual_care': df_outcome_uc,
        'msu': df_outcome_msu,
    }

    # Process LSOA and calculate outcomes:
    df_lsoa, df_mrs = calc.calculate_outcomes(
        dict_outcome_inputs, df_unit_services, geo.combined_data)

    # Calculate diff - msu minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['msu', 'usual_care']
        )
    df_mrs = calc.combine_results_by_diff(
        df_mrs,
        scenario_types=['msu', 'usual_care'],
        combine_mrs_dists=True
        )

    # Place probabilities of death into st.session_state['df_lsoa'] data
    # so that they are displayed in the results tables.
    cols_probs_of_death = [
        'usual_care_lvo_ivt_mrs_dists_noncum_6',
        'usual_care_lvo_ivt_mt_mrs_dists_noncum_6',
        'usual_care_lvo_mt_mrs_dists_noncum_6',
        'usual_care_nlvo_ivt_mrs_dists_noncum_6',
        'msu_lvo_ivt_mrs_dists_noncum_6',
        'msu_lvo_ivt_mt_mrs_dists_noncum_6',
        'msu_lvo_mt_mrs_dists_noncum_6',
        'msu_nlvo_ivt_mrs_dists_noncum_6',
    ]
    df_lsoa = pd.merge(
        df_lsoa, df_mrs[cols_probs_of_death],
        left_index=True, right_index=True, how='left'
    )

    df_icb, df_isdn, df_nearest_ivt, df_ambo = calc.group_results_by_region(
        df_lsoa, df_unit_services)

    return df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt, df_ambo


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )


# #####################################
# ########## CONTAINER SETUP ##########  --------------------------------------------- update me
# #####################################
# Make containers:
# +-----------------------------------------------+
# |                container_intro                |
# +-------------------------+---------------------+
# |      container_map      | container_mrs_dists |
# +-------------------------+---------------------+
# |              container_map_inputs             |
# +-----------------------------------------------+
# |            container_results_tables           |
# +-----------------------------------------------+

# Sidebar:
# form
#   +--------------------------+
#   |     container_inputs     |
#   +--------------------------+
#   |  container_unit_services |
#   +--------------------------+
# /form
# v Accessibility & advanced options
#   +--------------------------+
#   | container_select_outcome |
#   +--------------------------+
#   |  container_select_cmap   |
#   +--------------------------+

container_intro = st.container()
tab_inputs, tab_results = st.tabs(['Inputs', 'Results'])
with tab_inputs:
    container_inputs = st.container()
    # container_timeline_plot = st.container()
    container_unit_services = st.container()

with container_inputs:
    container_inputs_top = st.container()
    (
        container_inputs_standard,
        container_timeline_standard,
        container_inputs_msu,
        container_timeline_msu
    ) = st.columns(4)
with container_unit_services:
    (
        container_unit_services_top,
        container_services_map
    ) = st.columns([1, 2])
    with container_unit_services_top:
        st.header('Stroke unit services')
    container_services_buttons = st.container()
    container_services_dataeditor = st.container()
with container_services_buttons:    
    st.markdown('To update the services, use the following buttons and click the tick-boxes in the table.')

with tab_results:
    container_rerun = st.container()
    container_map, container_mrs_dists_etc = st.columns([2, 1])
    # Convert the map container to empty so that the placeholder map
    # is replaced once the real map is ready.
    with container_map:
        container_map = st.empty()
    # Convert mRS dists to empty so that re-running a fragment replaces
    # the bars rather than displays the new plot in addition.
    with container_mrs_dists_etc:
        container_mrs_dists = st.empty()
    container_map_inputs = st.container(border=True)
    with container_map_inputs:
        st.markdown('__Plot options__')
        (container_input_treatment,
         container_input_stroke_type,
         container_input_region_type,
         container_input_mrs_region) = st.columns(4)
    with container_input_mrs_region:
        container_input_mrs_region = st.empty()
    with st.expander('Full data tables'):
        container_results_tables = st.container()

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
    st.markdown('## Pathway inputs')
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
        ['None', 'ISDN', 'ICB', 'Nearest service', 'Ambulance service']
        )

# ----- Colourmap selection -----
cmap_names = [
    'cosmic', 'viridis', 'inferno', 'neutral'
    ]
cmap_diff_names = [
    'iceburn_r', 'seaweed', 'fusion', 'waterlily'
    ]
cmap_diff_names += [c[:-2] if c.endswith('_r') else f'{c}_r' for c in cmap_diff_names]
with container_select_cmap:
    st.markdown('### Colour schemes')
    cmap_name, cmap_diff_name = inputs.select_colour_maps(
        cmap_names, cmap_diff_names)
# If we're showing mRS scores then flip the colour maps:
if outcome_type == 'mrs_shift':
    cmap_name += '_r'
    cmap_diff_name += '_r'
    # Remove any double reverse reverse.
    if cmap_name.endswith('_r_r'):
        cmap_name = cmap_name[:-4]
    if cmap_diff_name.endswith('_r_r'):
        cmap_diff_name = cmap_diff_name[:-4]


# ----- Colour limits -----
# Load colour limits info (vmin, vmax, step_size):
dict_colours, dict_colours_diff = (
    colour_setup.load_colour_limits(outcome_type))
# User inputs for vmin and vmax with loaded values as defaults:
with container_select_vlim:
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
        f'{outcome_type_str} benefit of MSU: minimum value',
        value=dict_colours_diff['vmin'],
        help=f'Default value: {dict_colours_diff["vmin"]}',
    )
    vmax_diff = st.number_input(
        f'{outcome_type_str} benefit of MSU: maximum value',
        value=dict_colours_diff['vmax'],
        help=f'Default value: {dict_colours_diff["vmax"]}',
    )
    # Sanity checks:
    if ((vmax <= vmin) | (vmax_diff <= vmin_diff)):
        st.error(
            'Maximum value must be less than the minimum value.', icon='❗')
        st.stop()
# Overwrite default values:
dict_colours['vmin'] = vmin
dict_colours['vmax'] = vmax
dict_colours_diff['vmin'] = vmin_diff
dict_colours_diff['vmax'] = vmax_diff


# ######################################
# ########## PLOT USER INPUTS ##########
# ######################################

# ----- Timeline -----
time_dicts = timeline.build_time_dicts_muster(input_dict)
timeline_display_dict = timeline.get_timeline_display_dict()

# Setup for timeline plot.
# Leave this gap in minutes between separate chunks of pathway:
gap_between_chunks = 45
# Start each chunk at these offsets:
time_offsets = {
    'prehosp_usual_care': 0,
    'ivt_only_unit': (
        gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values())
        ),
    'mt_transfer_unit': (
        gap_between_chunks * 2.0 +
        sum(time_dicts['prehosp_usual_care'].values()) +
        sum(time_dicts['ivt_only_unit'].values())
    ),
    'ivt_mt_unit': (
        gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values())
    ),
    'msu_dispatch': 0,
    'prehosp_msu_ivt': (
        gap_between_chunks + sum(time_dicts['msu_dispatch'].values())
    ),
    'prehosp_msu_no_ivt': (
        gap_between_chunks + sum(time_dicts['msu_dispatch'].values())
    ),
    'mt_transfer_from_msu': (
        gap_between_chunks * 2.0 +
        sum(time_dicts['msu_dispatch'].values()) +
        max([
            sum(time_dicts['prehosp_msu_ivt'].values()),
            sum(time_dicts['prehosp_msu_no_ivt'].values())
            ])
    ),
}
# Find shared max time for setting same size across multiple plots
# so that 1 minute always spans the same number of pixels.
tmax = max(
    [time_offsets[k] + sum(time_dicts[k].values()) for k in time_dicts.keys()]
) + gap_between_chunks

# Separate the standard and MSU pathway data.
# Standard:
time_keys_standard = [
    'prehosp_usual_care',
    'ivt_only_unit',
    'mt_transfer_unit',
    'ivt_mt_unit',
]
time_dicts_standard = dict([(k, time_dicts[k]) for k in time_keys_standard])
time_offsets_standard = dict([(k, time_offsets[k]) for k in time_keys_standard])
# MSU:
time_keys_msu = [
    'msu_dispatch',
    'prehosp_msu_ivt',
    'prehosp_msu_no_ivt',
    'mt_transfer_from_msu',
]
time_dicts_msu = dict([(k, time_dicts[k]) for k in time_keys_msu])
time_offsets_msu = dict([(k, time_offsets[k]) for k in time_keys_msu])

# Draw the timelines:
with container_timeline_standard:
    timeline.plot_timeline(
        time_dicts_standard,
        timeline_display_dict,
        y_vals=[0.5, 1, 1, 0],
        time_offsets=time_offsets_standard,
        tmax=tmax,
        tmin=-10.0
        )
with container_timeline_msu:
    timeline.plot_timeline(
        time_dicts_msu,
        timeline_display_dict,
        y_vals=[0.5, 0, 1, 0.5],
        time_offsets=time_offsets_msu,
        tmax=tmax,
        tmin=-10.0
    )


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
    'msu': [[1, 2]],        # second map only
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1], [1, 2]]  # both maps
}
# Regions to draw
# Name of the column in the geojson that labels the shapes:
with container_unit_services_top:
    st.markdown('These maps show which services are provided by each stroke unit in England.')
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
    gdf_catchment_rhs_for_unit_map = None
else:
    if outline_name_for_unit_map == 'Nearest service':
        # Times to treatment:
        geo = calc.calculate_geography(df_unit_services).combined_data
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
    outline_names_col_for_unit_map, gdf_catchment_lhs_for_unit_map, gdf_catchment_rhs_for_unit_map = (
        calc.load_or_calculate_region_outlines(outline_name_for_unit_map, geo))
# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_catchment_lhs_for_unit_map, gdf_catchment_rhs_for_unit_map]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
        gdf['x'] = x_list
        gdf['y'] = y_list

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
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

try:
    inputs_changed = (
        (st.session_state['input_dict'] != input_dict) |
        (st.session_state['df_unit_services_on_last_run']['Use_IVT'] != df_unit_services['Use_IVT']).any() |
        (st.session_state['df_unit_services_on_last_run']['Use_MT'] != df_unit_services['Use_MT']).any() |
        (st.session_state['df_unit_services_on_last_run']['Use_MSU'] != df_unit_services['Use_MSU']).any()
    )
except KeyError:
    # First run of the app.
    inputs_changed = False

with container_rerun:
    if st.button('Calculate results'):
        st.session_state['input_dict'] = input_dict
        st.session_state['df_unit_services_on_last_run'] = df_unit_services
        st.session_state['df_unit_services_full_on_last_run'] = df_unit_services_full
        (
            st.session_state['df_lsoa'],
            st.session_state['df_mrs'],
            st.session_state['df_icb'],
            st.session_state['df_isdn'],
            st.session_state['df_nearest_ivt'],
            st.session_state['df_ambo']
        ) = main_calculations(input_dict, df_unit_services)
    else:
        if inputs_changed:
            with container_rerun:
                st.warning('Inputs have changed! The results currently being shown are for the previous set of inputs. Use the "calculate results" button to update the results.', icon='⚠️')


if 'df_lsoa' in st.session_state.keys():
    pass
else:
    # This hasn't been created yet and so the results cannot be drawn.
    st.stop()

# #########################################
# ########## RESULTS - FULL DATA ##########
# #########################################
with container_results_tables:
    results_tabs = st.tabs([
        'Results by IVT unit catchment',
        'Results by ISDN',
        'Results by ICB',
        'Results by ambulance service',
        'Full results by LSOA'
        ])

    # Set some columns to bool for nicer display:
    cols_bool = ['transfer_required', 'England']
    for col in cols_bool:
        for df in [st.session_state['df_icb'], st.session_state['df_isdn'], st.session_state['df_nearest_ivt'], st.session_state['df_ambo'], st.session_state['df_lsoa']]:
            df[col] = df[col].astype(bool)

    with results_tabs[0]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each IVT unit catchment area.'
            ]))
        st.dataframe(st.session_state['df_nearest_ivt'])

    with results_tabs[1]:
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(st.session_state['df_isdn'])

    with results_tabs[2]:
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(st.session_state['df_icb'])

    with results_tabs[3]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each ambulance service.'
            ]))
        st.dataframe(st.session_state['df_ambo'])

    with results_tabs[4]:
        st.dataframe(st.session_state['df_lsoa'])


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

# Limit the mRS data to only LSOA that benefit from an MSU,
# i.e. remove anything where the added utility of MSU is not better
# than the added utility of usual care.
d_str = 'diff_msu_minus_usual_care'

if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # This data doesn't exist so show no LSOAs.
    lsoa_to_keep = []
elif ((stroke_type == 'nlvo') & ('mt' in treatment_type)):
    # Use IVT-only data:
    c1 = f'{d_str}_{stroke_type}_ivt_{outcome_type}'
    lsoa_to_keep = st.session_state['df_lsoa'].index[(st.session_state['df_lsoa'][c1] > 0.0)]
else:
    # Look up the data normally.
    c1 = f'{d_str}_{stroke_type}_{treatment_type}_{outcome_type}'
    lsoa_to_keep = st.session_state['df_lsoa'].index[(st.session_state['df_lsoa'][c1] > 0.0)]

# mRS distributions that meet the criteria:
df_mrs_to_plot = st.session_state['df_mrs'][st.session_state['df_mrs'].index.isin(lsoa_to_keep)]

with container_mrs_dists_etc:
    st.markdown(''.join([
        'mRS distributions shown for only LSOA who would benefit ',
        'from an MSU (i.e. "added utility" for "MSU" scenario ',
        'is better than for "usual care" scenario).'
        ]))


# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)
bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]

# Which mRS distributions will be shown on the bars:
scenario_mrs = ['usual_care', 'msu']

# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.


@st.fragment
def display_mrs_dists():
    # User input:
    bar_option = st.selectbox('Region for mRS distributions', bar_options)

    mrs_lists_dict, region_selected, col_pretty = (
        mrs.setup_for_mrs_dist_bars(
            bar_option,
            stroke_type,
            treatment_type,
            stroke_type_str,
            treatment_type_str,
            st.session_state['df_lsoa'][['nearest_ivt_unit', 'nearest_ivt_unit_name']],
            df_mrs_to_plot,
            input_dict,
            scenarios=scenario_mrs
            ))

    mrs.plot_mrs_bars(
        mrs_lists_dict, title_text=f'{region_selected}<br>{col_pretty}')


with container_mrs_dists:
    display_mrs_dists()


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

# ----- Set up geodataframe -----
gdf = maps.load_lsoa_gdf()

# Merge in outcomes data:
gdf = pd.merge(
    gdf, st.session_state['df_lsoa'],
    left_on='LSOA11NM', right_on='lsoa', how='left'
    )


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
# Colour bar titles:
dict_colours['title'] = f'{outcome_type_str}'
dict_colours_diff['title'] = (
    f'{outcome_type_str}: Benefit of MSU over usual care')


# ----- Region outlines -----
if outline_name == 'None':
    outline_names_col = None
    gdf_catchment_lhs = None
    gdf_catchment_rhs = None
else:
    outline_names_col, gdf_catchment_lhs, gdf_catchment_rhs = (
        calc.load_or_calculate_region_outlines(outline_name, st.session_state['df_lsoa']))


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_catchment_lhs, gdf_catchment_rhs]
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
    'msu': [[1, 2]],        # second map only
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1], [1, 2]]  # both maps
}


# ----- Plot -----
# Display names:
subplot_titles = [
    'Usual care',
    'Benefit of MSU over usual care'
]
with container_map:
    plot_maps.plotly_many_heatmaps(
        burned_lhs,
        burned_rhs,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        outline_names_col,
        outline_name,
        traces_units,
        unit_subplot_dict,
        subplot_titles=subplot_titles,
        dict_colours=dict_colours,
        dict_colours_diff=dict_colours_diff,
        transform_dict=transform_dict,
        )
