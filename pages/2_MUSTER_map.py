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

    # Place probabilities of death into df_lsoa data
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
# ########## CONTAINER SETUP ##########
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
with st.sidebar:
    container_inputs = st.container()
    container_unit_services = st.container()
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

with container_intro:
    st.markdown('# Benefit in outcomes from Mobile Stroke Units')


# #################################
# ########## USER INPUTS ##########
# #################################

# These affect the data in all tables and all plots.

# ----- Pathway timings and stroke units -----
with container_inputs:
    with st.form('Model setup'):
        st.header('Pathway inputs')
        input_dict = inputs.select_parameters_map()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services())

        # Button for completing the form
        # (so script only re-runs once it is pressed, allows changes
        # to multiple widgets at once.)
        submitted = st.form_submit_button('Submit')


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


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_map:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt, df_ambo = (
    main_calculations(input_dict, df_unit_services))


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
        for df in [df_icb, df_isdn, df_nearest_ivt, df_ambo, df_lsoa]:
            df[col] = df[col].astype(bool)

    with results_tabs[0]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each IVT unit catchment area.'
            ]))
        st.dataframe(df_nearest_ivt)

    with results_tabs[1]:
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(df_isdn)

    with results_tabs[2]:
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(df_icb)

    with results_tabs[3]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each ambulance service.'
            ]))
        st.dataframe(df_ambo)

    with results_tabs[4]:
        st.dataframe(df_lsoa)


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

# df_mrs column names are in the format:
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
    lsoa_to_keep = df_lsoa.index[(df_lsoa[c1] > 0.0)]
else:
    # Look up the data normally.
    c1 = f'{d_str}_{stroke_type}_{treatment_type}_{outcome_type}'
    lsoa_to_keep = df_lsoa.index[(df_lsoa[c1] > 0.0)]

# mRS distributions that meet the criteria:
df_mrs_to_plot = df_mrs[df_mrs.index.isin(lsoa_to_keep)]

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
            df_lsoa[['nearest_ivt_unit', 'nearest_ivt_unit_name']],
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
    gdf, df_lsoa,
    left_on='LSOA11NM', right_on='lsoa', how='right'
    )

# ----- Find data for colours -----

# df_lsoa column names are in the format:
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
transform_dict = set_up_raster_transform(gdf, pixel_size=1000)
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
# Load colour limits info (vmin, vmax, step_size):
dict_colours, dict_colours_diff = (
    colour_setup.load_colour_limits(outcome_type))
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
        calc.load_or_calculate_region_outlines(outline_name, df_lsoa))


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
traces_units = plot_maps.create_stroke_team_markers(df_unit_services_full)
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
