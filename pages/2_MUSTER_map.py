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
# Containers:
import utilities.container_inputs as inputs


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


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_map:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt, df_ambo = (
    main_calculations(input_dict, df_unit_services))

# ###########################################
# ########## USER INPUTS FOR PLOTS ##########
# ###########################################
# These do not change the underlying data,
# but do change what is shown in the plots.
with container_select_outcome:
    st.markdown('### Alternative outcome measures')
    outcome_type, outcome_type_str = inputs.select_outcome_type()
with container_input_treatment:
    treatment_type, treatment_type_str = inputs.select_treatment_type()
with container_input_stroke_type:
    stroke_type, stroke_type_str = (
        inputs.select_stroke_type(use_combo_stroke_types=False))

# Gather these inputs:
scenario_dict = {}
scenario_dict['outcome_type_str'] = outcome_type_str
scenario_dict['outcome_type'] = outcome_type
scenario_dict['treatment_type_str'] = treatment_type_str
scenario_dict['treatment_type'] = treatment_type
scenario_dict['stroke_type_str'] = stroke_type_str
scenario_dict['stroke_type'] = stroke_type

# Name of the column in the geojson that labels the shapes:
with container_input_region_type:
    outline_name = st.radio(
        'Region type to draw on maps',
        ['None', 'ISDN', 'ICB', 'Nearest service', 'Ambulance service']
        )

# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)
bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]
# User input moved to fragment.

# Colourmap selection
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


# #########################################
# ########## VARIABLES FOR PLOTS ##########
# #########################################
# Which scenarios will be shown in the maps:
# (in this order)
scenario_types = ['usual_care', 'diff_msu_minus_usual_care']
# Which mRS distributions will be shown on the bars:
scenario_mrs = ['usual_care', 'msu']

# Display names:
subplot_titles = [
    'Usual care',
    'Benefit of MSU over usual care'
]
cmap_titles = [
    f'{scenario_dict["outcome_type_str"]}',
    f'{scenario_dict["outcome_type_str"]}: Benefit of MSU over usual care'
    ]

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
        st.markdown('Results are the mean values of all LSOA in each ambulance service.')
        st.dataframe(df_ambo)

    with results_tabs[4]:
        st.dataframe(df_lsoa)


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

# Limit the mRS data to only LSOA that benefit from an MSU,
# i.e. remove anything where the added utility of MSU is not better
# than the added utility of usual care.
c1 = ''.join([
    'diff_msu_minus_usual_care_',
    f'{scenario_dict["stroke_type"]}_',
    f'{scenario_dict["treatment_type"]}_utility_shift'
])
try:
    lsoa_to_keep = df_lsoa.index[(df_lsoa[c1] > 0.0)]
    df_mrs_to_plot = df_mrs[df_mrs.index.isin(lsoa_to_keep)]
except KeyError:
    # Looking up data that doesn't exist, e.g. nLVO with MT.
    if ((scenario_dict['stroke_type'] == 'nlvo') & (scenario_dict['treatment_type'] == 'mt')):
        # Use no-treatment data:
        c1 = ''.join([
            'diff_msu_minus_usual_care_',
            f'{scenario_dict["stroke_type"]}_',
            f'utility_shift'
        ])
        lsoa_to_keep = []  # df_lsoa.index[(df_lsoa[c1] > 0.0)]
        df_mrs_to_plot = df_mrs[df_mrs.index.isin(lsoa_to_keep)]
    elif ((scenario_dict['stroke_type'] == 'nlvo') & ('mt' in scenario_dict['treatment_type'])):
        # Use IVT-only data:
        c1 = ''.join([
            'diff_msu_minus_usual_care_',
            f'{scenario_dict["stroke_type"]}_',
            f'ivt_utility_shift'
        ])
        lsoa_to_keep = df_lsoa.index[(df_lsoa[c1] > 0.0)]
        df_mrs_to_plot = df_mrs[df_mrs.index.isin(lsoa_to_keep)]
    else:
        # This shouldn't happen!
        pass

with container_mrs_dists_etc:
    st.markdown(''.join([
        'mRS distributions shown for only LSOA who would benefit ',
        'from an MSU (i.e. "added utility" for "MSU" scenario ',
        'is better than for "usual care" scenario).'
        ]))

# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.
@st.fragment
def display_mrs_dists():
    # User input:
    bar_option = st.selectbox('Region for mRS distributions', bar_options)

    mrs_lists_dict, region_selected, col_pretty = (
        mrs.setup_for_mrs_dist_bars(
            bar_option,
            scenario_dict,
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

# gdf_lhs, colour_dict = maps.create_colour_gdf(
#     df_lsoa,
#     scenario_dict,
#     scenario_type=scenario_types[0],
#     cmap_name=cmap_name,
#     cbar_title=cmap_titles[0],
#     )
# gdf_rhs, colour_diff_dict = maps.create_colour_gdf(
#     df_lsoa,
#     scenario_dict,
#     scenario_type=scenario_types[1],
#     cmap_diff_name=cmap_diff_name,
#     cbar_title=cmap_titles[1],
#     )



import os
import geopandas
from shapely.validation import make_valid  # for fixing dodgy polygons
import numpy as np
import rasterio
from rasterio import features
import rasterio.plot
# Load LSOA geometry:
# path_to_lsoa = os.path.join('data', 'outline_lsoa11cds.geojson')
path_to_lsoa = os.path.join('data', 'outline_lsoa11cds.geojson')
gdf = geopandas.read_file(path_to_lsoa)
# Merge in column:
gdf = pd.merge(gdf, df_lsoa,
                left_on='LSOA11NM', right_on='lsoa', how='right')
gdf.index = range(len(gdf))

# Convert to British National Grid:
gdf = gdf.to_crs('EPSG:27700')

# Make geometry valid:
gdf['geometry'] = [
    make_valid(g) if g is not None else g
    for g in gdf['geometry'].values
    ]


# Find the names of the columns that contain the data
# that will be shown in the colour maps.
if ((scenario_dict['stroke_type'] == 'nlvo') & (scenario_dict['treatment_type'] == 'mt')):
    # Use no-treatment data.
    treatment_type = ''
    vals_for_colours = [0] * len(gdf)
    vals_for_colours_diff = [0] * len(gdf)  # TEMPORARY - need to account for pop not considered for redirection (?? maybe)
else:
    if ((scenario_dict['stroke_type'] == 'nlvo') & ('mt' in scenario_dict['treatment_type'])):
        # Use IVT-only data.
        treatment_type = 'ivt'
    else:
        treatment_type = scenario_dict['treatment_type']
    column_colours = '_'.join([
        scenario_types[0],
        scenario_dict['stroke_type'],
        treatment_type,
        scenario_dict['outcome_type']
    ])
    column_colours_diff = '_'.join([
        scenario_types[1],
        scenario_dict['stroke_type'],
        treatment_type,
        scenario_dict['outcome_type']
    ])
    vals_for_colours = gdf[column_colours]
    vals_for_colours_diff = gdf[column_colours_diff]

# Code source for conversion to raster: https://gis.stackexchange.com/a/475845
# Prepare some variables
xmin, ymin, xmax, ymax = gdf.total_bounds
pixel_size = 1000
width = int(np.ceil((pixel_size + xmax - xmin) // pixel_size))
height = int(np.ceil((pixel_size + ymax - ymin) // pixel_size))
transform = rasterio.transform.from_origin(xmin, ymax, pixel_size, pixel_size)

# For maps:
transform_dict = {
    'xmin': xmin,
    'ymin': ymin,
    'xmax': xmax,
    'ymax': ymax,
    'pixel_size': pixel_size,
    'width': width,
    'height': height
}
# TEMPORARY - feed through from fixed_params
transform_dict['zmin'] = 0.0
transform_dict['zmax'] = 0.15
transform_dict['zmax_diff'] = 0.05 

im_xmax = xmin + (pixel_size * width)
im_ymax = ymin + (pixel_size * height)

# Burn geometries for left-hand map:
shapes_lhs = ((geom, value) for geom, value in zip(gdf['geometry'], vals_for_colours))
burned_lhs = features.rasterize(
    shapes=shapes_lhs,
    out_shape=(height, width),
    fill=np.NaN,
    transform=transform,
    all_touched=True
)
burned_lhs = np.flip(burned_lhs, axis=0)


# Burn geometries for right-hand map:
shapes_rhs = ((geom, value) for geom, value in zip(gdf['geometry'], vals_for_colours_diff))
burned_rhs = features.rasterize(
    shapes=shapes_rhs,
    out_shape=(height, width),
    fill=np.NaN,
    transform=transform,
    all_touched=True
)
burned_rhs = np.flip(burned_rhs, axis=0)

# Load colour info:
cmap_lhs = inputs.make_colour_list(cmap_name)
cmap_rhs = inputs.make_colour_list(cmap_diff_name)


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

# ----- Plot -----
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
        cmap_lhs=cmap_lhs,
        cmap_rhs=cmap_rhs,
        transform_dict=transform_dict,
        )
