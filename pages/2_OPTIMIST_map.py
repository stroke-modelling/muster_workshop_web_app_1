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

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.plot_mrs_dists as mrs
# Containers:
import utilities.container_inputs as inputs


@st.cache_data
def main_calculations(input_dict, df_unit_services):
    # Process LSOA and calculate outcomes:
    df_lsoa, df_mrs = calc.calculate_outcomes(
        input_dict, df_unit_services, use_msu=False, use_mothership=True)

    # Remove the MSU data:
    cols_to_remove = [c for c in df_lsoa.columns if 'msu' in c]
    df_lsoa = df_lsoa.drop(cols_to_remove, axis='columns')
    # Current setup means that MSU data is calculated with default
    # values even if we don't explicitly give it any parameters.

    # Extra calculations for redirection:
    # Combine drip-and-ship and mothership results in proportions given:
    redirect_dict = {
        'sensitivity': input_dict['sensitivity'],
        'specificity': input_dict['specificity'],
    }
    df_lsoa = calc.combine_results_by_redirection(df_lsoa, redirect_dict)
    df_mrs = calc.combine_results_by_redirection(
        df_mrs, redirect_dict, combine_mrs_dists=True)

    # Make combined nLVO + LVO data in the proportions given:
    prop_dict = {
        'nlvo': input_dict['prop_nlvo'],
        'lvo': input_dict['prop_lvo']
    }
    df_lsoa = calc.combine_results_by_occlusion_type(df_lsoa, prop_dict)
    df_mrs = calc.combine_results_by_occlusion_type(
        df_mrs, prop_dict, combine_mrs_dists=True)

    # Calculate diff - redirect minus drip-ship:
    df_lsoa = calc.combine_results_by_diff(df_lsoa)
    df_mrs = calc.combine_results_by_diff(df_mrs, combine_mrs_dists=True)

    df_icb, df_isdn, df_nearest_ivt = calc.group_results_by_region(
        df_lsoa, df_unit_services)

    return df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='OPTIMIST',
    page_icon=':ambulance:',
    layout='wide'
    )


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
# utils.make_outline_lsoa_limit_to_england()
# # utils.make_outline_msoa_from_lsoa()
# # utils.make_outline_icbs('icb')
# # utils.make_outline_icbs('isdn')
# # utils.make_outline_england_wales()
# st.stop()


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
container_map, container_mrs_dists = st.columns([2, 1])
# Convert the map container to empty so that the placeholder map
# is replaced once the real map is ready.
with container_map:
    container_map = st.empty()
container_map_inputs = st.container(border=True)
with container_map_inputs:
    st.markdown('__Plot options__')
    (container_input_treatment,
     container_input_stroke_type,
     container_input_region_type,
     container_input_mrs_region) = st.columns(4)
with container_input_mrs_region:
    container_input_mrs_region = st.empty()
# Convert mRS dists to empty so that re-running a fragment replaces
# the bars rather than displays the new plot in addition.
with container_mrs_dists:
    container_mrs_dists = st.empty()
with st.expander('Full data tables'):
    container_results_tables = st.container()
with st.sidebar:
    with st.expander('Accessibility & advanced options'):
        container_select_outcome = st.container()
        container_select_cmap = st.container()

with container_intro:
    st.markdown('# Benefit in outcomes from redirection')


# #################################
# ########## USER INPUTS ##########
# #################################

# These affect the data in all tables and all plots.
with container_inputs:
    with st.form('Model setup'):
        st.markdown('### Pathway inputs')
        pathway_dict = inputs.select_parameters_pathway_optimist()

        st.markdown('### Population inputs')
        population_dict = inputs.select_parameters_population_optimist()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services(use_msu=False))

        # Button for completing the form
        # (so script only re-runs once it is pressed, allows changes
        # to multiple widgets at once.)
        submitted = st.form_submit_button('Submit')

# Combine the two input dicts:
input_dict = pathway_dict | population_dict

# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_map:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt = (
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
        inputs.select_stroke_type(use_combo_stroke_types=True))

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
    outline_name = st.radio('Region type to draw on maps',
                            ['None', 'ISDN', 'ICB', 'Nearest service'])

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
scenario_types = ['drip_ship', 'diff_redirect_minus_drip_ship']
# Which mRS distributions will be shown on the bars:
scenario_mrs = ['drip_ship', 'redirect']

# Display names:
subplot_titles = [
    'Usual care',
    'Benefit of redirection over usual care'
]
cmap_titles = [
    f'{scenario_dict["outcome_type_str"]}',
    ''.join([f'{scenario_dict["outcome_type_str"]}: ',
             'Benefit of redirection over usual care'])
    ]

# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'ivt': [[1, 1]],          # left map only
    'mt': [[1, 1], [1, 2]]    # both maps
}


# #########################################
# ########## RESULTS - FULL DATA ##########
# #########################################
with container_results_tables:
    results_tabs = st.tabs([
        'Results by IVT unit catchment',
        'Results by ISDN',
        'Results by ICB',
        'Full results by LSOA'
        ])

    # Set some columns to bool for nicer display:
    cols_bool = ['transfer_required', 'England']
    for col in cols_bool:
        for df in [df_icb, df_isdn, df_nearest_ivt, df_lsoa]:
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
        st.dataframe(df_lsoa)


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.
@st.experimental_fragment
def display_mrs_dists():
    # User input:
    with container_input_mrs_region:
        bar_option = st.selectbox('Region for mRS distributions', bar_options)

    mrs_lists_dict, region_selected, col_pretty = (
        mrs.setup_for_mrs_dist_bars(
            bar_option,
            scenario_dict,
            df_lsoa[['nearest_ivt_unit', 'nearest_ivt_unit_name']],
            df_mrs,
            scenarios=scenario_mrs
            ))

    with container_mrs_dists:
        mrs.plot_mrs_bars(
            mrs_lists_dict, title_text=f'{region_selected}<br>{col_pretty}')


display_mrs_dists()


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

gdf_lhs, colour_dict = maps.create_colour_gdf(
    df_lsoa,
    scenario_dict,
    scenario_type=scenario_types[0],
    cmap_name=cmap_name,
    cbar_title=cmap_titles[0],
    )
gdf_rhs, colour_diff_dict = maps.create_colour_gdf(
    df_lsoa,
    scenario_dict,
    scenario_type=scenario_types[1],
    cmap_diff_name=cmap_diff_name,
    cbar_title=cmap_titles[1],
    )


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
gdfs_to_convert = [gdf_lhs, gdf_rhs, gdf_catchment_lhs, gdf_catchment_rhs]
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
    plot_maps.plotly_many_maps(
        gdf_lhs,
        gdf_rhs,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        outline_names_col,
        outline_name,
        traces_units,
        unit_subplot_dict,
        subplot_titles=subplot_titles,
        colour_dict=colour_dict,
        colour_diff_dict=colour_diff_dict
        )
