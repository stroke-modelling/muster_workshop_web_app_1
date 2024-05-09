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
import utilities.plot_mrs_dists as mrs
# Containers:
import utilities.container_inputs as inputs


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='OPTIMIST',
    page_icon=':ambulance:',
    layout='wide'
    )

# import utilities.utils as utils
# utils.make_outline_msoa_from_lsoa()
# utils.make_outline_icbs('icb')
# utils.make_outline_icbs('isdn')
# utils.make_outline_england_wales()
# st.stop()


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
# +--------------------------+
# |     container_inputs     |
# +--------------------------+
# |  container_unit_services |
# +--------------------------+
# | container_select_outcome |
# +--------------------------+
# |  container_select_cmap   |
# +--------------------------+

container_intro = st.container()
with st.sidebar:
    container_inputs = st.container()
    container_unit_services = st.container()
container_map, container_mrs_dists = st.columns([2, 1])
# Convert the map container to empty so that the placeholder map
# is replaced once the real map is ready.
with container_map:
    container_map = st.empty()
container_map_inputs = st.container()
with container_map_inputs:
    st.markdown('__Map choices__')
    (container_input_treatment,
     container_input_stroke_type,
     container_input_region_type,
     container_input_mrs_region) = st.columns(4)
container_results_tables = st.container()
with st.sidebar:
    with st.expander('Accessibility & advanced options'):
        container_select_outcome = st.container()
        container_select_cmap = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

with container_intro:
    st.markdown('# Benefit in outcomes from redirection')

# ----- User inputs -----
with container_inputs:
    with st.form('Model setup'):
        st.markdown('### Pathway inputs')
        input_dict = inputs.select_parameters_optimist_OLD()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services(use_msu=False))
        submitted = st.form_submit_button('Submit')

with container_select_outcome:
    st.markdown('### Alternative outcome measures')
scenario_dict = inputs.select_scenario(
    containers=[
        container_select_outcome,
        container_input_treatment,
        container_input_stroke_type
        ],
    use_combo_stroke_types=True
    )
# Name of the column in the geojson that labels the shapes:
with container_input_region_type:
    outline_name = st.radio(
        'Region type to draw on maps',
        ['None', 'ISDN', 'ICB'],
        # horizontal=True
    )

# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)

bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]

# User input:
with container_input_mrs_region:
    bar_option = st.selectbox('Region for mRS distributions', bar_options)


# Colourmap selection
cmap_names = ['cosmic', 'viridis', 'inferno', 'neutral']
cmap_displays = [
    inputs.make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
    for cmap_name in cmap_names
    ]
cmap_diff_names = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
cmap_diff_displays = [
    inputs.make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
    for cmap_name in cmap_diff_names
    ]

try:
    cmap_name = st.session_state['cmap_name']
    cmap_diff_name = st.session_state['cmap_diff_name']
except KeyError:
    cmap_name = cmap_names[0]
    cmap_diff_name = cmap_diff_names[0]
cmap_ind = cmap_names.index(cmap_name)
cmap_diff_ind = cmap_diff_names.index(cmap_diff_name)

with container_select_cmap:
    st.markdown('### Colour schemes')
    cmap_name = st.radio(
        'Colour display for "usual care" map',
        cmap_names,
        captions=cmap_displays,
        index=cmap_ind,
        key='cmap_name'
    )

    cmap_diff_name = st.radio(
        'Colour display for difference map',
        cmap_diff_names,
        captions=cmap_diff_displays,
        index=cmap_diff_ind,
        key='cmap_diff_name'
    )


# ----- Setup for plots -----
# Which scenarios will be shown in the maps:
# (in this order)
scenario_types = ['drip_ship', 'diff_redirect_minus_drip_ship']

subplot_titles = [
    'Usual care',
    'Benefit of redirection over usual care'
]

legend_title = None
# legend_title = ''.join([
#     f'v: {scenario_dict["outcome_type_str"]};<br>',
#     'd: Benefit of redirection over drip-and-ship'
#     ])

cmap_titles = [
    f'{scenario_dict["outcome_type_str"]}',
    f'{scenario_dict["outcome_type_str"]}: Benefit of redirection over usual care'
    ]

# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'ivt': [[1, 1], [1, 2]],  # both maps
    'mt': [[1, 1], [1, 2]]    # both maps
}

# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    maps.plotly_blank_maps(subplot_titles, n_blank=2)

# # If the requested data is nLVO + MT, stop now.
# try:
#     stop_bool = ((scenario_dict['stroke_type'] in ['nlvo']) &
#                  (scenario_dict['treatment_type'] == 'mt'))
# except KeyError:
#     stop_bool = False
# if stop_bool:
#     with container_mrs_dists:
#         st.warning('No data for nLVO with MT.')
#         st.stop()


# ----- Main calculations -----
# Process LSOA and calculate outcomes:
df_lsoa, df_mrs = calc.calculate_outcomes(input_dict, df_unit_services)

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
df_mrs = calc.combine_results_by_redirection(df_mrs, redirect_dict, combine_mrs_dists=True)

# Make combined nLVO + LVO data in the proportions given:
prop_dict = {
    'nlvo': input_dict['prop_nlvo'],
    'lvo': input_dict['prop_lvo']
}
df_lsoa = calc.combine_results_by_occlusion_type(df_lsoa, prop_dict)
df_mrs = calc.combine_results_by_occlusion_type(df_mrs, prop_dict, combine_mrs_dists=True)

# Calculate diff - redirect minus drip-ship:
df_lsoa = calc.combine_results_by_diff(df_lsoa)
df_mrs = calc.combine_results_by_diff(df_mrs, combine_mrs_dists=True)

gdf_boundaries_msoa, df_msoa = maps.combine_geography_with_outcomes(df_lsoa)
df_icb, df_isdn, df_nearest_ivt = calc.group_results_by_region(
    df_lsoa, df_unit_services)

# ----- mRS dist results -----
# if ((scenario_dict['stroke_type'] in ['nlvo', 'combo']) & (scenario_dict['treatment_type'] == 'mt')):
#     with container_mrs_dists:
#         st.warning('No data for nLVO and MT.')
# else:
mrs_lists_dict, region_selected, col_pretty = (
    mrs.setup_for_mrs_dist_bars(
        bar_option,
        scenario_dict,
        df_lsoa[['nearest_ivt_unit', 'nearest_ivt_unit_name']],
        df_mrs,
        scenarios=['drip_ship', 'redirect']
        ))

with container_mrs_dists:
    mrs.plot_mrs_bars(
        mrs_lists_dict, title_text=f'{region_selected}<br>{col_pretty}')

# ----- Outcome results tables -----
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
        st.markdown('### Results by nearest IVT unit')
        st.markdown('Results are the mean values of all LSOA in each IVT unit catchment area.')
        st.dataframe(df_nearest_ivt)

    with results_tabs[1]:
        st.markdown('### Results by ISDN')
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(df_isdn)

    with results_tabs[2]:
        st.markdown('### Results by ICB')
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(df_icb)

    with results_tabs[3]:
        st.markdown('### Results by LSOA')
        st.dataframe(df_lsoa)

# ----- Colour setup -----
# Give the scenario dict a dummy 'scenario_type' entry
# so that the right colour map and colour limits are picked.
colour_dict = inputs.set_up_colours(
    scenario_dict | {'scenario_type': 'not diff'},
    cmap_name=cmap_name, cmap_diff_name=cmap_diff_name)
colour_diff_dict = inputs.set_up_colours(
    scenario_dict | {'scenario_type': 'diff'}, v_name='d',
    cmap_name=cmap_name, cmap_diff_name=cmap_diff_name)
# Find the names of the columns that contain the data
# that will be shown in the colour maps.
columns_colours = [
    '_'.join([
        scenario_dict['stroke_type'],
        scenario_type,
        scenario_dict['treatment_type'],
        scenario_dict['outcome_type']
    ])
    for scenario_type in scenario_types
    ]
colour_dict['column'] = columns_colours[0]
colour_diff_dict['column'] = columns_colours[1]

colour_dict['title'] = cmap_titles[0]
colour_diff_dict['title'] = cmap_titles[1]

# # Create hospital catchment areas from this MSOA geography data.
# cols = [('nearest_ivt_unit', 'scenario'), ('geometry', 'any')]
# import pandas as pd
# gdf_catchment = maps.find_geometry_ivt_catchment(pd.DataFrame(gdf_boundaries_msoa[cols]))
# # Save:
# gdf_catchment.to_file(f'data/outline_nearest_ivt.geojson')


# Left-hand subplot colours:
df_msoa_colours = maps.assign_colour_to_areas(
    df_msoa,
    colour_dict['column'],
    colour_dict['v_bands'],
    colour_dict['v_bands_str'],
    colour_dict['colour_map']
    )
# For each colour scale and data column combo,
# merge polygons that fall into the same colour band.
gdf_lhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    df_msoa_colours
    )

# Right-hand subplot colours:
df_msoa_diff_colours = maps.assign_colour_to_areas(
    df_msoa,
    colour_diff_dict['column'],
    colour_diff_dict['v_bands'],
    colour_diff_dict['v_bands_str'],
    colour_diff_dict['colour_map']
    )
gdf_rhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    df_msoa_diff_colours
    )

# Region outlines:
# Load in another gdf:
import geopandas
from shapely.validation import make_valid  # for fixing dodgy polygons



load_gdf_catchment = True
if outline_name == 'ISDN':
    outline_file = './data/outline_isdns.geojson'
    outline_names_col = 'isdn'
    outline_name = 'ISDN'  # to display
elif outline_name == 'ICB':
    outline_file = './data/outline_icbs.geojson'
    outline_names_col = 'icb'  # to display
else:
    load_gdf_catchment = False
    gdf_catchment = None
    outline_name = None
    outline_names_col = None

if load_gdf_catchment:
    gdf_catchment = geopandas.read_file(outline_file)
    # Convert to British National Grid:
    gdf_catchment = gdf_catchment.to_crs('EPSG:27700')
    # st.write(gdf_catchment['geometry'])
    # # Make geometry valid:
    # gdf_catchment['geometry'] = [
    #     make_valid(g) if g is not None else g
    #     for g in gdf_catchment['geometry'].values
    #     ]
    # Make colour transparent:
    gdf_catchment['colour'] = 'rgba(0, 0, 0, 0)'
    gdf_catchment['outline_type'] = outline_name
    # st.write(gdf_catchment['geometry'])

# Stroke unit scatter markers:
traces_units = maps.create_stroke_team_markers(df_unit_services_full)

# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_lhs, gdf_rhs]
if gdf_catchment is not None:
    gdfs_to_convert.append(gdf_catchment)
for gdf in gdfs_to_convert:
    x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
    gdf['x'] = x_list
    gdf['y'] = y_list

with container_map:
    maps.plotly_many_maps(
        gdf_lhs,
        gdf_rhs,
        gdf_catchment,
        outline_names_col,
        outline_name,
        traces_units,
        unit_subplot_dict,
        subplot_titles=subplot_titles,
        legend_title=legend_title,
        colour_dict=colour_dict,
        colour_diff_dict=colour_diff_dict
        )
