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

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
# Containers:
import utilities.container_inputs as inputs


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )

# Make containers:
# +-----------------------------+
# |       container_intro       |
# +-----------------------------+
# |        container_map        |
# +-----------------------------+
# |    container_map_inputs     |
# +-----------------------------+
# |  container_results_tables   |
# +-----------------------------+
# |  container_select_outcome   |
# +-----------------------------+
container_intro = st.container()
with st.sidebar:
    container_inputs = st.container()
    container_unit_services = st.container()
container_map = st.empty()
container_map_inputs = st.container()
container_results_tables = st.container()
container_select_outcome = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

with container_intro:
    st.markdown('# Benefit in outcomes from Mobile Stroke Units')

# ----- User inputs -----
with container_inputs:
    with st.form('Model setup'):
        st.header('Pathway inputs')
        input_dict = inputs.select_parameters_map()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services())
        submitted = st.form_submit_button('Submit')

with container_map_inputs:
    cols = st.columns(2)
with container_select_outcome:
    st.markdown('### Alternative outcome measure for map')
    st.markdown('Try these if you dare.')
    scenario_dict = inputs.select_scenario(containers=[container_select_outcome] + cols)



# ----- Setup for plots -----
# Which scenarios will be shown in the maps:
# (in this order)
scenario_types = ['drip_ship', 'diff_msu_minus_drip_ship']

subplot_titles = [
    'Drip-and-ship',
    'Benefit of MSU over drip-and-ship'
]
# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    maps.plotly_blank_maps(subplot_titles, n_blank=2)


legend_title = None
# legend_title = ''.join([
#     f'v: {scenario_dict["outcome_type_str"]};<br>',
#     'd: Benefit of redirection over drip-and-ship'
#     ])

cmap_titles = [
    f'{scenario_dict["outcome_type_str"]}',
    f'{scenario_dict["outcome_type_str"]}: Benefit of MSU over drip-and-ship'
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


# If the requested data is nLVO + MT, stop now.
try:
    stop_bool = ((scenario_dict['stroke_type'] in ['nlvo', 'combo']) &
                 ('mt' in scenario_dict['treatment_type']))
except KeyError:
    stop_bool = False
if stop_bool:
    st.warning('No data for nLVO with MT.')
    st.stop()

# ----- Main calculations -----
# Process LSOA and calculate outcomes:
df_lsoa = calc.calculate_outcomes(input_dict, df_unit_services)
gdf_boundaries_msoa = maps.combine_geography_with_outcomes(df_lsoa)
df_icb, df_isdn, df_nearest_ivt = calc.group_results_by_region(
    df_lsoa, df_unit_services)


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
    scenario_dict | {'scenario_type': 'not diff'})
colour_diff_dict = inputs.set_up_colours(
    scenario_dict | {'scenario_type': 'diff'}, v_name='d')
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

# Left-hand subplot colours:
# For each colour scale and data column combo,
# merge polygons that fall into the same colour band.
gdf_lhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    colour_dict['column'],
    colour_dict['v_bands'],
    colour_dict['v_bands_str'],
    colour_dict['colour_map']
    )

# Right-hand subplot colours:
gdf_rhs = maps.dissolve_polygons_by_colour(
    gdf_boundaries_msoa,
    colour_diff_dict['column'],
    colour_diff_dict['v_bands'],
    colour_diff_dict['v_bands_str'],
    colour_diff_dict['colour_map']
    )

# Region outlines:
# Load in another gdf:
import geopandas
from shapely.validation import make_valid  # for fixing dodgy polygons

# Name of the column in the geojson that labels the shapes:
with container_map_inputs:
    outline_name = st.radio(
        'Shapes for outlines',
        ['None', 'ISDN', 'ICB']
    )

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

# st.write(gdf_catchment)

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
