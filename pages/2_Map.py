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
import plotly.graph_objs as go
import plotly.express as px
import os
import geopandas
import pyproj  # for crs conversion
from shapely.validation import make_valid  # for fixing dodgy polygons
# from plotly.subplots import make_subplots
from shapely import Polygon  # for dummy polygons for legend order

# For setting up maps:
from stroke_maps.geo import import_geojson, check_scenario_level

# Custom functions:
from utilities.fixed_params import page_setup
from utilities.plot_timeline import build_data_for_timeline, draw_timeline
import utilities.utils as utils
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


@st.cache_data
def main_calculations(input_dict):
    # Run the outcomes with the selected pathway:
    df_lsoa = results.make_outcomes(input_dict)

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    df_msoa = inputs.convert_lsoa_to_msoa_results(df_lsoa)

    st.markdown('### Results by MSOA')
    st.write(df_msoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = _load_geometry_msoa(df_msoa)
    return gdf_boundaries_msoa


@st.cache_data
def _import_geojson(*args, **kwargs):
    """Wrapper for stroke-maps import_geojson so cache_data used."""
    return import_geojson(*args, **kwargs)


@st.cache_data
def _load_geometry_msoa(df_msoa):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.

    Inputs
    ------
    df_msoa - pd.DataFrame. msoa info.

    Returns
    -------
    gdf_boundaries_msoa - GeoDataFrame. msoa info and geometry.
    """

    # All msoa shapes:
    gdf_boundaries_msoa = _import_geojson(
        'MSOA11NM',
        path_to_file=os.path.join('data', 'MSOA_V3_reduced_simplified.geojson')
        )
    crs = gdf_boundaries_msoa.crs
    # Index column: msoa11CD.
    # Always has only one unnamed column index level.
    gdf_boundaries_msoa = gdf_boundaries_msoa.reset_index()
    # gdf_boundaries_msoa = gdf_boundaries_msoa.rename(
    #     columns={'msoa11NM': 'msoa', 'msoa11CD': 'msoa_code'})
    gdf_boundaries_msoa = gdf_boundaries_msoa.rename(
        columns={'MSOA11NM': 'msoa', 'MSOA11CD': 'msoa_code'})
    gdf_boundaries_msoa = gdf_boundaries_msoa.set_index(['msoa', 'msoa_code'])

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # Everything needs at least two levels: scenario and property.
    # Sometimes also a 'subtype' level.
    # Add another column level to the coordinates.
    col_level_names = df_msoa.columns.names
    cols_gdf_boundaries_msoa = [
        gdf_boundaries_msoa.columns,                 # property
        ['any'] * len(gdf_boundaries_msoa.columns),  # scenario
    ]
    if 'subtype' in col_level_names:
        cols_gdf_boundaries_msoa.append(
            [''] * len(gdf_boundaries_msoa.columns))

    # Make all data to be combined have the same column levels.
    # Geometry:
    gdf_boundaries_msoa = pd.DataFrame(
        gdf_boundaries_msoa.values,
        index=gdf_boundaries_msoa.index,
        columns=cols_gdf_boundaries_msoa
    )

    # ----- Create final data -----
    # Merge together all of the DataFrames.
    gdf_boundaries_msoa = pd.merge(
        gdf_boundaries_msoa, df_msoa,
        left_index=True, right_index=True, how='right'
    )
    # Name the column levels:
    gdf_boundaries_msoa.columns = (
        gdf_boundaries_msoa.columns.set_names(col_level_names))

    # Sort the results by scenario:
    gdf_boundaries_msoa = gdf_boundaries_msoa.sort_index(
        axis='columns', level='scenario')

    # Convert to GeoDataFrame:
    col_geo = utils.find_multiindex_column_names(
        gdf_boundaries_msoa, property=['geometry'])
    # Make geometry valid:
    gdf_boundaries_msoa[col_geo] = [
        make_valid(g) if g is not None else g
        for g in gdf_boundaries_msoa[col_geo].values
        ]
    gdf_boundaries_msoa = geopandas.GeoDataFrame(
        gdf_boundaries_msoa,
        geometry=col_geo,
        crs=crs
        )
    return gdf_boundaries_msoa


def plotly_blank_maps(subplot_titles=[]):
    """
    Show some blank England & Wales outlines while real map loads.
    """
    path_to_file = os.path.join('data', 'outline_england_wales.geojson')
    gdf = geopandas.read_file(path_to_file)
    # Has to be this CRS to prevent Picasso drawing:
    gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))

    # Blank name to make nothing show up in the legend:
    label = '.' + ' '*40 + '.'
    gdf[' '] = label

    # Make a new gdf containing all combined polygons
    # for all plots:
    # gdf_polys
    gdfs_to_combine = []

    if len(subplot_titles) == 0:
        subplot_titles = range(3)

    for i in range(3):
        gdf_here = gdf.copy()
        gdf_here['scenario'] = subplot_titles[i]
        gdfs_to_combine.append(gdf_here)

    gdf = pd.concat(gdfs_to_combine, axis='rows')

    # Begin plotting.
    fig = px.choropleth(
        gdf,
        locations=gdf.index,
        geojson=gdf.geometry.__geo_interface__,
        color=gdf[' '],
        color_discrete_map={label: 'rgba(0, 0, 0, 0)'},
        facet_col='scenario',
        category_orders={'scenario': subplot_titles}
        )

    fig.update_layout(
        width=1300,
        height=500
        )
    fig.update_layout(margin_t=20)
    fig.update_layout(margin_b=0)
    fig.update_geos(
        scope='world',
        projection=go.layout.geo.Projection(type='airy'),
        fitbounds='locations',
        visible=False,
        bgcolor='rgba(0,0,0,0)'  # transparent background
        )
    fig.update_traces(marker_line_color='grey')

    st.plotly_chart(fig)


def plotly_many_maps(
        gdf_all,
        columns_colour,
        column_geometry,
        v_bands,
        v_bands_str,
        colour_map,
        subplot_titles=[],  # plot titles
        legend_title='Outcome'
        ):

    if len(subplot_titles) == 0:
        subplot_titles = columns_colour

    # Make a new gdf containing all combined polygons
    # for all plots:
    # gdf_polys
    gdfs_to_combine = []

    for i, col_col in enumerate(columns_colour):
        gdf = gdf_all.copy()
        crs = gdf.crs
        gdf = gdf.reset_index()

        # Find geometry column for plot function:
        column_geometry = utils.find_multiindex_column_names(
            gdf, property=['geometry'])

        # Selected column to use for colour values:
        column_colour = utils.find_multiindex_column_names(
            gdf_boundaries_msoa,
            property=[col_col],
            # scenario=[scenario_type],
            # subtype=['mean']
            )

        # st.write(gdf.columns)

        # Only keep the required columns:
        gdf = gdf[[column_colour, column_geometry]]
        # Only keep the 'property' subheading:
        gdf = pd.DataFrame(
            gdf.values,
            columns=['outcome', 'geometry']
        )
        gdf = geopandas.GeoDataFrame(gdf, geometry='geometry', crs=crs)

        # Has to be this CRS to prevent Picasso drawing:
        gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))

        # Group by outcome band.
        # Only group by non-NaN values:
        mask = ~pd.isna(gdf['outcome'])
        inds = np.digitize(gdf.loc[mask, 'outcome'], v_bands)
        labels = v_bands_str[inds]
        # Flag NaN values:
        gdf.loc[mask, legend_title] = labels
        gdf.loc[~mask, legend_title] = 'rubbish'
        # Drop outcome column:
        gdf = gdf.drop('outcome', axis='columns')
        # Dissolve by shared outcome value:
        gdf = gdf.dissolve(by=legend_title, sort=False)
        gdf = gdf.reset_index()
        # Remove the NaN polygon:
        gdf = gdf[gdf[legend_title] != 'rubbish']

        # Add back in the inds:
        df_inds = pd.DataFrame(
            np.array([np.arange(len(v_bands_str)), v_bands_str]).T,
            columns=['inds', legend_title]
            )
        gdf = pd.merge(gdf, df_inds, left_on=legend_title, right_on=legend_title)
        # # Sort the dataframe for the sake of the legend order:
        # gdf = gdf.sort_values(by='inds')
        gdf['inds'] = gdf['inds'].astype(int)

        gdf['scenario'] = subplot_titles[i]

        # # # Simplify the polygons:
        # # For Picasso mode.
        # # Simplify geometry to 10000m accuracy
        # gdf['geometry'] = (
        #     gdf.to_crs(gdf.estimate_utm_crs()).simplify(10000).to_crs(gdf.crs)
        # )

        gdfs_to_combine.append(gdf)

    # Make a bonus gdf of the world's tiniest polygons, one of
    # each colour, so that the legend has all of the colour entries
    # and is always in increasing order.
    gdf_bonus = pd.DataFrame()
    gdf_bonus[legend_title] = v_bands_str
    gdf_bonus['inds'] = range(len(v_bands_str))
    gdf_bonus['scenario'] = subplot_titles[0]
    # Make a tiny polygon around these coordinates on the Isle of Man
    # (coordinates should be included on our England & Wales map
    # but not expecting anyone to closely look at this area).
    # Coords: 54.147729, -4.471397
    bonus_long = 54.147729
    bonus_lat = -4.471397
    poly = Polygon([
        [bonus_lat, bonus_long],
        [bonus_lat+1e-5, bonus_long],
        [bonus_lat+1e-5, bonus_long+1e-5],
        [bonus_lat, bonus_long+1e-5],
        ])
    gdf_bonus['geometry'] = poly
    gdf_bonus['inds'] = gdf_bonus['inds'].astype(int)
    gdf_bonus = geopandas.GeoDataFrame(
        gdf_bonus, geometry='geometry', crs='EPSG:4326')

    gdfs_to_combine.append(gdf_bonus)

    gdf_polys = pd.concat(gdfs_to_combine, axis='rows')
    # Make a new index column:
    gdf_polys['index'] = range(len(gdf_polys))
    gdf_polys = gdf_polys.set_index('index')
    # Otherwise the px.choropleth line below will only draw
    # the first polygon with each index value, not the one
    # that actually belongs to the scenario in facet_col.

    # Drop any 'none' geometry:
    gdf_polys = gdf_polys.dropna(axis='rows', subset=['geometry'])
    # If any polygon is None then all polygons in that facet_col
    # will fail to be displayed.
    # None seems to happen when there are very few (only one? or zero?)
    # polygons in that outcome band. Maybe a rounding error?

    # Sort the dataframe for the sake of the legend order:
    gdf_polys = gdf_polys.sort_values(by=['inds'])

    # Begin plotting.
    fig = px.choropleth(
        gdf_polys,
        locations=gdf_polys.index,
        geojson=gdf_polys.geometry.__geo_interface__,
        color=gdf_polys[legend_title],
        color_discrete_map=colour_map,
        facet_col='scenario',
        # Which order the plots should appear in:
        category_orders={'scenario': subplot_titles},
        # facet_col_wrap=3  # How many subplots to get on a single row
        )
    # Remove hover labels for choropleth:
    fig.update_traces(
        hovertemplate=None,
        hoverinfo='skip'
        )
    # Remove outlines of contours:
    fig.update_traces(marker_line_width=0)

    fig.update_geos(
        scope='world',
        projection=go.layout.geo.Projection(type='airy'),
        fitbounds='locations',
        visible=False,
        bgcolor='rgba(0,0,0,0)'  # transparent background
        )
    fig.update_layout(
        width=1300,
        height=500
        )

    fig.update_layout(margin_t=20)
    fig.update_layout(margin_b=0)
    # fig.update_layout(margin_t=0)
    # fig.update_layout(margin_t=0)

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    # Add a blank trace to put a gap in the legend.
    # Stupid? Yes. Works? Also yes.
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None], marker={'color': 'rgba(0,0,0,0)'}, name=''
    ))

    # Add stroke team markers.
    from stroke_maps.catchment import Catchment
    from stroke_maps.geo import _load_geometry_stroke_units, check_scenario_level
    catchment = Catchment()
    df_units = catchment.get_unit_services()
    # Build geometry:
    df_units = check_scenario_level(df_units)
    gdf_points_units = _load_geometry_stroke_units(df_units)

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'circle', dtype=object)
    # Update MT units:
    from stroke_maps.utils import find_multiindex_column_names
    col_use_mt = find_multiindex_column_names(
        gdf_points_units, property=['use_mt'])
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = 'square'
    # Store in the DataFrame:
    gdf_points_units[('marker', 'any')] = markers


    # Add markers in separate traces for the sake of legend entries.
    # Pick out which stroke unit types are where in the gdf:
    col_ivt = ('use_ivt', 'scenario')
    col_mt = ('use_mt', 'scenario')
    col_msu = ('use_msu', 'scenario')
    mask_ivt = (
        (gdf_points_units[col_ivt] == 1) &
        (gdf_points_units[col_mt] == 0) &
        (gdf_points_units[col_msu] == 0)
    )
    mask_mt = (
        (gdf_points_units[col_mt] == 1)
    )
    mask_msu = (
        (gdf_points_units[col_msu] == 1)
    )
    # Formatting for the markers:
    labels = ['IVT unit', 'MT unit', 'MSU base']
    masks = [mask_ivt, mask_mt, mask_msu]
    markers = ['circle', 'star', 'square']
    sizes = [6, 10, 13]
    colours = ['white', 'white', 'white']

    # Build the traces for the stroke units...
    traces = []
    for i in range(3):
        mask = masks[i]
        trace = go.Scattergeo(
            lon=gdf_points_units.loc[mask, ('Longitude', 'any')],
            lat=gdf_points_units.loc[mask, ('Latitude', 'any')],
            marker={
                'symbol': markers[i],
                'color': colours[i],
                'line': {'color': 'black', 'width': 1},
                'size': sizes[i]
            },
            name=labels[i],
            customdata=np.stack([gdf_points_units.loc[mask, ('ssnap_name', 'scenario')]], axis=-1),
            hovertemplate=(
                '%{customdata[0]}' +
                # Need the following line to remove default "trace" bit
                # in second "extra" box:
                '<extra></extra>'
                )
        )
        traces.append(trace)

    # ... and THEN add traces to the subplots.
    # MSU bases:
    fig.add_trace(traces[2], row=1, col=3)
    # IVT units:
    fig.add_trace(traces[0], row=1, col=1)
    fig.add_trace(traces[0], row=1, col=3)
    # MT units:
    fig.add_trace(traces[1], row=1, col=1)
    fig.add_trace(traces[1], row=1, col=2)
    fig.add_trace(traces[1], row=1, col=3)


    # Remove repeat legend names:
    # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    with container_map:
        st.plotly_chart(fig)


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
# +-----------------------+
# |    container_intro    |
# +-----------------------+
# |     container_map     |
# +-----------------------+
# | container_shared_data |
# +-----------------------+
# | container_params_here |
# +-----------------------+
# |  container_outcomes   |
# +-----------------------+
container_intro = st.container()
container_map_inputs = st.container()
container_map = st.empty()
container_shared_data = st.container()
container_params_here = st.container()
container_outcomes = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

# Draw the input selection boxes in this function:
with st.sidebar:
    st.header('Pathway inputs')
    input_dict = inputs.select_parameters()

with container_map_inputs:
    cols = st.columns(6)  # make more columns than needed to space closer
    scenario_dict = inputs.select_scenario(cols)

# If the requested data is nLVO + MT, stop now.
stop_bool = (
    (scenario_dict['stroke_type'] == 'nlvo') &
    ('mt' in scenario_dict['treatment_type'])
)
if stop_bool:
    st.warning('No data for nLVO with MT.')
    st.stop()

scenario_types = ['drip_ship', 'mothership', 'msu']
# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    plotly_blank_maps(scenario_types)

colour_dict = inputs.set_up_colours(scenario_dict | {'scenario_type': 'not diff'})

gdf_boundaries_msoa = main_calculations(input_dict)

# Find geometry column for plot function:
col_geo = utils.find_multiindex_column_names(
    gdf_boundaries_msoa, property=['geometry'])


columns_colours = [
    '_'.join([
        scenario_dict['stroke_type'],
        scenario_type,
        scenario_dict['treatment_type'],
        scenario_dict['outcome_type']
    ])
    for scenario_type in scenario_types
    ]

plotly_many_maps(
    gdf_boundaries_msoa,
    columns_colours,
    column_geometry=col_geo,
    v_bands=colour_dict['v_bands'],
    v_bands_str=colour_dict['v_bands_str'],
    colour_map=colour_dict['colour_map'],
    subplot_titles=scenario_types,
    legend_title=f'v: {scenario_dict["outcome_type_str"]}'
)

st.stop()

# Build up the times to treatment in the different scenarios:


# Pick out results for this scenario ID:
results_dict = inputs.find_scenario_results(scenario_id)

# Separate the fixed parameters
# (currently in results data for some reason):
fixed_keys = [
    'nearest_ivt_time',
    'nearest_mt_time',
    'transfer_time',
    'nearest_msu_time',
    'Admissions',
    'England',
    'nlvo_no_treatment_mrs_0-2',
    'nlvo_no_treatment_utility',
    'lvo_no_treatment_mrs_0-2',
    'lvo_no_treatment_utility',
    ]
fixed_dict = dict([(k, results_dict[k]) for k in fixed_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys())
                     if k not in fixed_keys])

# Separate times and outcomes:
time_keys = [
    'drip_ship_ivt_time',
    'drip_ship_mt_time',
    'mothership_ivt_time',
    'mothership_mt_time',
    'msu_ivt_time',
    'msu_mt_time'
]
treatment_time_dict = dict([(k, results_dict[k]) for k in time_keys])
results_dict = dict([(k, results_dict[k]) for k in list(results_dict.keys())
                     if k not in time_keys])

# Gather cumulative times and nicer-formatted cumulative time labels:
(times_dicts, times_cum_dicts, times_cum_label_dicts
 ) = build_data_for_timeline(fixed_dict | treatment_time_dict | input_dict)

# Convert results to a DataFrame with multiple column headers.
# Column header names: occlusion, pathway, treatment, outcome.
df_results = utils.convert_results_dict_to_multiindex_df(results_dict)

# ########################################
# ########## WRITE TO STREAMLIT ##########
# ########################################
with container_intro:
    # Title:
    st.markdown('# MUSTER')

    st.markdown('''
    This model shows predicted outcomes for non-large vessel occlusion (nLVO) and large vessel occlusion 
    stroke. Outcomes are calculated for 34,000 small areas (LSOAs) across England based on expected 
    travel times, and other timing parameters chosen by the slider bars on the right.

    More detail may be found on estimation of stroke outcomes [here](https://samuel-book.github.io/stroke_outcome/intro.html). 
    The reported outcomes are for treated patients (they do not include patients unsuitable for treatment, 
    haemorrhagic strokes, or mimics)

    Three pathways are modelled, through to thrombectomy (note: thrombectomy is only applied to large 
    vessel occlusions; non-large vessel occlusions are treated with thrombolysis only). The three pathways are:

    1) *Drip-and-ship*: All patients are taken to their closest emergency stroke unit, all of which 
    provide thrombolysis. For patients who receive thrombectomy there is a transfer to a thrombectomy-capable 
    if the patient has first attended a hopsital that provides thrombolysis only.

    2) *Mothership*: All patients are taken to a comprehensive stroke centre that can provide both 
    thrombolysis and thrombectomy.

    3) *Mobile stroke unit (MSU)*: MSUs are dispatched, from comprehensive stroke centres, to stroke patients. 
    Head scans and thrombolysis are provided on-scene, where the patient is. For patients who have been 
    treated with thrombolysis or considered suitable for thrombectomy, the MSU takes the patient to the 
    comprehensive stroke centre. Where a patient does not receive thrombolysis, and is not considered 
    a candidate for thrombectomy, the MSU becomes available for another stroke patient, and a standard 
    ambulance conveys the patient to the closest emergency stroke unit. In this particular model there 
    are no capacity limits for the MSU, and it is assumed all strokes are triaged correctly with the 
    emergency call - the model shows outcomes if all patients were seen by a MSU.
    ''')

    st.image('./pages/images/stroke_treatment.jpg')

with container_timeline:
    st.markdown('### Timeline for this scenario:')
    draw_timeline(times_cum_dicts, times_cum_label_dicts)

with container_shared_data:
    st.markdown('## Fixed values')
    st.markdown('Baseline outcomes')

    cols = [
        'nlvo_no_treatment_mrs_0-2',
        'nlvo_no_treatment_utility',
        'lvo_no_treatment_mrs_0-2',
        'lvo_no_treatment_utility',
        ]
    df_outcomes = pd.Series(dict([(k, fixed_dict[k]) for k in cols]))
    style_dict = results.make_column_style_dict(
        df_outcomes.index, format='%.3f')
    st.dataframe(
        pd.DataFrame(df_outcomes).T,
        column_config=style_dict,
        hide_index=True
        )

    # Travel times:
    st.markdown('Average travel times (minutes) to closest units')
    cols = [
        'nearest_ivt_time',
        'nearest_mt_time',
        'transfer_time',
        'nearest_msu_time'
        ]
    df_travel = pd.Series(dict([(k, fixed_dict[k]) for k in cols]))
    style_dict = results.make_column_style_dict(
        df_travel.index, format='%d')
    st.dataframe(
        pd.DataFrame(df_travel).T,
        column_config=style_dict,
        hide_index=True
        )

with container_params_here:
    st.markdown('## This scenario')

    st.markdown('### Treatment times ###')

    st.markdown('Average times (minutes) to treatment')
    # Times to treatment:
    columns = ['drip_ship', 'mothership', 'msu']
    index = ['ivt', 'mt']
    table = [[0, 0, 0], [0, 0, 0]]
    for c, col in enumerate(columns):
        for i, ind in enumerate(index):
            key = f'{col}_{ind}_time'
            table[i][c] = int(round(treatment_time_dict[key], 0))
    df_times = pd.DataFrame(table, columns=columns, index=index)
    style_dict = results.make_column_style_dict(
        df_times.columns, format='%d')
    st.dataframe(
        df_times,
        column_config=style_dict,
        # hide_index=True
        )

    # MSU bits:

    st.markdown('### MSU Use ###')

    st.markdown('MSU use time (minutes) per patient')
    cols = ['msu_occupied_treatment', 'msu_occupied_no_treatment']
    # Extra pd.DataFrame() here otherwise streamlit sees it's a Series
    # and overrides the style dict.
    dict_msu = dict(zip(cols, [results_dict[k] for k in cols]))
    df_msu = pd.DataFrame(pd.Series(dict_msu))
    style_dict = results.make_column_style_dict(
        df_msu.index, format='%d')
    st.dataframe(
        df_msu.T,
        column_config=style_dict,
        # hide_index=True
        )

with container_outcomes:
    st.markdown('''
    ### Outcomes ###

    * **mrs_0-2**: Proportion patients modified Rankin Scale 0-2 (higher is better)
    * **mrs_shift**: Average shift in modified Rankin Scale (negative is better)
    * **utility**: Average utility (higher is better)
    * **utility_shift**: Average improvement in (higher is better)
    ''')

    # User inputs for how to display the data:
    group_by = st.radio(
        'Group results by:',
        ['Treatment type', 'Outcome type']
        )

    if group_by == 'Treatment type':
        for stroke_type in ['ivt', 'mt', 'ivt_mt']:
            df_here = utils.take_subset_by_column_level_values(
                df_results.copy(), treatment=[stroke_type])
            df_here = utils.convert_row_to_table(
                df_here, ['occlusion', 'outcome'])
            st.markdown(f'### {stroke_type}')
            style_dict = results.make_column_style_dict(
                df_here.columns, format='%.3f')
            st.dataframe(
                df_here,
                column_config=style_dict
                )
    else:
        for outcome in ['mrs_shift', 'mrs_0-2', 'utility', 'utility_shift']:
            df_here = utils.take_subset_by_column_level_values(
                df_results, outcome=[outcome])
            df_here = utils.convert_row_to_table(
                df_here, ['occlusion', 'treatment'])
            st.markdown(f'### {outcome}')
            style_dict = results.make_column_style_dict(
                df_here.columns, format='%.3f')
            st.dataframe(
                df_here,
                column_config=style_dict
                )

# ----- The end! -----
