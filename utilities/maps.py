import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import geopandas
import pyproj  # for crs conversion
from shapely.validation import make_valid  # for fixing dodgy polygons
# from plotly.subplots import make_subplots
from shapely import Polygon  # for dummy polygons for legend order

# Custom functions:
import utilities.calculations as calc
import utilities.utils as utils
# For setting up maps:
from stroke_maps.geo import import_geojson, check_scenario_level


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
        # path_to_file=os.path.join('data', 'MSOA_Dec_2011_Boundaries_Super_Generalised_Clipped_BSC_EW_V3_2022_7707677027087735278.geojson')# 'MSOA_V3_reduced_simplified.geojson')
        # path_to_file=os.path.join('data', 'MSOA_V3_reduced_simplified.geojson')
        path_to_file=os.path.join('data', 'outline_msoa11cds.geojson')
        )
    crs = gdf_boundaries_msoa.crs
    # Index column: msoa11CD.
    # Always has only one unnamed column index level.
    gdf_boundaries_msoa = gdf_boundaries_msoa.reset_index()
    # gdf_boundaries_msoa = gdf_boundaries_msoa.rename(
    #     columns={'msoa11NM': 'msoa', 'msoa11CD': 'msoa_code'})
    gdf_boundaries_msoa = gdf_boundaries_msoa.rename(
        columns={'MSOA11NM': 'msoa', 'MSOA11CD': 'msoa_code'})
    gdf_boundaries_msoa = gdf_boundaries_msoa.set_index(['msoa_code'])
    gdf_boundaries_msoa = gdf_boundaries_msoa.drop('msoa', axis='columns')

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # Everything needs at least two levels: scenario and property.
    # Sometimes also a 'subtype' level.
    # Add another column level to the coordinates.
    df_msoa = df_msoa.reset_index()
    df_msoa = df_msoa.set_index('msoa_code')
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

    # for i in gdf_boundaries_msoa.index:
    #     st.write(gdf_boundaries_msoa.loc[i, col_geo])

    # for i, g in enumerate(gdf_boundaries_msoa[col_geo].values):
    #     try:
    #         if g.geom_type not in ['Polygon', 'MultiPolygon']:
    #             st.write(g.geom_type)
    #     except AttributeError:
    #         st.write(g)
    #         st.write(gdf_boundaries_msoa.iloc[i])

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


@st.cache_data
def combine_geography_with_outcomes(df_lsoa):
    # ----- MSOAs for geography -----
    df_msoa = calc.convert_lsoa_to_msoa_results(df_lsoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = _load_geometry_msoa(df_msoa)
    return gdf_boundaries_msoa


def make_dummy_gdf_for_legend(
        v_bands_str,
        legend_title,
        # subplot_title
        ):
    """
    Make a bonus gdf of the world's tiniest polygons, one of
    each colour, so that the legend has all of the colour entries
    and is always in increasing order.
    """
    gdf_bonus = pd.DataFrame()
    gdf_bonus[legend_title] = v_bands_str
    # gdf_bonus['inds'] = i*100 + np.arange(len(v_bands_str))
    # gdf_bonus['scenario'] = subplot_title
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
    gdf_bonus = geopandas.GeoDataFrame(
        gdf_bonus, geometry='geometry', crs='EPSG:4326')
    gdf_bonus = gdf_bonus.to_crs('EPSG:27700')
    return gdf_bonus


def dissolve_polygons_by_colour(
        gdf_all,
        col_col,
        v_bands,
        v_bands_str,
        combo_colour_map,
        legend_title='colour_str',
        # subplot_title
        ):

    gdf = gdf_all.copy()
    crs = gdf.crs
    gdf = gdf.reset_index()

    # Find geometry column for plot function:
    column_geometry = utils.find_multiindex_column_names(
        gdf, property=['geometry'])

    # Selected column to use for colour values:
    column_colour = utils.find_multiindex_column_names(
        gdf,
        property=[col_col],
        # scenario=[scenario_type],
        # subtype=['mean']
        )

    # Only keep the required columns:
    gdf = gdf[[column_colour, column_geometry]]
    # Only keep the 'property' subheading:
    gdf = pd.DataFrame(
        gdf.values,
        columns=['outcome', 'geometry']
    )
    gdf = geopandas.GeoDataFrame(gdf, geometry='geometry', crs=crs)

    # Has to be this CRS to prevent Picasso drawing:
    # gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))

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

    # Map the colours to the string labels:
    gdf['colour'] = gdf['colour_str'].map(combo_colour_map)

    # gdf['scenario'] = subplot_title

    # # # Simplify the polygons:
    # # For Picasso mode.
    # # Simplify geometry to 10000m accuracy
    # gdf['geometry'] = (
    #     gdf.to_crs(gdf.estimate_utm_crs()).simplify(10000).to_crs(gdf.crs)
    # )
    return gdf


def create_dummy_colour_gdf(
        colour_dicts
        ):
    """
    write me
    """
    # Combine all of the colour dictionaries for the legend
    # in the order in which they're given.
    # List of all dicts:
    combo_colour_maps = [cd['colour_map'] for cd in colour_dicts]
    # Start with only the first dict...
    combo_colour_map = combo_colour_maps[0]
    for i in range(1, len(combo_colour_maps)):
        # ... then add in later ones.
        # Repeat entries will be set to the latest value.
        combo_colour_map = combo_colour_map | combo_colour_maps[i]

    # Make a new gdf containing all combined polygons
    # for all plots:
    gdfs_to_combine = []

    # First create a dummy GeoDataFrame containing a tiny polygon
    # for each colour in the colour map. These will be rendered
    # first by plotly so that the legend entries are always
    # displayed in the same order as the v_bands lists.
    add_gap_after_legend = True
    for i, colour_dict in enumerate(colour_dicts):

        v_bands_str = colour_dict['v_bands_str']
        if add_gap_after_legend:
            name_for_gap = ' ' * (i+1)
            # Add a string that appears blank...
            v_bands_str = np.append(v_bands_str, name_for_gap)
            # ... and assign it a transparent colour:
            combo_colour_map[name_for_gap] = 'rgba(0, 0, 0, 0)'
            # This is pretty stupid but it works.
            # add_gap_after_legend = False

        gdf_bonus = make_dummy_gdf_for_legend(
            v_bands_str,
            legend_title='colour_str'
        )
        gdfs_to_combine.append(gdf_bonus)

    # Combine the separate GeoDataFrames into one.
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

    # Map the colours to the string labels:
    gdf_polys['colour'] = gdf_polys['colour_str'].map(combo_colour_map)

    return gdf_polys, combo_colour_map


def create_stroke_team_markers(df_units=None):
    from stroke_maps.utils import find_multiindex_column_names

    # Add stroke team markers.
    from stroke_maps.geo import _load_geometry_stroke_units, check_scenario_level
    if df_units is None:
        from stroke_maps.catchment import Catchment
        catchment = Catchment()
        df_units = catchment.get_unit_services()
    else:
        pass
    # Build geometry:
    df_units = check_scenario_level(df_units)
    gdf_points_units = _load_geometry_stroke_units(df_units)

    # # Convert to British National Grid.
    # The geometry column should be BNG on import, so just overwrite
    # the longitude and latitude columns that are by default long/lat.
    col_geo = find_multiindex_column_names(gdf_points_units, property=['geometry'])
    # gdf_points_units = gdf_points_units.set_crs('EPSG:27700', allow_override=True)

    # Overwrite long and lat:
    gdf_points_units[('Longitude', 'any')] = gdf_points_units[col_geo].x
    gdf_points_units[('Latitude', 'any')] = gdf_points_units[col_geo].y


    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'circle', dtype=object)
    # Update MT units:
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
    mask_ivt = gdf_points_units[col_ivt] == 1
    mask_mt = gdf_points_units[col_mt] == 1
    mask_msu = gdf_points_units[col_msu] == 1

    # Formatting for the markers:
    format_dict = {
        'ivt': {
            'label': 'IVT unit',
            'mask': mask_ivt,
            'marker': 'circle',
            'size': 6,
            'colour': 'white'
        },
        'mt': {
            'label': 'MT unit',
            'mask': mask_mt,
            'marker': 'star',
            'size': 10,
            'colour': 'white'
        },
        'msu': {
            'label': 'MSU base',
            'mask': mask_msu,
            'marker': 'square',
            'size': 13,
            'colour': 'white'
        },
    }

    # Build the traces for the stroke units...
    traces = {}
    for service, s_dict in format_dict.items():
        mask = s_dict['mask']

        trace = go.Scatter(
            x=gdf_points_units.loc[mask, ('Longitude', 'any')],
            y=gdf_points_units.loc[mask, ('Latitude', 'any')],
            mode='markers',
            marker={
                'symbol': s_dict['marker'],
                'color': s_dict['colour'],
                'line': {'color': 'black', 'width': 1},
                'size': s_dict['size']
            },
            name=s_dict['label'],
            customdata=np.stack(
                [gdf_points_units.loc[mask, ('ssnap_name', 'scenario')]],
                axis=-1
                ),
            hovertemplate=(
                '%{customdata[0]}' +
                # Need the following line to remove default "trace" bit
                # in second "extra" box:
                '<extra></extra>'
                )
        )
        traces[service] = trace
    return traces


def plotly_blank_maps(subplot_titles=[], n_blank=2):
    """
    write me
    """
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=n_blank,
        horizontal_spacing=0.02,
        subplot_titles=subplot_titles
        )

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    for i in gdf_ew.index:
        fig.add_trace(go.Scatter(
            x=gdf_ew.loc[i, 'x'],
            y=gdf_ew.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            showlegend=False,
            hoverinfo='skip',
            ), row='all', col='all'
            )

    # Add a blank trace to create space for a legend.
    # Stupid? Yes. Works? Also yes.
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker={'color': 'rgba(0,0,0,0)'},
        name=' ' * 70
    ))

    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Figure setup.
    fig.update_layout(
        width=1200,
        height=700,
        margin_t=40,
        margin_b=0
        )

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

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

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def plotly_many_maps(
        gdf_dummy,
        gdf_lhs,
        gdf_rhs,
        gdf_catchment=None,
        outline_names_col='',
        outline_name='',
        traces_units=None,
        unit_subplot_dict={},
        subplot_titles=[],
        legend_title=''
        ):
    """
    write me
    """
    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.02,
        subplot_titles=subplot_titles
        )

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    for i in gdf_dummy.index:
        fig.add_trace(go.Scatter(
            x=gdf_dummy.loc[i, 'x'],
            y=gdf_dummy.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor=gdf_dummy.loc[i, 'colour'],
            line_width=0,
            hoverinfo='skip',
            name=gdf_dummy.loc[i, 'colour_str'],
            ), row='all', col='all'
            )

    for i in gdf_lhs.index:
        fig.add_trace(go.Scatter(
            x=gdf_lhs.loc[i, 'x'],
            y=gdf_lhs.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor=gdf_lhs.loc[i, 'colour'],
            line_width=0,
            hoverinfo='skip',
            name=gdf_lhs.loc[i, 'colour_str'],
            showlegend=False
            ), row='all', col=1
            )    

    for i in gdf_rhs.index:
        fig.add_trace(go.Scatter(
            x=gdf_rhs.loc[i, 'x'],
            y=gdf_rhs.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor=gdf_rhs.loc[i, 'colour'],
            line_width=0,
            hoverinfo='skip',
            name=gdf_rhs.loc[i, 'colour_str'],
            showlegend=False
            ), row='all', col=2
            )

    if gdf_catchment is None:
        pass
    else:
        # I can't for the life of me get hovertemplate working here
        # for mysterious reasons, so just stick to "text" for hover info.
        for i in gdf_catchment.index:
            fig.add_trace(go.Scatter(
                x=gdf_catchment.loc[i, 'x'],
                y=gdf_catchment.loc[i, 'y'],
                mode='lines',
                fill="toself",
                fillcolor=gdf_catchment.loc[i, 'colour'],
                line_color='grey',
                name=gdf_catchment.loc[i, 'outline_type'],
                text=gdf_catchment.loc[i, outline_names_col],
                hoverinfo="text",
                hoverlabel=dict(bgcolor='red'),
                ), row='all', col='all'
                )

    fig.update_traces(
        hoverlabel=dict(
            bgcolor='grey',
            font_color='white'),
        selector={'name': outline_name}
    )
    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # --- Stroke unit scatter markers ---
    if len(unit_subplot_dict) > 0:
        # # Add a blank trace to put a gap in the legend.
        # Stupid? Yes. Works? Also yes.
        # Make sure the name isn't the same as any other blank name
        # already set, e.g. in combo_colour_dict, or this repeat
        # entry will be deleted later.
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            marker={'color': 'rgba(0,0,0,0)'},
            name=' ' * 10
        ))

        # Create the scatter traces for the stroke units in advance
        # and then add traces to the subplots.
        for service, grid_lists in unit_subplot_dict.items():
            for grid_list in grid_lists:
                row = grid_list[0]
                col = grid_list[1]
                fig.add_trace(traces_units[service], row=row, col=col)

    # Remove repeat legend names:
    # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    # This makes sure that if multiple maps use the exact same
    # colours and labels, the labels only appear once in the legend.

    fig.update_layout(legend_title_text=legend_title)

    # Figure setup.
    fig.update_layout(
        width=1200,
        height=700,
        margin_t=40,
        margin_b=0
        )

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

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

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


@st.cache_data
def find_geometry_ivt_catchment(gdf_boundaries_msoa):
    gdf_catchment = pd.DataFrame()
    gdf_catchment['nearest_ivt_unit'] = gdf_boundaries_msoa[
        (('nearest_ivt_unit', 'scenario'))]
    gdf_catchment['geometry'] = gdf_boundaries_msoa[(('geometry', 'any'))]
    gdf_catchment = geopandas.GeoDataFrame(gdf_catchment, geometry='geometry')
    gdf_catchment = gdf_catchment.dissolve(by='nearest_ivt_unit')
    return gdf_catchment


def convert_shapely_polys_into_xy(gdf):
    x_list = []
    y_list = []
    for i in gdf.index:
        geo = gdf.loc[i, 'geometry']
        try:
            geo.geom_type
            if geo.geom_type == 'Polygon':
                # Can use the data pretty much as it is.
                x, y = geo.exterior.coords.xy
                x_list.append(list(x))
                y_list.append(list(y))
            elif geo.geom_type == 'MultiPolygon':
                # Put None values between polygons.
                x_combo = []
                y_combo = []
                for poly in geo.geoms:
                    x, y = poly.exterior.coords.xy
                    x_combo += list(x) + [None]
                    y_combo += list(y) + [None]
                x_list.append(np.array(x_combo))
                y_list.append(np.array(y_combo))
            else:
                st.write('help', i)
        except AttributeError:
            # This isn't a geometry object. ???
            x_list.append([]),
            y_list.append([])
    return x_list, y_list
