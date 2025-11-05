"""Functions for plotly maps of England."""
import streamlit as st
import os
import geopandas
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import stroke_maps.load_data

from utilities.maps import gather_map_data
from utilities.utils import update_plotly_font_sizes


#MARK: Load geodata
# ###############################
# ##### LOAD GEOGRAPHY DATA #####
# ###############################
# @st.cache_data
def load_roads_gdf():
    # Load roads data:
    path_to_roads = os.path.join('data', 'major_roads_england.geojson')
    gdf_roads = geopandas.read_file(path_to_roads)
    gdf_roads = gdf_roads.set_index('roadNumber')

    # Convert Linestring to x and y coords:
    x_lists = []
    y_lists = []
    for i in gdf_roads.index:
        geo = gdf_roads.loc[i, 'geometry']
        if geo.geom_type == 'LineString':
            x, y = geo.coords.xy
            x_lists.append(list(x))
            y_lists.append(list(y))
        elif geo.geom_type == 'MultiLineString':
            x_multi = []
            y_multi = []
            for g in geo.geoms:
                x, y = g.coords.xy
                x_multi += list(x) + [None]
                y_multi += list(y) + [None]
            x_lists.append(np.array(x_multi))
            y_lists.append(np.array(y_multi))
        else:
            # ???
            x_lists.append([])
            y_lists.append([])
    gdf_roads['x'] = x_lists
    gdf_roads['y'] = y_lists
    return gdf_roads


def load_units_gdf(df_units):
    # Build geometry:
    gdf_points_units = stroke_maps.load_data.stroke_unit_coordinates()
    # Limit to units in df_units:
    mask_units = gdf_points_units.index.isin(df_units.index)
    gdf_points_units = gdf_points_units.loc[mask_units].copy()

    # Merge in services:
    gdf_points_units = pd.merge(
        gdf_points_units, df_units,
        left_index=True, right_index=True,
        how='right'
    )
    return gdf_points_units


def load_england_outline():
    # Don't replace this with stroke-maps!
    # This uses the same simplified LSOA shapes as plotted.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    gdf_ew = gdf_ew.squeeze()
    return gdf_ew


#MARK: Process geo
# ##################################
# ##### PROCESS GEOGRAPHY DATA #####
# ##################################
def convert_shapely_polys_into_xy(gdf: geopandas.GeoDataFrame):
    """
    Turn Polygon objects into two lists of x and y coordinates.

    Inputs
    ------
    gdf - geopandas.GeoDataFrame. Contains geometry.

    Returns
    -------
    x_list - list. The x-coordinates from the input polygons.
             One list entry per row in the input gdf.
    y_list - list. Same but for y-coordinates.
    """
    x_list = []
    y_list = []
    for i in gdf.index:
        geo = gdf.loc[i, 'geometry']
        try:
            geo.geom_type
            if geo.geom_type == 'Polygon':
                # Can use the data pretty much as it is.
                x, y = geo.exterior.coords.xy
                for interior in geo.interiors:
                    x_i, y_i = interior.coords.xy
                    x = list(x) + [None] + list(x_i)
                    y = list(y) + [None] + list(y_i)
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
                    for interior in poly.interiors:
                        x_i, y_i = interior.coords.xy
                        x_combo += list(x_i) + [None]
                        y_combo += list(y_i) + [None]
                x_list.append(np.array(x_combo))
                y_list.append(np.array(y_combo))
            elif geo.geom_type == 'GeometryCollection':
                # Treat this similarly to MultiPolygon but remove
                # anything that's not a polygon.
                polys = [t for t in geo.geoms
                         if t.geom_type in ['Polygon', 'MultiPolygon']]
                # Put None values between polygons.
                x_combo = []
                y_combo = []
                for t in polys:
                    if t.geom_type == 'Polygon':
                        # Can use the data pretty much as it is.
                        x, y = t.exterior.coords.xy
                        x_combo += list(x) + [None]
                        y_combo += list(y) + [None]
                        for interior in t.interiors:
                            x_i, y_i = interior.coords.xy
                            x_combo += list(x_i) + [None]
                            y_combo += list(y_i) + [None]
                    else:
                        # Multipolygon.
                        # Put None values between polygons.
                        for poly in t.geoms:
                            x, y = poly.exterior.coords.xy
                            x_combo += list(x) + [None]
                            y_combo += list(y) + [None]
                            for interior in poly.interiors:
                                x_i, y_i = interior.coords.xy
                                x_combo += list(x_i) + [None]
                                y_combo += list(y_i) + [None]
                x_list.append(np.array(x_combo))
                y_list.append(np.array(y_combo))
            else:
                raise TypeError('Geometry type error!') from None
        except AttributeError:
            # This isn't a geometry object. ???
            x_list.append([]),
            y_list.append([])
    return x_list, y_list


#MARK: Create traces
# #########################
# ##### CREATE TRACES #####
# #########################
def make_constant_map_traces(
        df_raster,
        transform_dict,
        dict_colours_pop
        ):
    """
    Make dict of plotly traces for constant map data.

    Units of British National Grid (BNG).

    TO DO? Should region outlines be pixellated to match raster arrs?
    """
    map_traces = {}
    # ----- Roads -----
    gdf_roads = load_roads_gdf()
    trace_roads = []
    for i in gdf_roads.index:
        trace_roads.append(go.Scatter(
            x=gdf_roads.loc[i, 'x'],
            y=gdf_roads.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            line_width=0.5,
            showlegend=False,
            hoverinfo='skip',
            ))
    map_traces['roads'] = trace_roads

    # ----- Country outline -----
    gdf_eng = load_england_outline()
    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    map_traces['england_outline'] = go.Scatter(
        x=gdf_eng['x'],
        y=gdf_eng['y'],
        mode='lines',
        fill="toself",
        fillcolor='rgba(0, 0, 0, 0)',
        line_color='grey',
        showlegend=False,
        hoverinfo='skip',
        )

    # ----- Region outlines -----
    region_dicts = {
        'isdn': {
            'display_name': 'ISDN',
            'file': './data/outline_isdns.geojson',
            'trace_dict_name': 'isdn_outlines',
        },
        'icb': {
            'display_name': 'ICB',
            'file':  './data/outline_icbs.geojson',
            'trace_dict_name': 'icb_outlines',
        },
        'ambo22': {
            'display_name': 'Amblance service',
            'file':  './data/outline_ambo22s.geojson',
            'trace_dict_name': 'ambo22_outlines',
        },
    }
    for n, reg_dict in region_dicts.items():
        # Convert to British National Grid:
        f = reg_dict['file']
        gdf_region = geopandas.read_file(f).to_crs('EPSG:27700')
        gdf_region['x'], gdf_region['y'] = (
            convert_shapely_polys_into_xy(gdf_region))
        # Make trace:
        trace_region = []
        for i in gdf_region.index:
            trace_region.append(go.Scatter(
                x=gdf_region.loc[i, 'x'],
                y=gdf_region.loc[i, 'y'],
                mode='lines',
                fill="toself",
                fillcolor='rgba(0, 0, 0, 0)',
                line_color='grey',
                name=reg_dict['display_name'],
                text=gdf_region.loc[i, n],
                hoverinfo="text",
                hoverlabel=dict(bgcolor='#ff4b4b'),
                ))
        # Store result:
        map_traces[reg_dict['trace_dict_name']] = trace_region

    # ----- Population density -----
    # For population map. Load in LSOA-level demographic data:
    df_demog = pd.read_csv(os.path.join('data', 'LSOA_popdens.csv'))
    arrs = gather_map_data(
        df_raster,
        transform_dict,
        df_demog,
        ['population_density'],
        _log=False
        )
    map_traces['pop'] = make_trace_heatmap(
        arrs[0], transform_dict, dict_colours_pop, name='pop')
    return map_traces


def make_shared_map_traces(
        df_unit_services,
        df_lsoa_units_times,
        df_raster,
        transform_dict
        ):
    """Make unit scatter, nearest unit CSC array."""
    map_traces = {}
    # ----- Stroke units -----
    gdf_units = load_units_gdf(df_unit_services)
    map_traces['units'] = make_units_traces(gdf_units)

    # ----- CSC regions -----
    df_lsoa_units_times = df_lsoa_units_times.copy()
    df_lsoa_units_times['nearest_csc'] = np.nan
    mask = df_lsoa_units_times['transfer_required']
    df_lsoa_units_times.loc[~mask, 'nearest_csc'] = 1

    arrs = gather_map_data(
        df_raster,
        transform_dict,
        df_lsoa_units_times,
        ['nearest_csc'],
        _log=False
        )
    colour = '#ff4b4b'
    map_traces['raster_nearest_csc'] = {}
    # Sneaky invisible marker so we can have just the colour
    # in the legend:
    map_traces['raster_nearest_csc']['trace_legend'] = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker={'color': colour, 'symbol': 'square', 'size': 10},
        name='Nearest unit has MT',
    )
    # The actual map:
    map_traces['raster_nearest_csc']['trace'] = go.Heatmap(
        z=arrs[0],
        transpose=False,
        x0=transform_dict['xmin'],
        dx=transform_dict['pixel_size'],
        y0=transform_dict['ymin'],
        dy=transform_dict['pixel_size'],
        zmin=0,
        zmax=1,
        showscale=False,
        colorscale=[[0, 'rgb(0,0,0)'], [1, colour]],
        hoverinfo='skip',
    )
    return map_traces


def make_units_traces(gdf_units):
    """
    Draw:
    + outline of England
    + stroke units with markers to show their services
    + major roads
    + LSOA nearest a CSC
    """
    # Add markers in separate traces for the sake of legend entries.
    # Pick out which stroke unit types are where in the gdf:
    mask_ivt = gdf_units['Use_IVT'] == 1
    mask_mt = gdf_units['Use_MT'] == 1
    try:
        mask_msu = gdf_units['Use_MSU'] == 1
    except KeyError:
        mask_msu = np.full(len(gdf_units), False)

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

    # Build the traces for the stroke units.
    traces = {}
    for service, s_dict in format_dict.items():
        mask = s_dict['mask']
        trace = go.Scatter(
            x=gdf_units.loc[mask, 'BNG_E'],
            y=gdf_units.loc[mask, 'BNG_N'],
            mode='markers',
            marker={
                'symbol': s_dict['marker'],
                'color': s_dict['colour'],
                'line': {'color': 'black', 'width': 1},
                'size': s_dict['size']
            },
            name=s_dict['label'],
            customdata=np.stack(
                [gdf_units.loc[mask, 'ssnap_name']],
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


def make_trace_heatmap(arr, transform_dict, dict_colours, name='name'):
    trace = go.Heatmap(
        z=arr,
        transpose=False,
        x0=transform_dict['xmin'],
        dx=transform_dict['pixel_size'],
        y0=transform_dict['ymin'],
        dy=transform_dict['pixel_size'],
        zmin=dict_colours['vmin'],
        zmax=dict_colours['vmax'],
        colorscale=dict_colours['cmap'],
        colorbar=dict(
            thickness=20,
            # tickmode='array',
            # tickvals=tick_locs,
            # ticktext=tick_names,
            # ticklabelposition='outside top'
            title=dict_colours['title'],
            title_font=dict(size=16),
            tickfont=dict(size=16),
            ),
        name=name,
        hoverinfo='skip',
    )
    return trace


#MARK: Figure setup
# ########################
# ##### FIGURE SETUP #####
# ########################
def get_map_config():
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
    return plotly_config


def england_map_setup(fig):

    # Remove repeat legend names:
    # (e.g. multiple sets of IVT unit, MT unit)
    # # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    # This makes sure that if multiple maps use the exact same
    # colours and labels, the labels only appear once in the legend.

    fig.update_layout(
        legend=dict(
            title_text='',
            bordercolor='grey',
            borderwidth=2,
            yanchor='top',
            y=1.0,
            xanchor='right',
            x=1,
        )
    )

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    return fig


#MARK: Plot figures
# ########################
# ##### PLOT FIGURES #####
# ########################
def draw_units_map(map_traces, outline_name='none'):
    fig = go.Figure()

    fig.add_trace(map_traces['raster_nearest_csc']['trace'])
    fig.add_trace(map_traces['raster_nearest_csc']['trace_legend'])
    if outline_name == 'none':
        fig.add_trace(map_traces['england_outline'])
    else:
        for t in map_traces[f'{outline_name}_outlines']:
            fig.add_trace(t)
    for r in map_traces['roads']:
        fig.add_trace(r)
    fig.add_trace(map_traces['units']['ivt'])
    fig.add_trace(map_traces['units']['mt'])

    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(
        width=500,
        height=600,
        margin_t=25,
        margin_b=0
        )
    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')
    plotly_config = get_map_config()

    st.plotly_chart(
        fig,
        width='content',
        config=plotly_config
        )


def plot_outcome_maps(
        map_traces, map_order, colour_dicts,
        outline_name='none',
        ):
    """"""
    # Map labels:
    subplot_titles = [colour_dicts[m]['title'] for m in map_order]
    fig = make_subplots(
        rows=1, cols=len(map_order),
        horizontal_spacing=0.0,
        subplot_titles=subplot_titles
        )

    cbar_len = 1.0 / len(map_order)
    for i, m in enumerate(map_order):
        fig.add_trace(map_traces[m], row=1, col=i+1)
        fig.update_traces(
            {'colorbar': {
                'orientation': 'h',
                'x': i * cbar_len,
                'y': -0.2,
                'len': cbar_len,
                'xanchor': 'left',
                'title_side': 'bottom'
                # 'xref': 'paper'
                }},
            selector={'name': m}
            )

    if outline_name == 'none':
        fig.add_trace(map_traces['england_outline'], row='all', col='all')
    else:
        for t in map_traces[f'{outline_name}_outlines']:
            fig.add_trace(t, row='all', col='all')
    for r in map_traces['roads']:
        fig.add_trace(r, row='all', col='all')
    fig.add_trace(map_traces['units']['ivt'], row='all', col='all')
    fig.add_trace(map_traces['units']['mt'], row='all', col='all')

    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=40,
        margin_b=0,
        )
    for i in range(len(map_order)):
        # Equivalent to pyplot set_aspect='equal':
        x = 'x' if i == 0 else f'x{i+1}'
        fig.update_yaxes(col=i+1, scaleanchor=x, scaleratio=1)
    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')
    plotly_config = get_map_config()

    st.plotly_chart(
        fig,
        # width='content',
        config=plotly_config
        )
