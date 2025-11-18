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
from utilities.colour_setup import make_colour_list_for_plotly_button


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


def load_england_outline(bounds_to_clip=[]):
    # Don't replace this with stroke-maps!
    # This uses the same simplified LSOA shapes as plotted.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    if len(bounds_to_clip) < 1:
        pass
    else:
        gdf_ew = geopandas.clip(gdf_ew, bounds_to_clip)

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
def make_constant_map_traces():
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
            'display_name': 'Ambulance service',
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


def draw_cmap_buttons(fig, colour_dicts, cmaps):
    # BUTTONS TEST - https://plotly.com/python/custom-buttons/

    if len(cmaps) > 0:
        pass
    else:
        # Set up some colour options now.
        cmaps = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
        # Add the reverse option after each entry. Remove any double reverse
        # reverse _r_r. Result is flat list.
        cmaps = sum(
            [[c, (c + '_r').replace('_r_r', '')] for c in cmaps], [])
    # Colour scales dict:
    keys = list(colour_dicts.keys())
    dicts_colourscales = dict([(k, {}) for k in keys])
    for i, c in enumerate(cmaps):
        for k in keys:
            dicts_colourscales[k][c] = (
                make_colour_list_for_plotly_button(
                    c,
                    vmin=colour_dicts[k]['vmin'],
                    vmax=colour_dicts[k]['vmax']
                    ))

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'colorscale': [
                                dicts_colourscales[keys[0]][c],
                                dicts_colourscales[keys[1]][c],
                                dicts_colourscales[keys[2]][c]
                                ]},
                              {'traces': ['lhs', 'rhs', 'pop']}],
                        label=c,
                        method='restyle'
                    )
                    for c in cmaps
                ]),
                type='buttons',
                direction='right',
                pad={'r': 10, 't': 10},
                showactive=False,
                x=0.2,
                xanchor='left',
                y=-0.25,
                yanchor='top'
            ),
        ]
    )

    # Add annotations in this unusual way to prevent
    # overwriting the subplot titles.
    annotations = (
        dict(
            text='Colour scale:',
            x=0.1,
            xref='paper',
            y=-0.28,
            yref='paper',
            align='left',
            yanchor='top',
            showarrow=False
            ),  # keep this comma
    )
    fig['layout']['annotations'] += annotations
    return fig


def generate_node_coordinates(df_unit_services, all_units):
    # Generate coordinates for nodes.
    # Stroke units use their real coordinates:
    df_units_here = df_unit_services.loc[all_units]
    gdf_units = load_units_gdf(df_units_here)
    return gdf_units


def load_region_outline_here(region_type, region):
    # Place catchment units outside the bounding box of the selected region
    # _and_ all transfer units.
    # n.b. this would be hideous for large enough regions...!
    if region_type in ['isdn', 'icb', 'ambo22']:
        # Get region geography:
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
                'display_name': 'Ambulance service',
                'file':  './data/outline_ambo22s.geojson',
                'trace_dict_name': 'ambo22_outlines',
            },
        }
        reg_dict = region_dicts[region_type]
        region_display_name = reg_dict['display_name']
        # Convert to British National Grid:
        f = reg_dict['file']
        gdf_region = geopandas.read_file(f).to_crs('EPSG:27700')
        # Only keep the selected region:
        gdf_region = gdf_region.loc[gdf_region[region_type] == region]
        gdf_region['x'], gdf_region['y'] = (
            convert_shapely_polys_into_xy(gdf_region))
    else:
        # TO DO - create nearest unit geography
        gdf_region = None
        region_display_name = 'to do'
    return gdf_region, region_display_name


def set_network_map_bounds(gdf_units, gdf_region=None):
    if gdf_region is not None:
        bounds_reg = gdf_region.total_bounds
        bounds_units = gdf_units.total_bounds
        bounds = [
            min(bounds_units[0], bounds_reg[0]),
            min(bounds_units[1], bounds_reg[1]),
            max(bounds_units[2], bounds_reg[2]),
            max(bounds_units[3], bounds_reg[3]),
        ]
    else:
        bounds = gdf_units.total_bounds

    # Add breathing room to region bounding box:
    x_buffer = (bounds[2] - bounds[0]) * 0.1
    y_buffer = (bounds[3] - bounds[1]) * 0.1
    bounds = [
        bounds[0] - x_buffer,
        bounds[1] - y_buffer,
        bounds[2] + x_buffer,
        bounds[3] + y_buffer,
        ]

    return bounds, x_buffer, y_buffer


def make_coords_nearest_unit_catchment(
        gdf_units,
        df_net_u,
        bounds,
        nearest_units,
        x_buffer,
        y_buffer,
        ):
    box_centre = [0.5*(bounds[0]+bounds[2]), 0.5*(bounds[1]+bounds[3])]

    gdf = gdf_units.copy()
    # Make coordinates for each unit in the region's "nearest unit"
    # anchor. Find the angle between the centre of the region and
    # each unit.
    gdf['x_off'] = gdf['BNG_E'] - box_centre[0]
    gdf['y_off'] = gdf['BNG_N'] - box_centre[1]
    gdf['angle'] = np.arctan2(gdf['y_off'], gdf['x_off'])
    gdf['angle_deg'] = gdf['angle'] * 180.0 / np.pi

    # Limit to units in the region:
    gdf['nearest_unit'] = 'nearest_' + gdf.index.astype(str)
    gdf = gdf.loc[
        gdf['nearest_unit'].isin(nearest_units)]

    anch_top = bounds[3] + (2.0 * y_buffer)
    anch_left = bounds[0] - (2.0 * x_buffer)
    anch_bottom = bounds[1] - (2.0 * y_buffer)
    anch_right = bounds[2] + (2.0 * x_buffer)

    angle_to_top_right = np.arctan2(
        (bounds[3] - box_centre[1]), (bounds[2] - box_centre[0]))
    angle_to_top_left = np.arctan2(
        (bounds[3] - box_centre[1]), (bounds[0] - box_centre[0]))
    angle_to_bottom_left = np.arctan2(
        (bounds[1] - box_centre[1]), (bounds[0] - box_centre[0]))
    angle_to_bottom_right = np.arctan2(
        (bounds[1] - box_centre[1]), (bounds[2] - box_centre[0]))

    mask_top = (
        (gdf['angle'] >= angle_to_top_right) &
        (gdf['angle'] < angle_to_top_left)
    )
    mask_left = (
        (gdf['angle'] >= angle_to_top_left) |
        (gdf['angle'] < angle_to_bottom_left)
    )
    mask_bottom = (
        (gdf['angle'] >= angle_to_bottom_left) &
        (gdf['angle'] < angle_to_bottom_right)
    )
    mask_right = (
        (gdf['angle'] >= angle_to_bottom_right) &
        (gdf['angle'] < angle_to_top_right)
    )

    gdf.loc[mask_top, 'side'] = 'top'
    gdf.loc[mask_left, 'side'] = 'left'
    gdf.loc[mask_bottom, 'side'] = 'bottom'
    gdf.loc[mask_right, 'side'] = 'right'

    gdf.loc[mask_top, 'y_anchor'] = anch_top
    gdf.loc[mask_left, 'x_anchor'] = anch_left
    gdf.loc[mask_bottom, 'y_anchor'] = anch_bottom
    gdf.loc[mask_right, 'x_anchor'] = anch_right

    gdf.loc[mask_top, 'x_anchor'] = box_centre[0] + (
        (bounds[3] - box_centre[1]) /
        np.tan(gdf.loc[mask_top, 'angle'])
        )
    gdf.loc[mask_bottom, 'x_anchor'] = box_centre[0] + (
        (bounds[1] - box_centre[1]) /
        np.tan(gdf.loc[mask_bottom, 'angle'])
        )
    gdf.loc[mask_left, 'y_anchor'] = box_centre[1] + (
        (bounds[0] - box_centre[0]) *
        np.tan(gdf.loc[mask_left, 'angle'])
        )
    gdf.loc[mask_right, 'y_anchor'] = box_centre[1] + (
        (bounds[2] - box_centre[0]) *
        np.tan(gdf.loc[mask_right, 'angle'])
        )

    cols_to_keep = ['nearest_unit', 'side', 'x_anchor', 'y_anchor',
                    'ssnap_name', 'Use_MT']
    gdf = gdf[cols_to_keep]
    gdf = pd.merge(
        gdf.reset_index(),
        df_net_u[['first_unit', 'admissions']],
        left_on='Postcode', right_on='first_unit', how='left'
        ).set_index('Postcode').drop('first_unit', axis='columns')
    return gdf


def make_unit_catchment_raster(
        df_lsoa_units_times,
        catchment_units,
        colour_lookup,
        df_raster,
        transform_dict,
        ):
    """
    colour lookup e.g. [[0, 'rgb(0,0,0)'], [1, colour]]
    """
    # Unit catchment:
    df_lsoa_units_times = df_lsoa_units_times.copy()
    mask = df_lsoa_units_times['nearest_ivt_unit'].isin(catchment_units)
    # Set up unit --> number --> colour lookup.
    df = df_lsoa_units_times.loc[mask].copy()
    unit_dict = dict(zip(catchment_units, np.linspace(0.0, 1.0, len(catchment_units))))
    df['unit_number'] = df['nearest_ivt_unit'].map(unit_dict)
    colour_lookup = pd.DataFrame(colour_lookup)
    colour_lookup['unit_number'] = colour_lookup.index.map(unit_dict)
    colour_scale = colour_lookup[['unit_number', 'colour']].values
    colour_scale = [list(i) for i in colour_scale]

    arrs = gather_map_data(
        df_raster,
        transform_dict,
        df,
        ['unit_number'],
        _log=False
        )
    arr = arrs[0]

    # Crop the array to non-NaN values:
    mask0 = np.all(np.isnan(arr), axis=0)
    min0 = np.where(mask0 == False)[0][0]
    max0 = np.where(mask0 == False)[0][-1]
    mask1 = np.all(np.isnan(arr), axis=1)
    min1 = np.where(mask1 == False)[0][0]
    max1 = np.where(mask1 == False)[0][-1]
    arr = arr[min1:max1+1, min0:max0+1]
    # Update the transform dictionary to reflect the crop.
    # Make a copy of the transform dict:
    transform_dict_here = {}
    for k, v in transform_dict.items():
        transform_dict_here[k] = v
    # Update the coordinates of the bottom left corner:
    transform_dict_here['xmin'] = (
        transform_dict['xmin'] + transform_dict['pixel_size'] * min0)
    transform_dict_here['ymin'] = (
        transform_dict['ymin'] + transform_dict['pixel_size'] * min1)

    # The actual map:
    catch_trace = go.Heatmap(
        z=arr,
        transpose=False,
        x0=transform_dict_here['xmin'],
        dx=transform_dict_here['pixel_size'],
        y0=transform_dict_here['ymin'],
        dy=transform_dict_here['pixel_size'],
        zmin=0,
        zmax=1,
        showscale=False,
        colorscale=colour_scale,
        hoverinfo='skip',
    )
    return catch_trace


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


def draw_units_msu_map(map_traces, outline_name='none'):
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.0,
        subplot_titles=['Usual care', 'MSU available'],
        )

    fig.add_trace(map_traces['raster_nearest_csc']['trace'], row='all', col='all')
    fig.add_trace(map_traces['raster_nearest_csc']['trace_legend'], row='all', col='all')
    if outline_name == 'none':
        fig.add_trace(map_traces['england_outline'], row='all', col='all')
    else:
        for t in map_traces[f'{outline_name}_outlines']:
            fig.add_trace(t, row='all', col='all')
    for r in map_traces['roads']:
        fig.add_trace(r, row='all', col='all')
    fig.add_trace(map_traces['units']['msu'], row='all', col=2)
    fig.add_trace(map_traces['units']['ivt'], row='all', col=1)
    fig.add_trace(map_traces['units']['mt'], row='all', col='all')

    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(
        width=500,
        height=600,
        margin_t=25,
        margin_b=0
        )

    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=0.0,
        xanchor="center",
        x=0.5,
    ))
    plotly_config = get_map_config()

    st.plotly_chart(
        fig,
        width='stretch',
        config=plotly_config
        )


def plot_outcome_maps(
        map_traces, map_order, colour_dicts,
        all_cmaps, outline_name='none', show_msu_bases=False,
        title=''
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
    if show_msu_bases:
        fig.add_trace(map_traces['units']['msu'], row='all', col=2)
    fig.add_trace(map_traces['units']['ivt'], row='all', col='all')
    fig.add_trace(map_traces['units']['mt'], row='all', col='all')

    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=600,
        margin_t=80,
        margin_b=0,
        )
    fig = draw_cmap_buttons(fig, colour_dicts, all_cmaps)
    for i in range(len(map_order)):
        # Equivalent to pyplot set_aspect='equal':
        x = 'x' if i == 0 else f'x{i+1}'
        fig.update_yaxes(col=i+1, scaleanchor=x, scaleratio=1)
    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text=title)

    return fig


def plot_networks(
        df_net_u, df_net_r, df_unit_services, gdf_nearest_units,
        gdf_units, bounds, catch_trace, gdf_region=None, region_display_name=None,
        subplot_titles=[]
        ):
    fig = make_subplots(
        rows=3, cols=1,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        )

    # Border:
    fig.add_shape(
        type="rect",
        x0=bounds[0], y0=bounds[1], x1=bounds[2], y1=bounds[3],
        fillcolor='rgba(0.95, 0.95, 0.95, 1.0)',
        line=dict(color='grey', width=4,),
        layer='between',  # below other traces
        col='all', row='all',
        # zorder=-2
    )

    # ----- Country outline -----
    gdf_eng = load_england_outline(bounds)
    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
    fig.add_trace(go.Scatter(
        x=gdf_eng['x'],
        y=gdf_eng['y'],
        mode='lines',
        fill="toself",
        fillcolor='rgba(0.753, 0.753, 0.753, 1)',  # silver
        # fillcolor='rgba(0.663, 0.663, 0.663, 1)',  # darkgrey
        # fillcolor='rgba(0.573, 0.573, 0.573, 1)',  # grey
        line_color='rgba(0, 0, 0, 0)',
        showlegend=False,
        hoverinfo='skip',
        zorder=-1,
        ), row='all', col='all',
    )

    # Region catchment:
    fig.add_trace(catch_trace, row=1, col=1)

    # Selected region:
    if isinstance(gdf_region, pd.DataFrame):
        for i in gdf_region.index:
            fig.add_trace(go.Scatter(
                x=gdf_region.loc[i, 'x'],
                y=gdf_region.loc[i, 'y'],
                mode='lines',
                fill="toself",
                fillcolor='rgba(0, 0, 0, 0)',
                line_color='black',
                name=region_display_name,
                # text=gdf_region.loc[i, region_type],
                hoverinfo='skip',
                # hoverlabel=dict(bgcolor='#ff4b4b'),
                ), row='all', col='all')

    # Links between units:
    link_drawn_to_first = False
    link_drawn_to_trans = False
    for d, df_net in enumerate([df_net_u, df_net_r]):
        for i in df_net.index:
            s = df_net.loc[i]
            name_catch = df_unit_services.loc[
                df_unit_services.index == s['nearest_unit']
                .replace('nearest_', ''), 'ssnap_name'
                ].values[0]
            name_first = df_unit_services.loc[
                df_unit_services.index == s['first_unit'], 'ssnap_name'
                ].values[0]
            # Catchment to first unit:
            m = gdf_nearest_units['nearest_unit'] == s['nearest_unit']
            x_nearest = gdf_nearest_units.loc[m, 'x_anchor'].values[0]
            y_nearest = gdf_nearest_units.loc[m, 'y_anchor'].values[0]
            colour = gdf_nearest_units.loc[m, 'colour'].values[0]
            m = gdf_units.index == s['first_unit']
            x_first = gdf_units.loc[m, 'BNG_E'].values[0]
            y_first = gdf_units.loc[m, 'BNG_N'].values[0]
            a = s['admissions_catchment_to_first_unit']
            w = np.log(a)  # / 75.0
            w = 1.0 if w < 1.0 else w
            aw = w * 5 if w < 3 else w*2
            standoff = 5
            # Setup for sneaking:
            bw = w * 1.3
            baw = aw * 1.0
            bcolour = 'black'
            # Sneaky background arrow for outline effect:
            fig.add_trace(go.Scatter(
                x=[x_nearest, x_first],
                y=[y_nearest, y_first],
                mode='lines+markers',
                marker=dict(size=baw, symbol='arrow-up', angleref='previous',
                            standoff=standoff, line=dict(color='black', width=2)),
                line_color=bcolour,
                line_width=bw,
                # text=[a],
                hoverinfo='skip',
                name=None,
            ), col=1, row=d+2)
            # The actual arrow:
            fig.add_trace(go.Scatter(
                x=[x_nearest, x_first],
                y=[y_nearest, y_first],
                mode='lines+markers',
                marker=dict(size=aw, symbol='arrow-up', angleref='previous',
                            standoff=standoff),
                line_color=colour,
                line_width=w,
                # text=[a],
                hoverinfo='skip',
                name=(None if link_drawn_to_first else 'Admissions to first unit'),
            ), col=1, row=d+2)
            # Add a sneaky trace halfway along the line for a hoverlabel:
            fig.add_trace(go.Scatter(
                x=[0.5*(x_nearest+x_first)],
                y=[0.5*(y_nearest+y_first)],
                mode='markers',
                marker=dict(color='rgba(0, 0, 0, 0)'),
                # text=[a],
                customdata=np.stack(
                    [[name_catch]*2,
                     [name_first]*2,
                     [a]*2],
                    axis=-1
                    ),
                hovertemplate=(
                    '%{customdata[2]:.1f} patients from catchment area of ' +
                    '<br>' +
                    '%{customdata[0]}' +
                    '<br>' +
                    'go to' +
                    '<br>' +
                    '%{customdata[1]}.' +
                    '<br>' +
                    # Need the following line to remove default "trace" bit
                    # in second "extra" box:
                    '<extra></extra>'
                    ),
                hoverlabel=dict(bordercolor=colour),
            ), col=1, row=d+2)

        # First unit to transfer unit:
        df_net_trans = df_net.copy()
        df_net_trans = df_net_trans.drop('nearest_unit', axis='columns')
        df_net_trans = df_net_trans.groupby(
            ['first_unit', 'transfer_unit']).sum().reset_index()
        colour = 'white'  #'rgba(0.95, 0.95, 0.95, 1.0)'
        for i in df_net_trans.index:
            s = df_net_trans.loc[i]
            name_first = df_unit_services.loc[
                df_unit_services.index == s['first_unit'], 'ssnap_name'
                ].values[0]
            name_trans = df_unit_services.loc[
                df_unit_services.index == s['transfer_unit'], 'ssnap_name'
                ].values[0]
            if s['first_unit'] != s['transfer_unit']:
                m = gdf_units.index == s['first_unit']
                x_first = gdf_units.loc[m, 'BNG_E'].values[0]
                y_first = gdf_units.loc[m, 'BNG_N'].values[0]
                m = gdf_units.index == s['transfer_unit']
                x_trans = gdf_units.loc[m, 'BNG_E'].values[0]
                y_trans = gdf_units.loc[m, 'BNG_N'].values[0]
                a = s['admissions_first_unit_to_transfer']
                w = np.log(a)  #  / 75.0
                w = 1.0 if w < 1.0 else w
                aw = w * 5 if w < 3 else w*2
                # Setup for sneaking:
                bw = w * 1.3
                baw = aw * 1.0
                bcolour = 'black'
                # Sneaky background arrow for outline effect:
                fig.add_trace(go.Scatter(
                    x=[x_first, x_trans],
                    y=[y_first, y_trans],
                    mode='lines+markers',
                    marker=dict(size=baw, symbol='arrow-up', angleref='previous',
                                standoff=standoff, line=dict(color='black', width=2)),
                    line_color=bcolour,
                    line_width=bw,
                    # text=[a],
                    hoverinfo='skip',
                    name=None,
                ), col=1, row=d+2)
                # The actual arrow:
                fig.add_trace(go.Scatter(
                    x=[x_first, x_trans],
                    y=[y_first, y_trans],
                    mode='lines+markers',
                    marker=dict(size=aw, symbol='arrow-up', angleref='previous',
                                standoff=standoff),
                    line_color=colour,
                    line_width=w,
                    hoverinfo='skip',
                    name=(None if link_drawn_to_trans else 'Transfers for thrombectomy'),
                ), col=1, row=d+2)
                # Add a sneaky trace halfway along the line for a hoverlabel:
                fig.add_trace(go.Scatter(
                    x=[0.5*(x_first+x_trans)],
                    y=[0.5*(y_first+y_trans)],
                    mode='markers',
                    marker=dict(color='rgba(0, 0, 0, 0)'),
                    # text=[a],
                    customdata=np.stack(
                        [[name_first]*2,
                         [name_trans]*2,
                         [a]*2],
                        axis=-1
                        ),
                    hovertemplate=(
                        '%{customdata[2]:.2f} patients transfer from ' +
                        '<br>' +
                        '%{customdata[0]}' +
                        '<br>' +
                        'to' +
                        '<br>' +
                        '%{customdata[1]}' +
                        '<br>' +
                        'for thrombectomy.' +
                        '<br>' +
                        # Need the following line to remove default "trace" bit
                        # in second "extra" box:
                        '<extra></extra>'
                        ),
                    hoverlabel=dict(bordercolor=colour),
                ), col=1, row=d+2)

    # Stroke units:
    unit_traces = make_units_traces(gdf_units)
    fig.add_trace(go.Scatter(unit_traces['ivt']), row='all', col='all')
    fig.add_trace(go.Scatter(unit_traces['mt']), row='all', col='all')

    # Catchment anchors:
    fig.add_trace(go.Scatter(
        x=gdf_nearest_units['x_anchor'],
        y=gdf_nearest_units['y_anchor'],
        mode='markers',
        text=gdf_nearest_units.index,
        marker={
            'symbol': 'square',
            'color': gdf_nearest_units['colour'],
            'line': {'color': 'black', 'width': 1},
            'size': 25
        },
        # name=s_dict['label'],
        customdata=np.stack(
            [gdf_nearest_units['ssnap_name'],
             gdf_nearest_units['admissions']],
            axis=-1
            ),
        hovertemplate=(
            'Catchment area of<br>%{customdata[0]}' +
            '<br>' +
            '%{customdata[1]:.1f} patients' +
            # Need the following line to remove default "trace" bit
            # in second "extra" box:
            '<extra></extra>'
            ),
        hoverlabel=dict(bordercolor=gdf_nearest_units['colour']),
    ), row=[2, 3], col='all')

    fig.update_layout(hovermode='closest')

    fig = england_map_setup(fig)
    fig.update_yaxes(row=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(row=2, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(row=3, scaleanchor='x', scaleratio=1)
    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')

    # Calculate height based on aspect ratio:
    width = 500
    height = (
        (0.75 * width) +  # space for subplot gaps
        (3.0 * 0.8 * width * (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]))
    )
    # Figure setup.
    fig.update_layout(
        width=width,
        height=height,
        margin_t=25,
        margin_b=0,
        margin_l=0,
        margin_r=0,
        )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
    ))
    plotly_config = get_map_config()
    st.plotly_chart(fig, width='stretch', config=plotly_config)
