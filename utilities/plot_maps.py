"""Functions for plotly maps of England."""
import streamlit as st
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from itertools import groupby

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
    """
    Load roads GeoDataFrame and prepare it for scatter plot.

    Returns
    -------
    gdf_roads - gpd.GeoDataFrame. Contains x and y coordinates for
                major roads in England.
    """
    # Load roads data:
    path_to_roads = os.path.join('data', 'major_roads_england.geojson')
    gdf_roads = gpd.read_file(path_to_roads)
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
    """
    Load stroke unit coordinates.

    Inputs
    ------
    df_units - pd.DataFrame. All stroke units whether they provide
               IVT, MT, and MSU for MUSTER.

    Returns
    -------
    gdf_points_units - gpd.GeoDataFrame. Stroke unit coordinates.
    """
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


def load_england_outline(bounds_to_clip: list = []):
    """
    Load England outline GeoDataFrame and prepare for scatter.

    Inputs
    ------
    bounds_to_clip - list. List of bounds xmin, ymin, xmax, ymax
                     to limit the area of the outline to.

    Returns
    -------
    gdf_ew - gpd.GeoDataFrame. Coordinates for the outline of England
             ready for plotting with scatter.
    """
    # Don't replace this with stroke-maps!
    # This uses the same simplified LSOA shapes as plotted.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = gpd.read_file(path_to_file)

    if len(bounds_to_clip) < 1:
        pass
    else:
        gdf_ew = gpd.clip(gdf_ew, bounds_to_clip)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    gdf_ew = gdf_ew.squeeze()
    return gdf_ew


#MARK: Process geo
# ##################################
# ##### PROCESS GEOGRAPHY DATA #####
# ##################################
def convert_shapely_polys_into_xy(gdf: gpd.GeoDataFrame):
    """
    Turn Polygon objects into two lists of x and y coordinates.

    Inputs
    ------
    gdf - gpd.GeoDataFrame. Contains geometry.

    Returns
    -------
    x_list - list. The x-coordinates from the input polygons.
             One list entry per row in the input gdf.
    y_list - list. Same but for y-coordinates.
    """
    def pick_out_coords(poly):
        """Helper function for taking coords from poly."""
        # Put None values between polygons.
        x, y = poly.exterior.coords.xy
        x_combo = list(x) + [None]
        y_combo = list(y) + [None]
        for interior in poly.interiors:
            x_i, y_i = interior.coords.xy
            x_combo += list(x_i) + [None]
            y_combo += list(y_i) + [None]
        return x_combo, y_combo

    x_list = []
    y_list = []
    for i in gdf.index:
        geo = gdf.loc[i, 'geometry']
        try:
            geo.geom_type
            if geo.geom_type == 'Polygon':
                # Can use the data pretty much as it is.
                x_, y_ = pick_out_coords(geo)
                x_list.append(list(x_))
                y_list.append(list(y_))
            elif geo.geom_type == 'MultiPolygon':
                x_combo = []
                y_combo = []
                for poly in geo.geoms:
                    x_, y_ = pick_out_coords(poly)
                    x_combo += x_
                    y_combo += y_
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
                        x_, y_ = pick_out_coords(t)
                        x_combo += x_
                        y_combo += y_
                    else:
                        # Multipolygon.
                        for poly in t.geoms:
                            x_, y_ = pick_out_coords(poly)
                            x_combo += x_
                            y_combo += y_
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
@st.cache_data
def make_constant_map_traces():
    """
    Make dict of plotly traces for constant map data.

    Units of British National Grid (BNG).

    TO DO? Should region outlines be pixellated to match raster arrs?

    Returns
    -------
    map_traces - dict. Contains plotly traces for major roads, England
                 outline, and fixed region outlines.
    """
    map_traces = {}
    # ----- Roads -----
    gdf_roads = load_roads_gdf()
    # Gather all roads into one long set of coordinates.
    # There are already None separating the end of one road from the
    # start of the next.
    x_roads = np.concatenate(np.array(gdf_roads['x'].values).flatten())
    y_roads = np.concatenate(np.array(gdf_roads['y'].values).flatten())

    trace_roads = go.Scatter(
        x=x_roads,
        y=y_roads,
        mode='lines',
        fill="toself",
        fillcolor='rgba(0, 0, 0, 0)',
        line_color='grey',
        line_width=0.5,
        showlegend=False,
        hoverinfo='skip',
        )
    map_traces['roads'] = trace_roads

    # ----- Country outline -----
    gdf_eng = load_england_outline()
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
        gdf_region = gpd.read_file(f).to_crs('EPSG:27700')
        gdf_region['x'], gdf_region['y'] = (
            convert_shapely_polys_into_xy(gdf_region))
        # Make trace:
        trace_region = []
        # Add each row of the dataframe separately.
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
        df_unit_services: pd.DataFrame,
        df_lsoa_units_times: pd.DataFrame,
        df_raster: pd.DataFrame,
        transform_dict: dict
        ):
    """
    Make traces that are used often and depend on user inputs.

    Traces here: stroke unit scatter, map of LSOA whose nearest unit
    provides MT, map of MT unit catchment, map of IVT unit catchment.

    Inputs
    ------
    df_unit_services    - pd.DataFrame. Stroke units and whether they
                          provide IVT, MT, and MSU for Muster.
    df_lsoa_units_times - pd.DataFrame. LSOA-level allocated units and
                          travel times.
    df_raster           - pd.DataFrame. Contains pixel indices and the
                          values that they will be set to in the array.
    transform_dict      - dict. Contains size and bound information for
                          the raster image.

    Returns
    -------
    map_traces       - dict. Contains the plotly traces.
    df_unit_services - pd.DataFrame. Same as input with addition of
                       assigned colour index and colours.
    """
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
        name='Nearest IVT unit has MT',
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

    # ----- Nearest MT units -----
    # Clear colours if they already exist.
    # "Colour index" is which element of the colour list was chosen.
    # Could potentially be used later if a new list of colours were
    # given.
    try:
        df_unit_services = df_unit_services.drop(
            ['colour_ind', 'colour'], axis='columns')
    except KeyError:
        pass
    map_traces['raster_nearest_mt_unit'], _, df_unit_services = (
        make_unit_catchment_raster(
            df_lsoa_units_times,
            df_unit_services,
            df_raster,
            transform_dict,
            nearest_unit_column='nearest_mt_unit',
            redo_transform=False,
            create_colour_scale=True,
            )
    )

    # ----- Nearest IVT units -----
    map_traces['raster_nearest_ivt_unit'], _, df_unit_services = (
        make_unit_catchment_raster(
            df_lsoa_units_times,
            df_unit_services,  # this is returned with colours
            df_raster,
            transform_dict,
            nearest_unit_column='nearest_ivt_unit',
            redo_transform=False,
            create_colour_scale=True,
            )
    )

    return map_traces, df_unit_services


def make_units_traces(gdf_units: gpd.GeoDataFrame):
    """
    Create plotly traces to show stroke unit locations and services.

    If the "Use_MSU" column is in gdf_units then a trace will be made
    for the MSU base locations.

    Inputs
    ------
    gdf_units - gpd.GeoDataFrame. Stroke unit coordinates and which
                services they provide.

    Returns
    -------
    traces - dict. Contains go.Scatter traces for the different types
             of stroke unit.
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


def make_trace_heatmap(
        arr: np.array,
        transform_dict: dict,
        dict_colours: dict,
        name: str = 'name'
        ):
    """
    Generic heatmap trace creation.

    Inputs
    ------
    arr            - np.array. Data for the heatmap.
    transform_dict - dict. How to transform the heatmap from pixels
                     to British National Grid coordinates.
    dict_colours   - dict. Contains the colour map and min and max
                     values for the colour scale.
    name           - str. Trace name.

    Returns
    -------
    trace - go.Heatmap trace.
    """
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


def make_unit_catchment_raster(
        df_lsoa_units_times: pd.DataFrame,
        df_unit_services: pd.DataFrame,
        df_raster: pd.DataFrame,
        transform_dict: dict,
        nearest_unit_column: str = 'nearest_ivt_unit',
        redo_transform: bool = True,
        create_colour_scale: bool = False,
        ):
    """
    Make a heatmap of the stroke unit catchment areas.

    Adjacent catchment areas have different colours assigned to them
    so that they can be picked out more easily on a big map.
    Current setup allows MT units to pick from a different selection
    of colours than IVT-only units.

    colour lookup e.g. [[0, 'rgb(0,0,0)'], [1, colour]]

    In this function, local work is done in a temporary "df_units"
    dataframe. Any results to keep are then copied back into the
    original "df_unit_services" object.

    Inputs
    ------
    df_lsoa_units_times - pd.DataFrame. LSOA-level allocated units and
                          travel times.
    df_unit_services    - pd.DataFrame. Stroke units and whether they
                          provide IVT, MT, and MSU for Muster.
    df_raster           - pd.DataFrame. Contains pixel indices and the
                          values that they will be set to in the array.
    transform_dict      - dict. Contains size and bound information for
                          the raster image.
    nearest_unit_column - str. Name of the column to define catchment
                          areas by. e.g. nearest IVT unit or MT unit.
    redo_transform      - bool. If the heatmap covers a considerably
                          smaller area than a full England heatmap,
                          then make a new transform_dict for that area.
    create_colour_scale - bool. Whether to assign new colours to these
                          areas.

    Returns
    -------
    catch_trace         - go.Heatmap. Each unit has its catchment area
                          with a different value assigned to it.
    transform_dict_here - dict. The transform dict for this Heatmap
                          trace. Could cover a much smaller area than
                          the input transform_dict for England.
    df_unit_services    - input unit services with added colour scale
                          and colour columns if these were found here.
    """
    # Check whether any units have already been assigned colours:
    if 'colour_ind' in df_unit_services.columns:
        pass
    else:
        # Set all units to have some placeholder value with dtype int.
        # Could be that not all units will receive a colour on the
        # first pass of this function, but can't leave them as NaN
        # or the dtype won't be int.
        df_unit_services['colour_ind'] = -1  # dtype int

    # If necessary, limit the stroke units considered:
    if nearest_unit_column == 'nearest_mt_unit':
        m = (df_unit_services['Use_MT'] == 1)
        catchment_units = df_unit_services.loc[m].index.values
        df_units = df_unit_services.loc[m].copy()
    else:
        catchment_units = df_unit_services.index.values
        df_units = df_unit_services.copy()

    # Set up unit --> number --> colour lookup.
    # The raster array prefers to work with numbers rather than strings.
    # These unit numbers will be used in the map array.
    unit_number_column = 'unit_number'
    df_units[unit_number_column] = np.round(
        np.linspace(0.0, 1.0, len(df_units)), 3)
    # Make a copy in the actual df.
    # It will be deleted again at the end of this function.
    df_unit_services = df_unit_services.merge(
        df_units[unit_number_column],
        left_index=True, right_index=True, how='left'
        )

    # Limit to LSOA caught by the given units:
    df_lsoa_units_times = df_lsoa_units_times.copy()
    mask = df_lsoa_units_times[nearest_unit_column].isin(catchment_units)
    df = df_lsoa_units_times.loc[mask].copy()

    # Gather LSOA and their catchment units:
    df = pd.merge(
        df, df_units.reset_index()[['Postcode', unit_number_column]],
        left_on=nearest_unit_column, right_on='Postcode', how='left'
        )
    # Create raster for maps:
    arrs = gather_map_data(
        df_raster,
        transform_dict,
        df,
        [unit_number_column],
        _log=False
        )
    arr = arrs[0]

    # Make a copy of the transform dict:
    transform_dict_here = {}
    for k, v in transform_dict.items():
        transform_dict_here[k] = v
    if redo_transform:
        # Update the data array and the transform dictionary.
        # Crop array to valid area and make new transform dict:
        arr, transform_dict_here = make_new_transform(
            arr, transform_dict_here)

    # If requested, assign colours to the units so that no two units
    # whose catchment areas border each other have the same colour.
    if create_colour_scale:
        df_unit_services = assign_colours_to_units(
            df_units, df_unit_services, arr, unit_number_column)

    # Pick out which unit has each colour.
    # Set up unit --> number --> colour lookup.
    mask_colours = df_unit_services['colour'].notna()
    colour_scale = df_unit_services.loc[
        mask_colours, [unit_number_column, 'colour']
        ].copy().sort_values(unit_number_column).values
    colour_scale = [list(i) for i in colour_scale]
    # colour_scale has an entry for each stroke unit here.
    # These (unit number, colour) tuples in colour_scale are used
    # to tell plotly which colour to give each pixel.

    # Draw the actual map:
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
    # Delete temporary data:
    df_unit_services = df_unit_services.drop(
        unit_number_column, axis='columns')
    return catch_trace, transform_dict_here, df_unit_services


def make_new_transform(arr: np.array, transform_dict_here: dict):
    """
    Crop arr to the valid data and update transform dict to match.

    For example, if the array only has data for Devon, then instead
    of storing mostly null data for the rest of England, remove the
    unused elements and keep only a small portion of the array.
    Then update the transform dict so that the width, height, corner
    coordinates etc. can tell plotly how to show this smaller array
    in British National Grid coordinates.

    Inputs
    ------
    arr                 - np.array.
    transform_dict_here - dict. Transform dict to be updated. Initial
                          values should be those for the whole of
                          England.

    Returns
    -------
    arr                 - np.array. A 2D grid of only the valid pixels
                          from the input array.
    transform_dict_here - dict. Updated transform grid to match smaller
                          data array.
    """
    height_before = arr.shape[0]  # width is arr.shape[1]

    # Crop the array to non-NaN values:
    mask0 = np.all(np.isnan(arr), axis=0)
    min0 = np.where(mask0 == False)[0][0]
    max0 = np.where(mask0 == False)[0][-1]
    mask1 = np.all(np.isnan(arr), axis=1)
    min1 = np.where(mask1 == False)[0][0]
    max1 = np.where(mask1 == False)[0][-1]
    arr = arr[min1:max1+1, min0:max0+1]

    # Update transform dict.
    # New image width/height in pixels:
    transform_dict_here['width'] = arr.shape[1]
    transform_dict_here['height'] = arr.shape[0]
    # Reference coordinates are xmin, ymax.
    # Update the corner coordinates:
    transform_dict_here['xmin'] = (
        transform_dict_here['xmin'] +
        transform_dict_here['pixel_size'] * min0
        )
    transform_dict_here['ymax'] = (
        transform_dict_here['ymax'] -
        transform_dict_here['pixel_size'] * (height_before - (max1 + 1))
        )
    # Far corner in terms of pixels:
    transform_dict_here['im_xmax'] = (
        transform_dict_here['xmin'] +
        transform_dict_here['pixel_size'] * transform_dict_here['width']
        )
    transform_dict_here['im_ymin'] = (
        transform_dict_here['ymax'] -
        transform_dict_here['pixel_size'] * transform_dict_here['height']
        )
    # Remove xmax and ymax because we now only have pixel-scale
    # limits:
    transform_dict_here['xmax'] = transform_dict_here['im_xmax']
    transform_dict_here['ymin'] = transform_dict_here['im_ymin']
    return arr, transform_dict_here


def assign_colours_to_units(
        df_units: pd.DataFrame,
        df_unit_services: pd.DataFrame,
        arr: np.array,
        unit_number_column: str
        ):
    """
    Assign colours so bordering unit catchment areas are different.

    Find which stroke units have catchment areas that border each other
    and then make sure that those units are assigned different colours.
    When assigning new colours, start with the unit with the most
    neighbours and work down to the least.

    The colours are first assigned as a number (index). Then the
    actual colours are picked out from a list using those indices.

    Inputs
    ------
    df_units           - pd.DataFrame. Contains only the units being
                         assigned colours here.
    df_unit_services   - pd.DataFrame. Full data for all stroke units.
    arr                - np.array. Catchment map array. The values are
                         numbers that can be matched to stroke units
                         using the unit_number_column in the two dfs.
    unit_number_column - str. Numbers assigned to the stroke units for
                         use in the maps.

    Returns
    -------
    df_unit_services - pd.DataFrame. The input dataframe with updated
                       columns for colour index and colour strings.
    """
    # Check which regions border each other.
    # This df_pairs contains a row and a column for each stroke unit.
    # Initally all values are 0.
    # When two units border each other, the intersection of the rows
    # and columns are set to 1.
    df_pairs = pd.DataFrame(
        np.zeros((len(df_units), len(df_units))),
        columns=df_units[unit_number_column],
        index=df_units[unit_number_column],
        )
    for row in arr:
        # From https://stackoverflow.com/a/5738933
        vals_order = [key for key, _group in groupby(row[~np.isnan(row)])]
        for i in range(len(vals_order))[:-1]:
            df_pairs.loc[vals_order[i], vals_order[i+1]] = 1
            df_pairs.loc[vals_order[i+1], vals_order[i]] = 1

    # Convert unit numbers to postcodes.
    # This isn't necessary for assigning colours but makes checking
    # the working much easier.
    # Dict of unit number to unit postcode:
    dict_number_unit = (
        df_units.reset_index().set_index(unit_number_column)
        ['Postcode'].to_dict()
    )
    # Convert pairs df to postcode lookup:
    df_pairs = df_pairs.rename(
        columns=dict_number_unit, index=dict_number_unit)

    # Check if any units already have colours assigned to them:
    try:
        units_with_colours = list(
            df_units[df_units['colour'].notna()].index.values)
    except KeyError:
        units_with_colours = []
    # Sort columns from units that already have colours and then
    # from most to fewest neighbours:
    n_neighbours = df_pairs.sum(axis='rows').sort_values(ascending=False)
    unit_order = (
        units_with_colours +
        [u for u in n_neighbours.index if u not in units_with_colours]
    )
    df_pairs = df_pairs[unit_order]

    # Keep track of which units are allowed to be assigned each colour
    # and which ones cannot because their neighbour uses that colour.
    # Initially allow all colours for all units (df_colours_allowed is
    # 1 everywhere) then disallow colours (set to 0).
    # Allow more colours than we'll likely need.
    colour_options = range(10)
    df_colours_allowed = pd.DataFrame(
        np.ones((len(colour_options), len(df_pairs))),
        columns=df_pairs.columns,
        index=colour_options
    )

    for unit in df_pairs.columns:
        # Pick out all colour options for this unit:
        s = df_colours_allowed[unit]
        # Only keep the colours that are still allowed:
        colours_allowed = s[s > 0]
        # If this unit already has a colour, pick it out now.
        # Otherwise select the first available colour from the
        # allowed list.
        if unit in units_with_colours:
            colour = df_units.loc[unit, 'colour_ind']
        else:
            colour = colours_allowed.index[0]
        # Set this unit to only be allowed this colour:
        other_colours = [c for c in colour_options if c != colour]
        df_colours_allowed.loc[other_colours, unit] = 0
        # Update the allowed colours for its neighbours.
        # Do not let these neighbours have the same colour as here.
        neighbours = df_pairs[df_pairs[unit] > 0].index.values
        df_colours_allowed.loc[colour, neighbours] = 0

    # Pick out the colour index assigned to each unit:
    # colour_scale = df_unit_services[[unit_number_column]].copy()
    # From now on update the original df_unit_services dataframe,
    # not the temporary df_units dataframe.
    for unit in df_units.index:
        if unit not in units_with_colours:
            ind = df_colours_allowed[
                df_colours_allowed[unit] == 1].index.values[0]
            df_unit_services.loc[unit, 'colour_ind'] = int(ind)
    # Use these indexes to pick out the colour for each unit.
    # Setup for picking:
    colours_dict = {
        'Use_MT': [
            'red', 'firebrick', 'darkorange', 'lightcoral',
            'crimson', 'indianred', 'darksalmon', 'darkred',
            ],
        'Use_IVT': [
            'deepskyblue', 'dodgerblue', 'lightblue',
            'mediumblue', 'royalblue', 'powderblue',
            'skyblue', 'slateblue', 'steelblue',
            'cornflowerblue', 'lightskyblue', 'navy', 'cyan', 'blue',
            ],
    }
    masks_dict = {
        'Use_MT': df_unit_services['Use_MT'] == 1,
        'Use_IVT': df_unit_services['Use_MT'] != 1,
    }
    # Duplicate colours if necessary (shouldn't be!):
    while (
        df_unit_services
        .loc[masks_dict['Use_MT'], 'colour_ind'].max() + 1
            ) > len(colours_dict['Use_MT']):
        colours_dict['Use_MT'] += colours_dict['Use_MT']
    while (
        df_unit_services
        .loc[masks_dict['Use_IVT'], 'colour_ind'].max() + 1
            ) > len(colours_dict['Use_IVT']):
        colours_dict['Use_IVT'] += colours_dict['Use_IVT']

    # Assign colours to units:
    for t in ['Use_MT', 'Use_IVT']:
        m = masks_dict[t]
        c = colours_dict[t]
        max_ind = df_unit_services.loc[m, 'colour_ind'].max()
        if max_ind == -1:
            # No units here.
            pass
        else:
            for i in range(max_ind+1):
                mask = m & (df_unit_services['colour_ind'] == i)
                df_unit_services.loc[mask, 'colour'] = c[i]
    return df_unit_services


#MARK: Figure setup
# ########################
# ##### FIGURE SETUP #####
# ########################
def get_map_config():
    """Get dict of standard plotly chart config options."""
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


def england_map_setup(fig: go.Figure):
    """Standard plotly chart setup for maps of England."""
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

    # Legend location and format:
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


def draw_cmap_buttons(fig: go.Figure, colour_dicts: dict, cmaps: list):
    """
    Add restyle buttons to plotly figure for updating heatmap colours.

    Inputs
    ------
    fig           - go.Figure. A plotly chart with three heatmap traces
                    named 'lhs', 'rhs', and 'pop' (for left-hand-side,
                    right-hand-side, population maps).
    colours_dicts - dict. One key per map, each value is a dict
                    containing min/max colour limits for the map.
    cmaps         - list. List of colour maps to choose from.

    Returns
    -------
    fig - go.Figure. The input fig.
    """
    # BUTTONS TEST - https://plotly.com/python/custom-buttons/

    if len(cmaps) > 0:
        pass
    else:
        # Set up some colour options now.
        cmaps = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
        # Add the reverse option after each entry. Remove any double
        # reverse reverse _r_r. Result is flat list.
        cmaps = sum(
            [[c, (c + '_r').replace('_r_r', '')] for c in cmaps], [])
    # Make a new colour scales dict:
    keys = list(colour_dicts.keys())
    dicts_colourscales = dict([(k, {}) for k in keys])
    # Place colours in the new dict:
    for i, c in enumerate(cmaps):
        for k in keys:
            dicts_colourscales[k][c] = (
                make_colour_list_for_plotly_button(
                    c,
                    vmin=colour_dicts[k]['vmin'],
                    vmax=colour_dicts[k]['vmax']
                    ))

    # Set up buttons:
    buttons = list([
        dict(
            args=[{'colorscale': [dicts_colourscales[keys[0]][c],
                                  dicts_colourscales[keys[1]][c],
                                  dicts_colourscales[keys[2]][c]]},
                  {'traces': ['lhs', 'rhs', 'pop']}],
            label=c,
            method='restyle'
        )
        for c in cmaps
    ])
    # Draw buttons:
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
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


def generate_node_coordinates(
        df_unit_services: pd.DataFrame, units_to_show: list):
    """Wrapper for load_units_gdf for unit subset for flow maps."""
    # Generate coordinates for nodes.
    # Stroke units use their real coordinates:
    df_units_here = df_unit_services.loc[units_to_show]
    gdf_units = load_units_gdf(df_units_here)
    return gdf_units


def load_region_outline_here(region_type: str, region: str):
    """
    Load the outline for a single region.

    Inputs
    ------
    region_type - str. Either 'isdn', 'icb', 'ambo'.
    region      - str. The name of the required region in the
                  region geojson file.

    Returns
    -------
    gdf_region          - gpd.GeoDataFrame. The outline of the single
                          chosen region.
    region_display_name - str. Nicer formatting for the input
                          region_type, e.g. "ISDN" for "icb".
    """
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
        gdf_region = gpd.read_file(f).to_crs('EPSG:27700')
        # Only keep the selected region:
        gdf_region = gdf_region.loc[gdf_region[region_type] == region]
        gdf_region['x'], gdf_region['y'] = (
            convert_shapely_polys_into_xy(gdf_region))
    else:
        # TO DO - create nearest unit geography
        gdf_region = None
        region_display_name = 'to do'
    return gdf_region, region_display_name


def set_network_map_bounds(
        gdf_units: gpd.GeoDataFrame,
        gdf_region: gpd.GeoDataFrame = None,
        transform_dict_units: dict = None
        ):
    """
    Calculate the bounding area for the patient flow maps.

    Rather than showing the whole of England when the relevant units
    are all in a smaller area, limit the bounds to everything that
    needs to be shown plus some buffer room. Create a separate list of
    bounds for each input to this function, then combine to find bounds
    for everything.

    Inputs
    ------
    gdf_units            - gpd.GeoDataFrame. The subset of stroke
                           units for this network map.
    gdf_region           - gpd.GeoDataFrame. Outline of the region
                           selected if region is ICB, ISDN, or ambo.
    transform_dict_units - dict. Pixel array to BNG coordinate
                           transform lookup for raster maps, e.g. unit
                           catchment map.

    Returns
    -------
    bounds   - list. New bounds that cover all inputs plus buffer.
    x_buffer - float. The breathing room given in the x direction.
    y_buffer - float. The breathing room given in the y direction.
    """
    # Gather bounds of all the input geography.
    # Start with the bounds of the selected stroke units:
    bounds_lists = [gdf_units.total_bounds]
    # If region outline exists, gather its bounds:
    if gdf_region is not None:
        bounds_lists.append(gdf_region.total_bounds)
    # If unit catchment map exists, gather its bounds:
    if transform_dict_units is not None:
        bounds_units_raster = [
            transform_dict_units['xmin'],
            transform_dict_units['im_ymin'],
            transform_dict_units['im_xmax'],
            transform_dict_units['ymax'],
        ]
        bounds_lists.append(bounds_units_raster)

    # Pick out the bounds that cover all of the input geography:
    bounds = [
        min([b[0] for b in bounds_lists]),
        min([b[1] for b in bounds_lists]),
        max([b[2] for b in bounds_lists]),
        max([b[3] for b in bounds_lists])
    ]

    # Add breathing room to region bounding box:
    x_buffer = (bounds[2] - bounds[0]) * 0.1
    y_buffer = (bounds[3] - bounds[1]) * 0.1
    bounds = [bounds[0] - x_buffer, bounds[1] - y_buffer,
              bounds[2] + x_buffer, bounds[3] + y_buffer]

    return bounds, x_buffer, y_buffer


def make_coords_nearest_unit_catchment(
        gdf_units: gpd.GeoDataFrame,
        df_net_u: pd.DataFrame,
        bounds: list,
        nearest_units: list,
        x_buffer: float,
        y_buffer: float,
        ):
    """
    Make coordinates for the catchment boxes outside the network maps.

    The catchment boxes sit outside the main map area. They have a
    label for how many people are in the catchment area of that unit.
    Then arrows go from that box to any stroke units that its patients
    attend.

    Here we set up the box locations by drawing a line from the centre
    of the map to the stroke unit and then extending it out to the
    map's outside border. This means that stroke units in the top right
    corner of the map will have their boxes placed in the top right of
    the axis.

    There are currently no checks whether the catchment boxes will
    overlap once drawn.

    Inputs
    ------
    gdf_units     - gpd.GeoDataFrame. Contains coordinates of the
                    stoke units in this network map, including those
                    outside the selected region who collect transfers
                    from within the region or whose catchment area
                    covers inside the region.
    df_net_u      - pd.DataFrame. Contains the admissions numbers
                    in each unit's usual catchment area. This is only
                    used here to gather the admission numbers. It is
                    not part of the coordinate calculations.
    bounds        - list. Bounds of the network map.
    nearest_units - list. List of only units in the selected region.
                    This list is usually smaller than the units in
                    gdf_units.
    x_buffer      - float. The buffer space between the contents of the
                    network map and the drawn boundaries in the
                    x-direction.
    y_buffer      - float. Same for y-direction.

    Returns
    -------
    gdf - gpd.GeoDataFrame. Contains coordinates for the catchment
          boxes named 'x_anchor', 'y_anchor'.
    """
    box_centre = [0.5*(bounds[0]+bounds[2]), 0.5*(bounds[1]+bounds[3])]

    # Store catchment box coordinates in here:
    gdf = gdf_units.copy()
    # Limit to units in the region:
    gdf['nearest_unit'] = 'nearest_' + gdf.index.astype(str)
    gdf = gdf.loc[gdf['nearest_unit'].isin(nearest_units)]
    # Make coordinates for each unit in the region's "nearest unit"
    # anchor. Find the angle between the centre of the region and
    # each unit.
    gdf['x_off'] = gdf['BNG_E'] - box_centre[0]
    gdf['y_off'] = gdf['BNG_N'] - box_centre[1]
    gdf['angle'] = np.arctan2(gdf['y_off'], gdf['x_off'])
    gdf['angle_deg'] = gdf['angle'] * 180.0 / np.pi

    # Define the locations of the rectangle around the network map.
    # The catchment boxes will use one of these four coordinates
    # depending on which side of the map they are placed.
    anch_top = bounds[3] + (2.0 * y_buffer)
    anch_left = bounds[0] - (2.0 * x_buffer)
    anch_bottom = bounds[1] - (2.0 * y_buffer)
    anch_right = bounds[2] + (2.0 * x_buffer)

    # Find angles to the corners of the network map so that we can
    # decide which side each catchment box must go on:
    angle_to_top_right = np.arctan2(
        (bounds[3] - box_centre[1]), (bounds[2] - box_centre[0]))
    angle_to_top_left = np.arctan2(
        (bounds[3] - box_centre[1]), (bounds[0] - box_centre[0]))
    angle_to_bottom_left = np.arctan2(
        (bounds[1] - box_centre[1]), (bounds[0] - box_centre[0]))
    angle_to_bottom_right = np.arctan2(
        (bounds[1] - box_centre[1]), (bounds[2] - box_centre[0]))

    # Pick out which catchment boxes go on each side of the network
    # map:
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
    # Label the chosen sides:
    gdf.loc[mask_top, 'side'] = 'top'
    gdf.loc[mask_left, 'side'] = 'left'
    gdf.loc[mask_bottom, 'side'] = 'bottom'
    gdf.loc[mask_right, 'side'] = 'right'
    # Pick out the fixed coordinate for each side:
    gdf.loc[mask_top, 'y_anchor'] = anch_top
    gdf.loc[mask_left, 'x_anchor'] = anch_left
    gdf.loc[mask_bottom, 'y_anchor'] = anch_bottom
    gdf.loc[mask_right, 'x_anchor'] = anch_right
    # Calculate the other coordinate for each side:
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

    # Limit the results gdf to the most relevant info.
    # Just the catchment box coordinates and some formatting:
    cols_to_keep = ['nearest_unit', 'side', 'x_anchor', 'y_anchor',
                    'ssnap_name', 'Use_MT', 'colour']
    gdf = gdf[cols_to_keep]
    # Add in the admissions numbers in each unit's catchment area:
    gdf = pd.merge(
        gdf.reset_index(),
        df_net_u[['first_unit', 'admissions']],
        left_on='Postcode', right_on='first_unit', how='left'
        ).set_index('Postcode').drop('first_unit', axis='columns')
    return gdf


#MARK: Plot figures
# ########################
# ##### PLOT FIGURES #####
# ########################
def draw_units_map(map_traces: dict, outline_name: str = 'none'):
    """
    Draw a map of England with the stroke units by service type.

    Draw:
    + either:
      + outline of England and map of whether LSOA nearest a CSC
      + region outlines and map of whether LSOA nearest a CSC
      + map of unit catchment areas
    + stroke units with markers to show their services
    + major roads

    Inputs
    ------
    map_traces   - dict. Contains plotly traces of everything that
                   will be displayed on this figure.
    outline_name - str. Which outline type if any to draw on the
                   maps. Can be 'isdn', 'icb', 'ambo',
                   'nearest_ivt_unit', or 'nearest_mt_unit'.
    """
    fig = go.Figure()

    # --- Draw traces ---
    # "Nearest unit has MT" map.
    # Only draw the "nearest unit has MT" raster if we're not
    # drawing all the unit catchments anyway.
    if outline_name in ['nearest_ivt_unit', 'nearest_mt_unit']:
        pass
    else:
        fig.add_trace(map_traces['raster_nearest_csc']['trace'])
    # Always draw the "nearest unit has MT" legend cheat:
    fig.add_trace(map_traces['raster_nearest_csc']['trace_legend'])

    # Region outline or unit catchment raster:
    if outline_name == 'none':
        fig.add_trace(map_traces['england_outline'])
    else:
        if f'{outline_name}_outlines' in map_traces.keys():
            # ISDN, ICB, or ambulance outlines.
            for t in map_traces[f'{outline_name}_outlines']:
                fig.add_trace(t)
        else:
            # Unit catchment raster.
            fig.add_trace(map_traces[f'raster_{outline_name}'])

    # Always add roads and unit locations.
    fig.add_trace(map_traces['roads'])
    fig.add_trace(map_traces['units']['ivt'])
    fig.add_trace(map_traces['units']['mt'])

    # --- Layout ---
    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(width=500, height=600, margin_t=25, margin_b=0)
    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(scaleanchor='x', scaleratio=1)

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')
    plotly_config = get_map_config()

    # --- Display fig ---
    st.plotly_chart(
        fig,
        config=plotly_config,
        width='content',
        )


def draw_units_msu_map(map_traces: dict, outline_name: str = 'none'):
    """
    Draw two map of England: stroke units IVT/MT and MSU/MT.

    Draw:
    + either:
      + outline of England and map of whether LSOA nearest a CSC
      + region outlines and map of whether LSOA nearest a CSC
      + map of unit catchment areas
    + stroke units with markers to show their services
    + major roads

    Inputs
    ------
    map_traces   - dict. Contains plotly traces of everything that
                   will be displayed on this figure.
    outline_name - str. Which outline type if any to draw on the
                   maps. Can be 'isdn', 'icb', 'ambo',
                   'nearest_ivt_unit', or 'nearest_mt_unit'.
    """
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.0,
        subplot_titles=['Usual care', 'MSU available'],
        )

    # --- Draw traces ---
    # "Nearest unit has MT" map.
    # Only draw the "nearest unit has MT" raster if we're not
    # drawing all the unit catchments anyway.
    if outline_name in ['nearest_ivt_unit', 'nearest_mt_unit']:
        pass
    else:
        fig.add_trace(
            map_traces['raster_nearest_csc']['trace'], row='all', col='all')
    # Always draw the "nearest unit has MT" legend cheat:
    fig.add_trace(map_traces['raster_nearest_csc']['trace_legend'])

    # Region outline or unit catchment raster:
    if outline_name == 'none':
        fig.add_trace(map_traces['england_outline'], row='all', col='all')
    else:
        if f'{outline_name}_outlines' in map_traces.keys():
            # ISDN, ICB, or ambulance outlines.
            for t in map_traces[f'{outline_name}_outlines']:
                fig.add_trace(t, row='all', col='all')
        else:
            # Unit catchment raster.
            fig.add_trace(
                map_traces[f'raster_{outline_name}'], row='all', col='all')

    # Always add roads and unit locations.
    fig.add_trace(map_traces['roads'], row='all', col='all')
    fig.add_trace(map_traces['units']['msu'], row='all', col=2)
    fig.add_trace(map_traces['units']['ivt'], row='all', col=1)
    fig.add_trace(map_traces['units']['mt'], row='all', col='all')

    # --- Layout ---
    fig = england_map_setup(fig)
    # Figure setup.
    fig.update_layout(width=500, height=600, margin_t=25, margin_b=0)

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

    # --- Display fig ---
    st.plotly_chart(
        fig,
        config=plotly_config,
        width='stretch',
        )


def plot_outcome_maps(
        map_traces: dict,
        map_order: list,
        colour_dicts: dict,
        all_cmaps: list,
        outline_name: str = 'none',
        show_msu_bases: bool = False,
        title: str = ''
        ):
    """
    Draw three maps: usual outcome, difference, and population.

    The left map 'lhs' shows the outcome in usual care scenario.
    The middle map 'rhs' shows the difference between the outcomes
    in the given scenario and in usual care. The right map 'pop'
    shows the population density.

    The map traces have been made in advance.

    Inputs
    ------
    map_traces     - dict. Contains plotly traces for maps, units,
                     roads, regions...
    map_order      - list. Ordered traces to show as the three maps.
    colour_dicts   - dict. Contains the colour limits and display names
                     for each map.
    all_cmaps      - list. The colour map options to be shown as
                     restyle buttons.
    outline_name   - str. Which region types to draw as outlines.
    show_msu_bases - bool. Whether to draw MSU bases.
    title          - str. Top title for the figure.

    Returns
    -------
    fig - go.Figure. Return the figure so that it can be cached.
          Call st.plotly_chart in the main script instead of here.
    """
    # Map labels:
    subplot_titles = [colour_dicts[m]['title'] for m in map_order]
    fig = make_subplots(rows=1, cols=len(map_order), horizontal_spacing=0.0,
                        subplot_titles=subplot_titles)

    # --- Map traces ---
    # Size of the colour bar in figure units:
    cbar_len = 1.0 / len(map_order)
    # Draw the three maps and their colour bars:
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

    # --- Region, unit, road traces ---
    if outline_name == 'none':
        # Draw England:
        fig.add_trace(map_traces['england_outline'], row='all', col='all')
    else:
        # Draw region outlines:
        for t in map_traces[f'{outline_name}_outlines']:
            fig.add_trace(t, row='all', col='all')
    # Always draw roads:
    fig.add_trace(map_traces['roads'], row='all', col='all')
    # Draw stroke units:
    if show_msu_bases:
        fig.add_trace(map_traces['units']['msu'], row='all', col=2)
    fig.add_trace(map_traces['units']['ivt'], row='all', col='all')
    fig.add_trace(map_traces['units']['mt'], row='all', col='all')

    # --- Layout ---
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
        df_net_u: pd.DataFrame,
        df_net_r: pd.DataFrame,
        df_unit_services: pd.DataFrame,
        gdf_nearest_units: gpd.GeoDataFrame,
        gdf_units: gpd.GeoDataFrame,
        bounds: list,
        catch_trace: go.Heatmap,
        gdf_region: gpd.GeoDataFrame = None,
        region_display_name: str = None,
        subplot_titles: list = []
        ):
    """
    Draw maps with arrows to show patient flow between units.

    Inputs
    ------
    df_net_u            - pd.DataFrame. Patient numbers to nearest,
                          first, transfer units for the units here
                          in the usual care scenario.
    df_net_r            - pd.DataFrame. As above for the unusual
                          scenario e.g. redirection.
    df_unit_services    - pd.DataFrame. Use as a lookup for stroke
                          units and their display names.
    gdf_nearest_units   - gpd.GeoDataFrame. Coordinates for catchment
                          area boxes and the admissions in each area. 
    gdf_units           - gpd.GeoDataFrame. Coordinates of stroke
                          units.
    bounds              - list. Extent of the network map geography.
    catch_trace         - go.HeatMap. Plotly trace showing unit
                          catchment areas.
    gdf_region          - gpd.GeoDataFrame or None. Outline for a
                          single selected region to draw on maps.
    region_display_name - str. Display name of selected region.
    subplot_titles      - list. Subplot titles. Top left, right,
                          bottom left, right.
    """
    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.0,
        vertical_spacing=0.0,
        subplot_titles=subplot_titles,
        )

    # --- Draw traces ---
    # Border:
    fig.add_shape(
        type="rect",
        x0=bounds[0], y0=bounds[1], x1=bounds[2], y1=bounds[3],
        fillcolor='rgba(0.95, 0.95, 0.95, 1.0)',
        line=dict(color='grey', width=4,),
        layer='between',  # below other traces
        col=[1, 2, 2], row=[1, 1, 2],
    )

    # Load England and restrict it to the area being drawn:
    gdf_eng = load_england_outline(bounds)
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
        ),
        col=[1, 2, 2], row=[1, 1, 2],
    )

    # Region catchment heatmap:
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
                ),
                col=[1, 2, 2], row=[1, 1, 2]
                )

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
                            standoff=standoff,
                            line=dict(color='black', width=2)),
                line_color=bcolour,
                line_width=bw,
                # text=[a],
                hoverinfo='skip',
                name=None,
            ), col=2, row=d+1)
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
                name=(None if link_drawn_to_first
                      else 'Admissions to first unit'),
            ), col=2, row=d+1)
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
            ), col=2, row=d+1)

        # First unit to transfer unit:
        df_net_trans = df_net.copy()
        df_net_trans = df_net_trans.drop('nearest_unit', axis='columns')
        df_net_trans = df_net_trans.groupby(
            ['first_unit', 'transfer_unit']).sum().reset_index()
        colour = 'rgba(0.9, 0.9, 0.9, 1.0)'
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
                w = np.log(a)
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
                    marker=dict(size=baw, symbol='arrow-up',
                                angleref='previous',
                                standoff=standoff,
                                line=dict(color='black', width=2)),
                    line_color=bcolour,
                    line_width=bw,
                    # text=[a],
                    hoverinfo='skip',
                    name=None,
                ), col=2, row=d+1)
                # The actual arrow:
                fig.add_trace(go.Scatter(
                    x=[x_first, x_trans],
                    y=[y_first, y_trans],
                    mode='lines+markers',
                    marker=dict(size=aw, symbol='arrow-up',
                                angleref='previous', standoff=standoff),
                    line_color=colour,
                    line_width=w,
                    hoverinfo='skip',
                    name=(None if link_drawn_to_trans
                          else 'Transfers for thrombectomy'),
                ), col=2, row=d+1)
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
                        '%{customdata[2]:.1f} patients transfer from ' +
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
                ), col=2, row=d+1)

    # Stroke units:
    unit_traces = make_units_traces(gdf_units)
    fig.add_trace(go.Scatter(unit_traces['ivt']),
                  col=[1, 2, 2], row=[1, 1, 2],)
    fig.add_trace(go.Scatter(unit_traces['mt']),
                  col=[1, 2, 2], row=[1, 1, 2],)

    # Catchment boxes:
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
    ), row='all', col=2)

    # --- Layout ---
    fig.update_layout(hovermode='closest')

    fig = england_map_setup(fig)
    fig.update_yaxes(col=1, row=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, row=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, row=2, scaleanchor='x', scaleratio=1)
    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text=region_display_name)

    # Calculate height based on aspect ratio:
    width = 800
    height = width
    # Figure setup.
    fig.update_layout(width=width, height=height, margin_t=50, margin_b=0,
                      margin_l=0, margin_r=0,)
    fig.update_layout(legend=dict(
        # orientation="h",
        yanchor='middle',
        y=0.25,
        xanchor='center',
        x=0.25,
    ))
    plotly_config = get_map_config()
    st.plotly_chart(fig, config=plotly_config, width='stretch')
