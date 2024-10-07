"""
Functions for making plotly maps for Streamlit.
"""
import streamlit as st
import numpy as np
import os
import geopandas
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from utilities.maps import convert_shapely_polys_into_xy
from utilities.inputs import load_roads_gdf

import stroke_maps.load_data


def create_stroke_team_markers(df_units=None):
    """
    Create plotly traces for the stroke team scatter markers.

    Create the traces first and then later actually draw them
    on the figures because some traces need to appear identically
    across multiple subplots.

    Inputs
    ------
    df_units - pd.DataFrame. Information about which units provide
               which services, and this affects which markers are
               drawn for each unit.

    Returns
    -------
    traces - dict. An entry for IVT, MT, and MSU units. Contains
             plotly traces for the markers for the stroke units.
    """

    # Add stroke team markers.
    if df_units is None:
        df_units = stroke_maps.load_data.stroke_unit_region_lookup()
        # Limit to England:
        mask = df_units['country'] == 'England'
        df_units = df_units.loc[mask].copy()
    else:
        pass
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

    # # Convert to British National Grid.
    # The geometry column should be BNG on import, so just overwrite
    # the longitude and latitude columns that are by default long/lat.
    col_geo = 'geometry'
    # gdf_points_units = gdf_points_units.set_crs(
    #     'EPSG:27700', allow_override=True)
    # # Overwrite long and lat:
    # gdf_points_units[('Longitude', 'any')] = gdf_points_units[col_geo].x
    # gdf_points_units[('Latitude', 'any')] = gdf_points_units[col_geo].y

    # Set up markers using a new column in DataFrame.
    # Set everything to the IVT marker:
    markers = np.full(len(gdf_points_units), 'circle', dtype=object)
    # Update MT units:
    col_use_mt = 'Use_MT'
    mask_mt = (gdf_points_units[col_use_mt] == 1)
    markers[mask_mt] = 'square'
    # Store in the DataFrame:
    gdf_points_units['marker'] = markers

    # Add markers in separate traces for the sake of legend entries.
    # Pick out which stroke unit types are where in the gdf:
    col_ivt = 'Use_IVT'
    col_mt = 'Use_MT'
    col_msu = 'Use_MSU'
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
            x=gdf_points_units.loc[mask, 'BNG_E'],
            y=gdf_points_units.loc[mask, 'BNG_N'],
            mode='markers',
            marker={
                'symbol': s_dict['marker'],
                'color': s_dict['colour'],
                'line': {'color': 'black', 'width': 1},
                'size': s_dict['size']
            },
            name=s_dict['label'],
            customdata=np.stack(
                [gdf_points_units.loc[mask, 'ssnap_name']],
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


def plotly_blank_maps(subplot_titles: list = None, n_blank: int = 2):
    """
    Create dummy subplots with blank maps in them to mask load times.

    Inputs
    ------
    subplot_titles - list or None. Titles for the subplots.
    n_blank        - int. How many subplots to create.
    """
    # Don't replace this with stroke-maps!
    # This uses the same simplified LSOA shapes as plotted.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=n_blank,
        horizontal_spacing=0.0,
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
        name=' ' * 20
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
        # width=1200,
        height=700,
        margin_t=40,
        margin_b=60  # mimic space taken up by colourbar
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


def plotly_unit_maps(
        traces_units: dict = None,
        unit_subplot_dict: dict = {},
        gdf_catchment_lhs: geopandas.GeoDataFrame = None,
        gdf_catchment_rhs: geopandas.GeoDataFrame = None,
        outline_names_col: str = '',
        outline_name: str = '',
        subplot_titles: list = [],
        ):
    """
    Create England outline with unit locations.

    Inputs
    ------
    traces_units      - dict. Plotly traces of scatter markers for
                        stroke units.
    unit_subplot_dict - dict. Which unit traces should be shown on
                        which subplots (by number).
    """
    # Don't replace this with stroke-maps!
    # This uses the same simplified LSOA shapes as plotted.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.0,
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
            fillcolor='rgba(1.0, 1.0, 1.0, 0.2)',
            line_color='grey',
            showlegend=False,
            hoverinfo='skip',
            ), col='all', row='all')

    gdf_roads = load_roads_gdf()

    for i in gdf_roads.index:
        fig.add_trace(go.Scatter(
            x=gdf_roads.loc[i, 'x'],
            y=gdf_roads.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            line_width=0.5,
            showlegend=False,
            hoverinfo='skip',
            ), col='all', row='all')

    def draw_outline(fig, gdf_catchment, col='all'):
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
                ), row='all', col=col
                )

    if gdf_catchment_lhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_lhs, col=1)

    if gdf_catchment_rhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_rhs, col=2)

    fig.update_traces(
        hoverlabel=dict(
            bgcolor='grey',
            font_color='white'),
        selector={'name': outline_name}
    )

    # --- Stroke unit scatter markers ---
    if len(unit_subplot_dict) > 0:
        if gdf_catchment_lhs is None:
            pass
        else:
            # # Add a blank trace to put a gap in the legend.
            # Stupid? Yes. Works? Also yes.
            # Make sure the name isn't the same as any other blank name
            # already set, e.g. in combo_colour_dict.
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
    # (e.g. multiple sets of IVT unit, MT unit)
    # from https://stackoverflow.com/a/62162555
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
            borderwidth=2
        )
    )

    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks:
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=600,
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
        gdf_lhs: geopandas.GeoDataFrame,
        gdf_rhs: geopandas.GeoDataFrame,
        gdf_catchment_lhs: geopandas.GeoDataFrame = None,
        gdf_catchment_rhs: geopandas.GeoDataFrame = None,
        outline_names_col: str = '',
        outline_name: str = '',
        traces_units: dict = None,
        unit_subplot_dict: dict = {},
        subplot_titles: list = [],
        legend_title: str = '',
        colour_dict: dict = {},
        colour_diff_dict: dict = {}
        ):
    """
    Main map-drawing function.

    Inputs
    ------
    gdf_lhs           - geopandas.GeoDataFrame. Data for left-hand
                        side.
    gdf_rhs           - geopandas.GeoDataFrame. Data for right-hand
                        side.
    gdf_catchment_lhs - geopandas.GeoDataFrame. Optional. Data to
                        plot over the top of the other gdf, for example
                        catchment area outlines. For left-hand map.
    gdf_catchment_rhs - geopandas.GeoDataFrame. Optional. Same but for
                        right-hand map.
    outline_names_col - str. Name of the column in gdf_catchment that
                        contains data to show on the hover text.
    outline_name      - str. One value from the 'outcome_type' column
                        in gdf_catchment. (Should all be same values).
    traces_units      - dict. Plotly traces of scatter markers for
                        stroke units.
    unit_subplot_dict - dict. Which unit traces should be shown on
                        which subplots (by number).
    subplot_titles    - list. Title appearing above each subplot.
    legend_title      - str. Title for the legend.
    colour_dict       - dict. Colour band labels to hex colour lookup
                        for the left-hand-side map.
    colour_diff_dict  - dict. Same for the right-hand-side map.
    """
    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.0,
        subplot_titles=subplot_titles
        )

    # Add a blank outline of England:
    # Don't replace this with stroke-maps! This one matches
    # the simplified LSOA shapes.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

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

    # Return a gdf of some x, y coordinates to scatter
    # in such a tiny size that they'll never be seen,
    # but that will cause the colourbar of the colour scale to display.
    # Separate colour scales for the two maps.

    def draw_dummy_scatter(fig, colour_dict, col=1, trace_name=''):
        # Dummy coordinates:
        # Isle of Man: 238844, 482858
        bonus_x = 238844
        bonus_y = 482858
        x_dummy = np.array([bonus_x]*2)
        y_dummy = np.array([bonus_y]*2)
        z_dummy = np.array([0.0, 1.0])

        # Sometimes the ticks don't show at the very ends of the colour bars.
        # In that case, cheat with e.g.
        # tick_locs = [bounds[0] + 1e-2, *bounds[1:-1], bounds[-1] - 1e-3]
        tick_locs = colour_dict['bounds_for_colour_scale']

        tick_names = [f'{t:.3f}' for t in colour_dict['v_bands']]
        tick_names = ['←', *tick_names, '→']

        # Replace zeroish with zero:
        # (this is a visual difference only - it combines two near-zero
        # ticks and their labels into a single tick.)
        # If vmin is negative (not zero) and vmax is positive (not zero):
        if ((np.sign(colour_dict['vmin']) == -1) & (np.sign(colour_dict['vmax']) == 1)):
            ind_z = np.where(np.sign(colour_dict['v_bands']) >= 0.0)[0][0] + 1
            tick_z = np.mean([tick_locs[ind_z-1], tick_locs[ind_z]])
            name_z = '0'

            tick_locs_z = np.append(tick_locs[:ind_z - 1], tick_z)
            tick_locs_z = np.append(tick_locs_z, tick_locs[ind_z+1:])
            tick_locs = tick_locs_z

            tick_names_z = np.append(tick_names[:ind_z - 1], name_z)
            tick_names_z = np.append(tick_names_z, tick_names[ind_z+1:])
            tick_names = tick_names_z

        # Add dummy scatter:
        fig.add_trace(go.Scatter(
            x=x_dummy,
            y=y_dummy,
            marker=dict(
                color=z_dummy,
                colorscale=colour_dict['colour_scale'],
                colorbar=dict(
                    thickness=20,
                    tickmode='array',
                    tickvals=tick_locs,
                    ticktext=tick_names,
                    # ticklabelposition='outside top'
                    title=colour_dict['title']
                    ),
                size=1e-4,
                ),
            showlegend=False,
            mode='markers',
            hoverinfo='skip',
            name=trace_name
        ), row='all', col=col)

        return fig

    fig = draw_dummy_scatter(fig, colour_dict, col=1, trace_name='cbar')
    fig = draw_dummy_scatter(fig, colour_diff_dict, col=2,
                             trace_name='cbar_diff')
    fig.update_traces(
        {'marker': {'colorbar': {
            'orientation': 'h',
            'x': 0.0,
            'y': -0.1,
            'len': 0.5,
            'xanchor': 'left',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }}},
        selector={'name': 'cbar'}
        )
    fig.update_traces(
        {'marker': {'colorbar': {
            'orientation': 'h',
            'x': 1.0,
            'y': -0.1,
            'len': 0.5,
            'xanchor': 'right',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }}},
        selector={'name': 'cbar_diff'}
        )

    # Add each row of the dataframe separately.
    # Scatter the edges of the polygons and use "fill" to colour
    # within the lines.
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

    def draw_outline(fig, gdf_catchment, col='all'):
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
                ), row='all', col=col
                )

    if gdf_catchment_lhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_lhs, col=1)

    if gdf_catchment_rhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_rhs, col=2)

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
        if gdf_catchment_lhs is None:
            pass
        else:
            # # Add a blank trace to put a gap in the legend.
            # Stupid? Yes. Works? Also yes.
            # Make sure the name isn't the same as any other blank name
            # already set, e.g. in combo_colour_dict.
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
    # (e.g. multiple sets of IVT unit, MT unit)
    # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    # This makes sure that if multiple maps use the exact same
    # colours and labels, the labels only appear once in the legend.

    fig.update_layout(
        legend=dict(
            title_text=legend_title,
            bordercolor='grey',
            borderwidth=2
        )
    )

    # Figure setup.
    fig.update_layout(
        # width=1200,
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
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=plotly_config
        )


def plotly_many_heatmaps(
        burned_lhs: geopandas.GeoDataFrame,
        burned_rhs: geopandas.GeoDataFrame,
        burned_pop: geopandas.GeoDataFrame,
        gdf_catchment_lhs: geopandas.GeoDataFrame = None,
        gdf_catchment_rhs: geopandas.GeoDataFrame = None,
        gdf_catchment_pop: geopandas.GeoDataFrame = None,
        outline_names_col: str = '',
        outline_name: str = '',
        traces_units: dict = None,
        unit_subplot_dict: dict = {},
        subplot_titles: list = [],
        legend_title: str = '',
        dict_colours: dict = {},
        dict_colours_diff: dict = {},
        dict_colours_pop: dict = {},
        transform_dict: dict = {},
        ):
    """
    Main map-drawing function.

    UPDATE ME for heatmaps ok

    Inputs
    ------
    gdf_lhs           - geopandas.GeoDataFrame. Data for left-hand
                        side.
    gdf_rhs           - geopandas.GeoDataFrame. Data for right-hand
                        side.
    gdf_catchment_lhs - geopandas.GeoDataFrame. Optional. Data to
                        plot over the top of the other gdf, for example
                        catchment area outlines. For left-hand map.
    gdf_catchment_rhs - geopandas.GeoDataFrame. Optional. Same but for
                        right-hand map.
    outline_names_col - str. Name of the column in gdf_catchment that
                        contains data to show on the hover text.
    outline_name      - str. One value from the 'outcome_type' column
                        in gdf_catchment. (Should all be same values).
    traces_units      - dict. Plotly traces of scatter markers for
                        stroke units.
    unit_subplot_dict - dict. Which unit traces should be shown on
                        which subplots (by number).
    subplot_titles    - list. Title appearing above each subplot.
    legend_title      - str. Title for the legend.
    colour_dict       - dict. Colour band labels to hex colour lookup
                        for the left-hand-side map.
    colour_diff_dict  - dict. Same for the right-hand-side map.
    """
    # ----- Plotting -----
    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.0,
        subplot_titles=subplot_titles
        )

    # Add a blank outline of England:
    # Don't replace this with stroke-maps! This one matches
    # the simplified LSOA shapes.
    path_to_file = os.path.join('data', 'outline_england.geojson')
    gdf_ew = geopandas.read_file(path_to_file)

    x_list, y_list = convert_shapely_polys_into_xy(gdf_ew)
    gdf_ew['x'] = x_list
    gdf_ew['y'] = y_list

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

    # Draw the outcome maps:
    fig.add_trace(go.Heatmap(
        z=burned_lhs,
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
            title=dict_colours['title']
            ),
        name='lhs',
        hoverinfo='skip',
    ), row='all', col=1)

    fig.add_trace(go.Heatmap(
        z=burned_rhs,
        transpose=False,
        x0=transform_dict['xmin'],
        dx=transform_dict['pixel_size'],
        y0=transform_dict['ymin'],
        dy=transform_dict['pixel_size'],
        zmin=dict_colours_diff['vmin'],
        zmax=dict_colours_diff['vmax'],
        colorscale=dict_colours_diff['cmap'],
        colorbar=dict(
            thickness=20,
            # tickmode='array',
            # tickvals=tick_locs,
            # ticktext=tick_names,
            # ticklabelposition='outside top'
            title=dict_colours_diff['title']
            ),
        name='rhs',
        hoverinfo='skip',
    ), row='all', col=2)

    fig.add_trace(go.Heatmap(
        z=burned_pop,
        transpose=False,
        x0=transform_dict['xmin'],
        dx=transform_dict['pixel_size'],
        y0=transform_dict['ymin'],
        dy=transform_dict['pixel_size'],
        zmin=dict_colours_pop['vmin'],
        zmax=dict_colours_pop['vmax'],
        colorscale=dict_colours_pop['cmap'],
        colorbar=dict(
            thickness=20,
            # tickmode='array',
            # tickvals=tick_locs,
            # ticktext=tick_names,
            # ticklabelposition='outside top'
            title=dict_colours_pop['title']
            ),
        name='pop',
        hoverinfo='skip',
    ), row='all', col=3)

    fig.update_traces(
        {'colorbar': {
            'orientation': 'h',
            'x': 0.0,
            'y': -0.2,
            'len': 0.333,
            'xanchor': 'left',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }},
        selector={'name': 'lhs'}
        )
    fig.update_traces(
        {'colorbar': {
            'orientation': 'h',
            'x': 0.5,
            'y': -0.2,
            'len': 0.333,
            'xanchor': 'center',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }},
        selector={'name': 'rhs'}
        )
    fig.update_traces(
        {'colorbar': {
            'orientation': 'h',
            'x': 1.0,
            'y': -0.2,
            'len': 0.333,
            'xanchor': 'right',
            'title_side': 'bottom'
            # 'xref': 'paper'
            }},
        selector={'name': 'pop'}
        )

    gdf_roads = load_roads_gdf()

    for i in gdf_roads.index:
        fig.add_trace(go.Scatter(
            x=gdf_roads.loc[i, 'x'],
            y=gdf_roads.loc[i, 'y'],
            mode='lines',
            fill="toself",
            fillcolor='rgba(0, 0, 0, 0)',
            line_color='grey',
            line_width=0.5,
            showlegend=False,
            hoverinfo='skip',
            ), col='all', row='all')

    def draw_outline(fig, gdf_catchment, col='all'):
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
                ), row='all', col=col
                )

    if gdf_catchment_lhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_lhs, col=1)

    if gdf_catchment_rhs is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_rhs, col=2)

    if gdf_catchment_pop is None:
        pass
    else:
        draw_outline(fig, gdf_catchment_pop, col=3)

    fig.update_traces(
        hoverlabel=dict(
            bgcolor='grey',
            font_color='white'),
        selector={'name': outline_name}
    )
    # Equivalent to pyplot set_aspect='equal':
    fig.update_yaxes(col=1, scaleanchor='x', scaleratio=1)
    fig.update_yaxes(col=2, scaleanchor='x2', scaleratio=1)
    fig.update_yaxes(col=3, scaleanchor='x3', scaleratio=1)

    # Shared pan and zoom settings:
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Remove axis ticks.
    fig.update_xaxes(tickvals=[], showticklabels=False,
                     showgrid=False, zeroline=False)
    fig.update_yaxes(tickvals=[], showticklabels=False,
                     showgrid=False, zeroline=False)

    # --- Stroke unit scatter markers ---
    if len(unit_subplot_dict) > 0:
        if gdf_catchment_lhs is None:
            pass
        else:
            # # Add a blank trace to put a gap in the legend.
            # Stupid? Yes. Works? Also yes.
            # Make sure the name isn't the same as any other blank name
            # already set, e.g. in combo_colour_dict.
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
    # (e.g. multiple sets of IVT unit, MT unit)
    # from https://stackoverflow.com/a/62162555
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    # This makes sure that if multiple maps use the exact same
    # colours and labels, the labels only appear once in the legend.

    fig.update_layout(
        legend=dict(
            title_text=legend_title,
            bordercolor='grey',
            borderwidth=2
        )
    )

    # Figure setup.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=40,
        margin_b=0
        )

    # Disable clicking legend to remove trace:
    fig.update_layout(legend_itemclick=False)
    fig.update_layout(legend_itemdoubleclick=False)

    # BUTTONS TEST - https://plotly.com/python/custom-buttons/
    # Add drowdowns
    # Colour scales dict:
    dict_colourscales_lhs = {}
    dict_colourscales_rhs = {}
    dict_colourscales_pop = {}
    import utilities.colour_setup as colour_setup
    cmaps = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
    cmaps += [c[:-2] if c.endswith('_r') else f'{c}_r' for c in cmaps]
    for c in cmaps:
        dict_colourscales_lhs[c] = colour_setup.make_colour_list_for_plotly_button(
            c, vmin=dict_colours['vmin'], vmax=dict_colours['vmax'])
        dict_colourscales_rhs[c] = colour_setup.make_colour_list_for_plotly_button(
            c, vmin=dict_colours_diff['vmin'], vmax=dict_colours_diff['vmax'])
        dict_colourscales_pop[c] = colour_setup.make_colour_list_for_plotly_button(
            c, vmin=dict_colours_pop['vmin'], vmax=dict_colours_pop['vmax'])

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=[{'colorscale': [dict_colourscales_lhs[c], dict_colourscales_rhs[c], dict_colourscales_pop[c]]},
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
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=plotly_config
        )
