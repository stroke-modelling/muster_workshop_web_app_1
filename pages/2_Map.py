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
import os
import geopandas
import pyproj  # for crs conversion
import matplotlib.pyplot as plt  # for colour maps
from datetime import datetime

# For running outcomes:
from classes.geography_processing import Geoprocessing
from classes.model import Model
from classes.scenario import Scenario

# Custom functions:
from utilities.fixed_params import page_setup
from utilities.plot_timeline import build_data_for_timeline, draw_timeline
import utilities.utils as utils
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


@st.cache_data
def import_geojson(region_type: 'str'):
    """
    Import a geojson file as GeoDataFrame.

    The crs (coordinate reference system) is set to British National
    Grid.

    Inputs
    ------
    region_type - str. Lookup name for selecting a geojson file.
                  This should be one of the column names from the
                  various regions files.

    Returns
    -------
    gdf_boundaries - GeoDataFrame. One row per region shape in the
                     file. Expect columns for region name and geometry.
    """
    # Select geojson file based on input region type:
    geojson_file_dict = {
        # 'LSOA11NM': 'LSOA_V3_reduced.geojson',  # 'LSOA.geojson',
        'LSOA11NM': 'LSOA_V3_reduced_simplified.geojson',
        'MSOA11NM': 'MSOA_V3_reduced_simplified.geojson',
        'SICBL22NM': 'SICBL.geojson',
        'LHB20NM': 'LHB.geojson'
    }

    # Import region file:
    file_input = geojson_file_dict[region_type]
    # Relative import from package files:
    # path_to_file = files('stroke_maps.data').joinpath(file_input)
    path_to_file = os.path.join('data', file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)

    if region_type == 'LSOA11NM':
        index_col = 'LSOA11CD'
        # Only keep these columns.
        geo_cols = [
            'LSOA11NM',
            # 'BNG_E', 'BNG_N',
            # 'LONG', 'LAT', 'GlobalID',
            'geometry']
    elif region_type == 'MSOA11NM':
        index_col = 'MSOA11CD'
        # Only keep these columns.
        geo_cols = [
            'MSOA11NM',
            # 'BNG_E', 'BNG_N',
            # 'LONG', 'LAT', 'GlobalID',
            'geometry']

    else:
        index_col = 'region_code'
        # Only keep these columns:
        geo_cols = ['region', 'BNG_E', 'BNG_N',
                    'LONG', 'LAT', 'GlobalID', 'geometry']

        # Find which columns to rename to 'region' and 'region_code'.
        if (region_type.endswith('NM') | region_type.endswith('NMW')):
            region_prefix = region_type.removesuffix('NM')
            region_prefix = region_prefix.removesuffix('NMW')
            region_code = region_prefix + 'CD'
        elif (region_type.endswith('nm') | region_type.endswith('nmw')):
            region_prefix = region_type.removesuffix('NM')
            region_prefix = region_prefix.removesuffix('NMW')
            region_code = region_prefix + 'cd'
        else:
            # This shouldn't happen.
            # TO DO - does this need a proper exception or can
            # we just change the above to if/else? ------------------------------
            region_code = region_type[:-2] + 'CD'

        try:
            # Rename this column:
            gdf_boundaries = gdf_boundaries.rename(columns={
                region_type: 'region',
                region_code: 'region_code'
            })
        except KeyError:
            # That column doesn't exist.
            # Try finding a column that has the same start and end
            # as requested:
            prefix = region_type[:3]
            suffix = region_type[-2:]
            success = False
            for column in gdf_boundaries.columns:
                # Casefold turns all UPPER into lower case.
                match = ((column[:3].casefold() == prefix.casefold()) &
                         (column[-2:].casefold() == suffix.casefold()))
                if match:
                    # Rename this column:
                    col_code = column[:-2] + region_code[-2:]
                    gdf_boundaries = gdf_boundaries.rename(columns={
                        column: 'region',
                        col_code: 'region_code'
                        })
                    success = True
                else:
                    pass
            if success is False:
                pass
                # TO DO - proper error here --------------------------------

    # Set the index:
    gdf_boundaries = gdf_boundaries.set_index(index_col)
    # Only keep geometry data:
    gdf_boundaries = gdf_boundaries[geo_cols]

    # If crs is given in the file, geopandas automatically
    # pulls it through. Convert to National Grid coordinates:
    if gdf_boundaries.crs != 'EPSG:27700':
        gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
    return gdf_boundaries


@st.cache_data
def _load_geometry_lsoa(df_lsoa):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.

    Inputs
    ------
    df_lsoa - pd.DataFrame. LSOA info.

    Returns
    -------
    gdf_boundaries_lsoa - GeoDataFrame. LSOA info and geometry.
    """

    # All LSOA shapes:
    gdf_boundaries_lsoa = import_geojson('MSOA11NM')#'LSOA11NM')
    crs = gdf_boundaries_lsoa.crs
    # Index column: LSOA11CD.
    # Always has only one unnamed column index level.
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.reset_index()
    # gdf_boundaries_lsoa = gdf_boundaries_lsoa.rename(
    #     columns={'LSOA11NM': 'lsoa', 'LSOA11CD': 'lsoa_code'})
    # gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_index(['lsoa', 'lsoa_code'])
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.rename(
        columns={'MSOA11NM': 'lsoa', 'MSOA11CD': 'lsoa_code'})
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_index(['lsoa', 'lsoa_code'])

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # Everything needs at least two levels: scenario and property.
    # Sometimes also a 'subtype' level.
    # Add another column level to the coordinates.
    col_level_names = df_lsoa.columns.names
    cols_gdf_boundaries_lsoa = [
        gdf_boundaries_lsoa.columns,                 # property
        ['any'] * len(gdf_boundaries_lsoa.columns),  # scenario
    ]
    if 'subtype' in col_level_names:
        cols_gdf_boundaries_lsoa.append([''] * len(gdf_boundaries_lsoa.columns))

    # Make all data to be combined have the same column levels.
    # Geometry:
    gdf_boundaries_lsoa = pd.DataFrame(
        gdf_boundaries_lsoa.values,
        index=gdf_boundaries_lsoa.index,
        columns=cols_gdf_boundaries_lsoa
    )

    # ----- Create final data -----
    # Merge together all of the DataFrames.
    gdf_boundaries_lsoa = pd.merge(
        gdf_boundaries_lsoa, df_lsoa,
        left_index=True, right_index=True, how='right'
    )
    # Name the column levels:
    gdf_boundaries_lsoa.columns = (
        gdf_boundaries_lsoa.columns.set_names(col_level_names))

    # Sort the results by scenario:
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.sort_index(
        axis='columns', level='scenario')

    # Convert to GeoDataFrame:
    col_geo = utils.find_multiindex_column_names(
        gdf_boundaries_lsoa, property=['geometry'])
    gdf_boundaries_lsoa = geopandas.GeoDataFrame(
        gdf_boundaries_lsoa,
        geometry=col_geo,
        crs=crs
        )
    return gdf_boundaries_lsoa



def plotly_big_map(
        gdf,
        column_colour,
        column_geometry,
        v_bands,
        v_bands_str,
        colour_map
        ):
    time_f_start = datetime.now()
    gdf = gdf.copy()
    crs = gdf.crs
    gdf = gdf.reset_index()

    # col_lsoa = utils.find_multiindex_column_names(
    #     gdf, property=['lsoa'])

    # Only keep the required columns:
    gdf = gdf[[column_colour, column_geometry]]
    # Only keep the 'property' subheading:
    gdf = pd.DataFrame(
        gdf.values,
        columns=['outcome', 'geometry']
    )
    # gdf = gdf.set_index('lsoa')
    gdf = geopandas.GeoDataFrame(gdf, geometry='geometry', crs=crs)

    # Has to be this CRS to prevent Picasso drawing:
    gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))

    # Group by outcome band.
    # Only group by non-NaN values:
    mask = ~pd.isna(gdf['outcome'])
    inds = np.digitize(gdf.loc[mask, 'outcome'], v_bands)
    # Store inds for sorting the resulting gdf:
    # gdf.loc[mask, 'inds'] = inds
    # gdf.loc[~mask, 'inds'] = np.NaN
    labels = v_bands_str[inds]
    # Flag NaN values:
    gdf.loc[mask, 'labels'] = labels
    gdf.loc[~mask, 'labels'] = 'rubbish'
    # Drop outcome column:
    gdf = gdf.drop('outcome', axis='columns')
    # Dissolve by shared outcome value:
    time_diss_start = datetime.now()
    gdf = gdf.dissolve(by='labels', sort=False)
    # from shapely import unary_union
    # ind_values = range(len(v_bands) + 1)
    # gdf_new = pd.DataFrame(#geopandas.GeoDataFrame(
    #     index=ind_values,
    #     # crs=gdf.crs
    #     )
    # gdf_new.index.names = ['inds']
    # gdf_new['labels'] = v_bands_str
    # for i in ind_values:
    #     polys = gdf.loc[gdf['inds'] == i, 'geometry']
    #     poly = unary_union(polys)
    #     gdf_new.loc[i, 'geometry'] = poly
    # gdf_new = geopandas.GeoDataFrame(gdf_new, geometry='geometry', crs=gdf.crs)
    # # gdf = gdf.set_geometry('geometry')
    # gdf_new = gdf_new.reset_index()
    # gdf = gdf_new
    time_diss_end = datetime.now()
    # st.write(f'Time to dissolve: {time_diss_end - time_diss_start}')
    gdf = gdf.reset_index()
    # Remove the NaN polygon:
    gdf = gdf[gdf['labels'] != 'rubbish']

    # Add back in the inds:
    df_inds = pd.DataFrame(
        np.array([np.arange(len(v_bands_str)), v_bands_str]).T,
        columns=['inds', 'labels']
        )
    gdf = pd.merge(gdf, df_inds, left_on='labels', right_on='labels')
    # Sort the dataframe for the sake of the legend order:
    gdf = gdf.sort_values(by='inds')

    # Simplify the polygons:
    # # simplify geometry to 1000m accuracy
    # gdf['geometry'] = (
    #     gdf.to_crs(gdf.estimate_utm_crs()).simplify(1000).to_crs(gdf.crs)
    # )
    time_f_end = datetime.now()
    # st.write(f'Time to prepare map: {time_f_end - time_f_start}')

    time_d_start = datetime.now()
    # Begin plotting.
    fig = go.Figure()

    import plotly.express as px
    fig = px.choropleth(
        gdf,
        locations=gdf.index,
        geojson=gdf.geometry.__geo_interface__,
        color=gdf['labels'],
        color_discrete_map=colour_map
        )

    fig.update_layout(
        width=800,
        height=800
        )

    # fig.add_trace(
    #     go.Choropleth(
    #         geojson=gdf.geometry.__geo_interface__,
    #         locations=gdf.index,
    #         z=gdf['mids'].astype(str),
    #         coloraxis='coloraxis',
    #     )
    # ).update_geos(fitbounds="locations", visible=False).update_layout(
    #     coloraxis={"colorscale": colour_map})

    # fig.add_trace(go.Choropleth(
    #     # gdf,
    #     geojson=gdf.geometry.__geo_interface__,
    #     locations=gdf.index,
    #     z=gdf.mids.astype(str),
    #     colorscale=colour_map,  # gdf.inds,  # pd.cut(gdf.outcome, bins=np.arange(v_min, v_max+0.11, 0.1)).astype(str),
    #     # featureidkey='properties.LSOA11NM',
    #     coloraxis="coloraxis",
    #     # colorscale='Inferno',
    #     autocolorscale=False
    # ))

    fig.update_layout(
        geo=dict(
            scope='world',
            projection=go.layout.geo.Projection(type='airy'),
            fitbounds='locations',
            visible=False
        ))
    # Remove LSOA borders:
    fig.update_traces(marker_line_width=0, selector=dict(type='choropleth'))

    # The initial colour map setting can take very many options,
    # but the later update with the drop-down menu only has a small list
    # of about ten coded in. You can't even provide a list of colours instead.
    # The available options are:
    # Blackbody, Bluered, Blues, Cividis, Earth, Electric, Greens, Greys,
    # Hot, Jet, Picnic, Portland, Rainbow, RdBu, Reds, Viridis, YlGnBu, YlOrRd.
    # As listed in: https://plotly.com/python-api-reference/generated/
    # plotly.graph_objects.Choropleth.html

    # fig.update_layout(
    #     coloraxis_colorscale='Electric',
    #     coloraxis_colorbar_title_text='Added utility',
    #     coloraxis_cmin=outcome_vmin,
    #     coloraxis_cmax=outcome_vmax,
    #     )

    # fig.update_layout(title_text='<b>Drip and Ship</b>', title_x=0.5)

    # fig.update_layout(
    #     updatemenus=[go.layout.Updatemenu(
    #         x=0, xanchor='right', y=1.15, type="dropdown",
    #         pad={'t': 5, 'r': 20, 'b': 0, 'l': 30},
    #         # ^ around all buttons (not indiv buttons)
    #         buttons=list([
    #             dict(
    #                 args=[
    #                     {
    #                         'z': [df_outcomes[
    #                             'drip_ship_lvo_mt_added_utility']],
    #                     },
    #                     {
    #                         'coloraxis.colorscale': 'Electric',
    #                         'coloraxis.reversescale': False,
    #                         'coloraxis.cmin': outcome_vmin,
    #                         'coloraxis.cmax': outcome_vmax,
    #                         'title.text': '<b>Drip and Ship</b>'
    #                     }],
    #                 label='Drip & Ship',
    #                 method='update'
    #             ),
    #             dict(
    #                 args=[
    #                     {
    #                         'z': [df_outcomes[
    #                             'mothership_lvo_mt_added_utility']],
    #                     },
    #                     {
    #                         'coloraxis.colorscale': 'Electric',
    #                         'coloraxis.reversescale': False,
    #                         'coloraxis.cmin': outcome_vmin,
    #                         'coloraxis.cmax': outcome_vmax,
    #                         'title.text': '<b>Mothership</b>'
    #                     }],
    #                 label='Mothership',
    #                 method='update'
    #             ),
    #             dict(
    #                 args=[
    #                     {
    #                         'z': [df_outcomes['diff_lvo_mt_added_utility']],
    #                     },
    #                     {
    #                         'coloraxis.colorscale': 'RdBu',
    #                         'coloraxis.reversescale': True,
    #                         'coloraxis.cmin': diff_vmin,
    #                         'coloraxis.cmax': diff_vmax,
    #                         'title.text': '<b>Difference</b>'
    #                     }],
    #                 label='Diff',
    #                 method='update'
    #             )
    #             ])
    #     )]
    # )
    fig.update_traces(hovertemplate='%{z}<extra>%{location}</extra>')

    # fig.write_html('./plotly_choro_test.html')

    st.plotly_chart(fig)
    time_d_end = datetime.now()
    # st.write(f'Time to draw map: {time_d_end - time_d_start}')


def make_colour_map_dict(v_bands_str, cmap_name='viridis'):
    # Get colour values:
    cmap = plt.get_cmap(cmap_name)
    cbands = np.linspace(0.0, 1.0, len(v_bands_str))
    colour_list = cmap(cbands)
    # # Convert from (0.0 to 1.0) to (0 to 255):
    # colour_list = (colour_list * 255.0).astype(int)
    # # Convert tuples to strings:
    # colour_list = np.array([
    #     '#%02x%02x%02x%02x' % tuple(c) for c in colour_list])
    colour_list = np.array([
        f'rgba{tuple(c)}' for c in colour_list])
    # colour_list[2] = 'red'
    # # Sample colour list:
    # lsoa_colours = colour_list[inds]
    # # Set NaN to invisible:
    # lsoa_colours[pd.isna(gdf['outcome'])] = '#00000000'
    # gdf['colour'] = lsoa_colours

    # colour_map = [[float(c), colour_list[i]] for i, c in enumerate(cbands)]
    # colour_map = [[float(c), colour_list[i]] for i, c in enumerate(midpoints)]
    colour_map = [(c, colour_list[i]) for i, c in enumerate(v_bands_str)]

    # # Set over and under colours:
    # colour_list[0] = 'black'
    # colour_list[-1] = 'LimeGreen'
    colour_map = dict(zip(v_bands_str, colour_list))
    return colour_map


def make_v_bands_str(v_bands):
    v_min = v_bands[0]
    v_max = v_bands[-1]

    v_bands_str = [f'v < {v_min:.3f}']
    for i, band in enumerate(v_bands[:-1]):
        b = f'{band:.3f} <= v < {v_bands[i+1]:.3f}'
        v_bands_str.append(b)
    v_bands_str.append(f'{v_max:.3f} <= v')

    v_bands_str = np.array(v_bands_str)
    return v_bands_str


def check_scenario_level(
        df,
        scenario_name='scenario'
        ):
    """
    Ensure DataFrame contains a column level named 'scenario'.

    Inputs
    ------
    df - pd.DataFrame. Check this for a MultiIndex column heading
         with a level named 'scenario'.
    scenario_name - str. If the 'scenario' level has to be made here,
         name the scenario this string.

    Returns
    -------
    df - pd.DataFrame. Same as the input with a 'scenario' column level.
    """
    if df is None:
        # Nothing to do here.
        return df
    else:
        pass

    # Check if 'scenario' column level exists:
    levels = df.columns.names
    if 'scenario' in levels:
        # Nothing to do here.
        return df
    else:
        if len(levels) == 1:
            # Only 'property' exists.
            # Add columns for 'scenario' below it:
            df_cols = [df.columns, [scenario_name] * len(df.columns)]
            if levels[0] is None:
                levels = ['property', 'scenario']
            else:
                levels = [levels[0]] + ['scenario']
        else:
            # Assume that a 'property' level exists and will go above
            # 'scenario', and anything else will go below it.
            df_cols_property = df.columns.get_level_values('property')
            df_cols_other = [df.columns.get_level_values(lev)
                             for lev in levels[1:]]
            df_cols = [
                df_cols_property,
                [scenario_name] * len(df.columns),
                *df_cols_other
                ]
            levels = [levels[0]] + ['scenario'] + levels[1:]

        df = pd.DataFrame(
            df.values,
            index=df.index,
            columns=df_cols
        )
        df.columns.names = levels
        return df


# ###########################
# ##### START OF SCRIPT #####
# ###########################
page_setup()

# Make containers:
# +-----------------------+
# |    container_intro    |
# +-----------------------+
# |  container_timeline   |
# +-----------------------+
# | container_shared_data |
# +-----------------------+
# | container_params_here |
# +-----------------------+
# |  container_outcomes   |
# +-----------------------+
container_intro = st.container()
container_timeline = st.container()
container_shared_data = st.container()
container_params_here = st.container()
container_outcomes = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

# Draw the input selection boxes in this function:
with st.sidebar:
    st.header('Inputs')
    input_dict = inputs.select_parameters()

with st.sidebar:
    # Outcome type input:
    outcome_type_str = st.radio(
        'Select the outcome measure',
        ['Utility', 'Added utility', 'Mean shift in mRS', 'mRS <= 2'],
        # horizontal=True
    )
    # Match the input string to the file name string:
    outcome_type_dict = {
        'Utility': 'utility',
        'Added utility': 'utility_shift',
        'Mean shift in mRS': 'mrs_shift',
        'mRS <= 2': 'mrs_0-2'
    }
    outcome_type = outcome_type_dict[outcome_type_str]

    # Scenario input:
    scenario_type_str = st.radio(
        'Select the scenario',
        ['Drip-and-ship', 'Mothership', 'MSU'],
        # horizontal=True
    )
    # Match the input string to the file name string:
    scenario_type_dict = {
        'Drip-and-ship': 'drip_ship',
        'Mothership': 'mothership',
        'MSU': 'msu'
    }
    scenario_type = scenario_type_dict[scenario_type_str]

    # Treatment type:
    treatment_type_str = st.radio(
        'Treatment type',
        ['IVT', 'MT', 'IVT & MT']
        )
    # Match the input string to the file name string:
    treatment_type_dict = {
        'IVT': 'ivt',
        'MT': 'mt',
        'IVT & MT': 'ivt_mt'
    }
    treatment_type = treatment_type_dict[treatment_type_str]

    # Stroke type:
    stroke_type_str = st.radio(
        'Stroke type',
        ['LVO', 'nLVO']
        )
    # Match the input string to the file name string:
    stroke_type_dict = {
        'LVO': 'lvo',
        'nLVO': 'nlvo',
    }
    stroke_type = stroke_type_dict[stroke_type_str]

    column_to_plot = f'{stroke_type}_{scenario_type}_{treatment_type}_{outcome_type}'
    st.write(column_to_plot)

# Feed input parameters into Scenario:
scenario = Scenario({
    'name': 1,
    'limit_to_england': False,
    **input_dict
})

# Process and save geographic data (only needed when hospital data changes)
geo = Geoprocessing(); geo.run()

# Reset index because Model expects a column named 'LSOA':
geo.combined_data = geo.combined_data.reset_index()

# Set up model
model = Model(
    scenario=scenario,
    geodata=geo.combined_data
    )

# Run model
model.run()


df_lsoa = model.full_results.copy()
df_lsoa.index.names = ['lsoa']
df_lsoa.columns.names = ['property']

# TO DO - the results df contains a mix of scenarios
# (drip and ship, mothership, msu) in the column names.
# Pull them out and put them into 'scenario' header.
# Also need to do something with separate nlvo, lvo, treatment types
# because current setup just wants some averaged added utility outcome
# rather than split by stroke type.

# Convert LSOA to MSOA:
df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
df_lsoa = df_lsoa.reset_index()
df_lsoa = pd.merge(
    df_lsoa,
    df_lsoa_to_msoa[['lsoa11nm', 'msoa11cd', 'msoa11nm']],
    left_on='lsoa', right_on='lsoa11nm', how='left'
    )
# Remove string columns:
# (temporary - I don't know how else to groupby a df with some object columns)
df_lsoa = df_lsoa.drop(['lsoa', 'nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit', 'nearest_msu_unit', 'lsoa11nm', 'msoa11nm'], axis='columns')
# Aggregate by MSOA:
df_lsoa = df_lsoa.groupby('msoa11cd').mean()
# df_lsoa = df_lsoa.set_index('msoa11cd')
# Merge the MSOA names back in and set the index to (lsoa_code, lsoa):
df_lsoa = df_lsoa.reset_index()
df_lsoa = pd.merge(
    df_lsoa, df_lsoa_to_msoa[['msoa11cd', 'msoa11nm']],
    left_on='msoa11cd', right_on='msoa11cd', how='left'
    )
# Remove duplicate rows:
df_lsoa = df_lsoa.drop_duplicates()
df_lsoa = df_lsoa.rename(columns={'msoa11cd': 'lsoa_code', 'msoa11nm': 'lsoa'})
df_lsoa = df_lsoa.set_index(['lsoa', 'lsoa_code'])

# Check whether the input DataFrames have a 'scenario' column level.
# If not, add one now with a placeholder scenario name.
df_lsoa = check_scenario_level(df_lsoa)

# Load in geometry:

# Define shared colour scales:
cbar_dict = {
    'utility': {
        'scenario': {
            'vmin': 0.0,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'inferno'
        },
        'diff': {
            'vmin': -0.3,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'RdBu'
        },
    },
    'utility_shift': {
        'scenario': {
            'vmin': 0.0,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'inferno'
        },
        'diff': {
            'vmin': -0.3,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'RdBu'
        },
    },
    'mrs_shift': {
        'scenario': {
            'vmin': 0.0,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'inferno'
        },
        'diff': {
            'vmin': -0.3,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'RdBu'
        },
    },
    'mrs_0-2': {
        'scenario': {
            'vmin': 0.0,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'inferno'
        },
        'diff': {
            'vmin': -0.3,
            'vmax': 0.3,
            'step_size': 0.05,
            'cmap_name': 'RdBu'
        },
    }
}
if scenario_type.startswith('diff'):
    scen = 'diff'
else:
    scen = 'scenario'

v_min = cbar_dict[outcome_type][scen]['vmin']
v_max = cbar_dict[outcome_type][scen]['vmax']
step_size = cbar_dict[outcome_type][scen]['step_size']
cmap_name = cbar_dict[outcome_type][scen]['cmap_name']

# Make a new column for the colours.
v_bands = np.arange(v_min, v_max + step_size, step_size)
v_bands_str = make_v_bands_str(v_bands)
colour_map = make_colour_map_dict(v_bands_str, cmap_name)


# # Load geography
# time_g_start = datetime.now()
# gdf_boundaries_lsoa = import_geojson('LSOA11NM')
# time_g_end = datetime.now()
# st.write(f'Time to load geography: {time_g_end - time_g_start}')

# Merge outcome and geography:
time_m_start = datetime.now()
gdf_boundaries_lsoa = _load_geometry_lsoa(df_lsoa)
# st.write(gdf_boundaries_lsoa)
# st.write(type(gdf_boundaries_lsoa))
# st.write('')
# st.write(gdf_boundaries_lsoa.info())
# st.write(gdf_boundaries_lsoa.crs)
time_m_end = datetime.now()
# st.write(f'Time to merge geography and outcomes: {time_m_end - time_m_start}')

# Find geometry column for plot function:
col_geo = utils.find_multiindex_column_names(
    gdf_boundaries_lsoa, property=['geometry'])
gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_geometry(col_geo)

# # Find shared colour scale limits for this outcome measure:
# if 'diff' in scenario_type:

#     cols = utils.find_multiindex_column_names(
#         gdf_boundaries_lsoa,
#         property=[outcome_type],
#         scenario=[scenario_type],
#         # subtype=['mean']
#         )
#     v_values = gdf_boundaries_lsoa[cols]
#     v_max = np.nanmax(v_values.values)
#     v_min = np.nanmin(v_values.values)
#     v_limit = max(abs(v_max), abs(v_min))
#     v_max = abs(v_limit)
#     v_min = -abs(v_limit)
# else:
#     cols = utils.find_multiindex_column_names(
#         gdf_boundaries_lsoa,
#         property=[outcome_type],
#         # subtype=['mean']
#         )
#     st.write(cols)
#     # Remove the 'diff' column:
#     cols = [c for c in cols if 'diff' not in c[1]]
#     st.write(cols)
#     v_values = gdf_boundaries_lsoa[cols]
#     st.write(v_values)
#     v_max = np.nanmax(v_values.values)
#     v_min = np.nanmin(v_values.values)

# Selected column to use for colour values:
col_col = utils.find_multiindex_column_names(
    gdf_boundaries_lsoa,
    property=[column_to_plot],
    # scenario=[scenario_type],
    # subtype=['mean']
    )

# Plot map:
time_p_start = datetime.now()
with st.spinner(text='Drawing map'):
    plotly_big_map(
        gdf_boundaries_lsoa,
        column_colour=col_col,
        column_geometry=col_geo,
        v_bands=v_bands,
        v_bands_str=v_bands_str,
        colour_map=colour_map
        )
time_p_end = datetime.now()
# st.write(f'Time to make and draw map: {time_p_end - time_p_start}')

st.stop()

# Build up the times to treatment in the different scenarios:

# Run the outcome model:

#  


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
