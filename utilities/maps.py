import streamlit as st
import pandas as pd
import numpy as np
import os
import geopandas
import pyproj  # for crs conversion
from shapely.validation import make_valid  # for fixing dodgy polygons

# Custom functions:
import utilities.calculations as calc
import utilities.container_inputs as inputs
import utilities.utils as utils
# For setting up maps:
from stroke_maps.geo import import_geojson, check_scenario_level


@st.cache_data
def _import_geojson(*args, **kwargs):
    """Wrapper for stroke-maps import_geojson so cache_data used."""
    return import_geojson(*args, **kwargs)


@st.cache_data
def _load_geometry_msoa(df_msoa: pd.DataFrame):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.

    TO DO - why is this here and not just using the one from stroke-maps package? --------------------------------------------------

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
def combine_geography_with_outcomes(df_lsoa: pd.DataFrame):
    """
    Main function for building geometry for each area.

    Inputs
    ------
    df_lsoa - pd.DataFrame. Should contain at least LSOA codes.

    Returns
    -------
    gdf_boundaries_msoa - geopandas.GeoDataFrame. Geometry for each
                          MSOA with df_msoa contents merged in.
    df_msoa             - pd.DataFrame. Just the MSOA data without
                          geometry. (TO DO - why have I done this??? -------------------------------------------------)
    """
    # ----- MSOAs for geography -----
    df_msoa = calc.convert_lsoa_to_msoa_results(df_lsoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = _load_geometry_msoa(df_msoa)
    return gdf_boundaries_msoa, df_msoa


@st.cache_data
def create_colour_gdf(
        _gdf_boundaries_msoa: geopandas.GeoDataFrame,
        df_msoa: pd.DataFrame,
        scenario_dict: dict,
        scenario_type: str,
        cmap_name: str = '',
        cmap_diff_name: str = '',
        cbar_title: str = '',
        ):
    """
    Main colour map creation function for Streamlit apps.

    Inputs
    ------
    _gdf_boundaries_msoa - geopandas.GeoDataFrame. Geometry info for
                           each MSOA (with outcome data merged in).
    df_msoa              - pd.DataFrame. Main results for each MSOA.
    scenario_dict        - dict. User inputs such as occlusion and
                           treatment type.
    cmap_name            - str. Name of the colourmap for assigning
                           colours, e.g. 'viridis'.
    cbar_title           - str. Label that will be displayed with
                           the colourbar.
    scenario_type        - str. Starts with either 'diff' or anything
                           else. Used to pick out either the continuous
                           or the diverging colourmap.

    Returns
    -------
    gdf         - geopandas.GeoDataFrame. Contains one entry per
                  colour band in the colour dict so long as at least
                  one area in the input data matched that colour band.
                  The geometry is now merged together into one large
                  area rather than multiple separate MSOA of the
                  same value.
    colour_dict - dict. The information used to set up the colours.
    """
    # ----- Colour setup -----
    # Give the scenario dict a dummy 'scenario_type' entry
    # so that the right colour map and colour limits are picked.
    colour_dict = inputs.set_up_colours(
        scenario_dict | {'scenario_type': scenario_type},
        cmap_name=cmap_name,
        cmap_diff_name=cmap_diff_name
        )
    # Pull down colourbar titles from earlier in this script:
    colour_dict['title'] = cbar_title
    # Find the names of the columns that contain the data
    # that will be shown in the colour maps.
    column_colours = '_'.join([
            scenario_dict['stroke_type'],
            scenario_type,
            scenario_dict['treatment_type'],
            scenario_dict['outcome_type']
        ])
    colour_dict['column'] = column_colours

    # ----- Outcome maps -----
    # Left-hand subplot colours:
    df_msoa_colours = assign_colour_bands_to_areas(
        df_msoa,
        colour_dict['column'],
        colour_dict['v_bands'],
        colour_dict['v_bands_str']
        )
    # For each colour scale and data column combo,
    # merge polygons that fall into the same colour band.
    gdf = dissolve_polygons_by_colour(
        _gdf_boundaries_msoa,
        df_msoa_colours
        )
    # Map the colours to the colour names:
    gdf = assign_colour_to_areas(
        gdf,
        colour_dict['colour_map']
    )

    return gdf, colour_dict


def assign_colour_bands_to_areas(
        df_msoa: pd.DataFrame,
        col_col: str,
        v_bands: list,
        v_bands_str: list,
        ):
    """
    Assign labels to each row based on their value.

    Inputs
    ------
    df_msoa     - pd.DataFrame. MSOA names and outcomes.
    col_col     - str. Name of the column that contains values to
                  assign the colours to.
    v_bands     - list. The cutoff points for the colour bands.
    v_bands_str - list. Labels for the colour bands.

    Returns
    -------
    df_msoa_colours - pd.DataFrame. The outcome values from the input
                      data and the colour bands that they have been
                      assigned to.
    """
    df_msoa = df_msoa.copy()

    # Selected column to use for colour values:
    column_colour = utils.find_multiindex_column_names(
        df_msoa,
        property=[col_col],
        )

    # Only keep the required columns:
    df_msoa = df_msoa[[column_colour]]

    df_msoa_colours = pd.DataFrame(
        df_msoa.values,
        columns=['outcome'],
        index=df_msoa.index
    )

    # Group by outcome band.
    # Only group by non-NaN values:
    mask = ~pd.isna(df_msoa_colours['outcome'])
    # Pick out only regions with zero exactly for the zero label.
    inds = np.digitize(df_msoa_colours.loc[mask, 'outcome'], v_bands)
    labels = v_bands_str[inds]
    df_msoa_colours.loc[mask, 'colour_str'] = labels
    # Flag NaN values:
    df_msoa_colours.loc[~mask, 'colour_str'] = 'rubbish'
    # Remove the NaN values:
    df_msoa_colours = df_msoa_colours[
        df_msoa_colours['colour_str'] != 'rubbish']

    return df_msoa_colours


def dissolve_polygons_by_colour(
        gdf_all: geopandas.GeoDataFrame,
        df_msoa: pd.DataFrame,
        ):
    """
    Merge the dataframes and then merge polygons with same value.

    TO DO - split this function apart more. What's with all the column
    lookup and dataframe merging?

    Inputs
    ------
    gdf_all - geopandas.GeoDataFrame. Contains geometry for each MSOA
              separately.
    df_msoa - pd.DataFrame. Contains the colour bands that each MSOA
              has been assigned to.
    """

    # Merge in the colour information:
    gdf = gdf_all.copy()
    crs = gdf.crs
    gdf = gdf.reset_index()

    gdf[('msoa_code', 'scenario')] = gdf[('msoa_code', '')]
    gdf = gdf.drop([('msoa_code', '')], axis='columns')

    df_msoa = df_msoa.reset_index()
    df_msoa = df_msoa.drop('msoa', axis='columns')
    # Give an extra column level to df_msoa:
    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)#, scenario_name='')

    col_msoa_msoa = utils.find_multiindex_column_names(
        df_msoa, property=['msoa_code'])

    column_msoa = utils.find_multiindex_column_names(
        gdf, property=['msoa_code'])

    gdf = pd.merge(
        gdf,
        df_msoa,
        left_on=[column_msoa],
        right_on=[col_msoa_msoa],
        how='right'
        )

    # Find geometry column for plot function:
    column_geometry = utils.find_multiindex_column_names(
        gdf, property=['geometry'])

    # Selected column to use for colour values:
    column_colour_str = utils.find_multiindex_column_names(
        gdf,
        property=['colour_str'],
        # scenario=[scenario_type],
        # subtype=['mean']
        )

    # Only keep the required columns:
    gdf = gdf[[column_colour_str, column_geometry]]
    # Only keep the 'property' subheading:
    gdf = pd.DataFrame(
        gdf.values,
        columns=['colour_str', 'geometry']
    )
    # gdf['iszero'] = False
    gdf = geopandas.GeoDataFrame(gdf, geometry='geometry', crs=crs)

    # Has to be this CRS to prevent Picasso drawing:
    # gdf = gdf.to_crs(pyproj.CRS.from_epsg(4326))

    # Dissolve by shared outcome value:
    # I have no idea why, but using sort=False in the following line
    # gives unexpected results in the map. e.g. areas that the data
    # says should be exactly zero will show up as other colours.
    # Maybe filling in holes in geometry? Maybe incorrect sorting?
    gdf = gdf.dissolve(by='colour_str')#, sort=False)
    gdf = gdf.reset_index()
    # Remove the NaN polygon:
    gdf = gdf[gdf['colour_str'] != 'rubbish']

    # # # Simplify the polygons:
    # # For Picasso mode.
    # # Simplify geometry to 10000m accuracy
    # gdf['geometry'] = (
    #     gdf.to_crs(gdf.estimate_utm_crs()).simplify(10000).to_crs(gdf.crs)
    # )
    return gdf


def assign_colour_to_areas(
        df: pd.DataFrame,
        colour_dict: dict,
        ):
    """
    Map colours to a label column using a dictionary.

    Inputs
    ------
    df          - pd.DataFrame. Contains a 'colour_str' column that
                  labels each row as one of the keys of colour_dict.
    colour_dict - dict. Keys are colour labels, values are the colours.

    Returns
    -------
    df - pd.DataFrame. The input data with the new column of colours.
    """
    df['colour'] = df['colour_str'].map(colour_dict)
    return df


@st.cache_data
def find_geometry_ivt_catchment(gdf_boundaries_msoa: geopandas.GeoDataFrame):
    """
    Merge geometry for all polygons with a shared value (stroke unit).

    TO DO - the input to this function needs fixing - current setup
    means that MSOA can be counted for multiple nearest units if
    their constituent LSOAs are counted for multiple nearest units.

    TO DO also - shouldn't this use a more generic dissolve function?
    Combine with the other dissolve by colour function. --------------------------------

    Inputs
    ------
    gdf_boundaries_msoa - geopandas.GeoDataFrame. Contains geometry
                          and some value to merge shapes by.

    Returns
    -------
    gdf_catchment - geopandas.GeoDataFrame. A new dataframe of the
                    merged geometry data. Its index is the values that
                    were globbed together (e.g. nearest stroke unit).
    """
    gdf_catchment = pd.DataFrame()
    gdf_catchment['nearest_ivt_unit'] = gdf_boundaries_msoa[
        (('nearest_ivt_unit', 'scenario'))]
    gdf_catchment['geometry'] = gdf_boundaries_msoa[(('geometry', 'any'))]
    gdf_catchment = geopandas.GeoDataFrame(gdf_catchment, geometry='geometry')
    gdf_catchment = gdf_catchment.dissolve(by='nearest_ivt_unit')
    return gdf_catchment


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
                    else:
                        # Multipolygon.
                        # Put None values between polygons.
                        for poly in t.geoms:
                            x, y = poly.exterior.coords.xy
                            x_combo += list(x) + [None]
                            y_combo += list(y) + [None]
                x_list.append(np.array(x_combo))
                y_list.append(np.array(y_combo))
            else:
                st.write('help', i)  # TO DO - turn this into a proper Warning or Exception -----------------------
        except AttributeError:
            # This isn't a geometry object. ???
            x_list.append([]),
            y_list.append([])
    return x_list, y_list
