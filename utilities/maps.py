import streamlit as st
import pandas as pd
import numpy as np
import os
import geopandas
from shapely.validation import make_valid  # for fixing dodgy polygons

# Custom functions:
import utilities.inputs as inputs
import utilities.colour_setup


@st.cache_data
def load_lsoa_gdf():
    # Load LSOA geometry:
    path_to_lsoa = os.path.join('data', 'outline_lsoa11cds.geojson')
    gdf = geopandas.read_file(path_to_lsoa)
    gdf.index = range(len(gdf))

    # Convert to British National Grid:
    gdf = gdf.to_crs('EPSG:27700')

    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]
    return gdf


@st.cache_data
def create_colour_gdf(
        df: pd.DataFrame,
        colour_dict: dict
        ):
    """
    Main colour map creation function for Streamlit apps.

    TO DO - make column name an input instead of finding it here. --------------------

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
    # ----- Outcome maps -----
    # Left-hand subplot colours:
    df_colours = assign_colour_bands_to_areas(
        df,
        colour_dict['column'],
        colour_dict['v_bands'],
        colour_dict['v_bands_str']
        )
    # For each colour scale and data column combo,
    # merge polygons that fall into the same colour band.
    gdf = dissolve_polygons_by_value(
        df_colours.copy().reset_index()[['lsoa', 'colour_str']],
        col='colour_str'
        )
    # Map the colours to the colour names:
    gdf = assign_colour_to_areas(gdf, colour_dict['colour_map'])

    return gdf


def assign_colour_bands_to_areas(
        df: pd.DataFrame,
        col_col: str,
        v_bands: list,
        v_bands_str: list,
        ):
    """
    Assign labels to each row based on their value.

    Inputs
    ------
    df          - pd.DataFrame. Region names and outcomes.
    col_col     - str. Name of the column that contains values to
                  assign the colours to.
    v_bands     - list. The cutoff points for the colour bands.
    v_bands_str - list. Labels for the colour bands.

    Returns
    -------
    df_colours - pd.DataFrame. The outcome values from the input
                 data and the colour bands that they have been
                 assigned to.
    """
    df = df.copy()

    if col_col is None:
        # Set all shifts to zero.
        df[col_col] = 0.0

    # Only keep the required columns:
    df = df[[col_col]]

    df_colours = pd.DataFrame(
        df.values,
        columns=['outcome'],
        index=df.index
    )

    # Group by outcome band.
    # Only group by non-NaN values:
    mask = ~pd.isna(df_colours['outcome'])
    # Pick out only regions with zero exactly for the zero label.
    inds = np.digitize(df_colours.loc[mask, 'outcome'], v_bands)
    labels = v_bands_str[inds]
    df_colours.loc[mask, 'colour_str'] = labels
    # Flag NaN values:
    df_colours.loc[~mask, 'colour_str'] = 'rubbish'
    # Remove the NaN values:
    df_colours = df_colours[
        df_colours['colour_str'] != 'rubbish']

    return df_colours


@st.cache_data
def dissolve_polygons_by_value(
        df_lsoa: pd.DataFrame,
        col='colour_str',
        load_msoa=True
        ):
    """
    Merge the dataframes and then merge polygons with same value.

    With the load_msoa option, this function speeds up by
    first checking whether all LSOA within an MSOA share a value.
    If they do, the MSOA outline can be used instead of the
    separate LSOA. Fewer coordinates to munge means faster dissolve.

    Inputs
    ------
    df_lsoa   - pd.DataFrame. Contains the LSOA names and the values
                that will be used to combine areas.
    col       - str. Name of the column containing values for
                combining regions.
    load_msoa - bool. If true, speed up the dissolve by using MSOA
                instead of LSOA outlines where possible.

    Returns
    -------
    gdf - geopandas.GeoDataFrame. Contains one row per value, and
          each geometry entry is the combined regions of all areas
          in the input data that meet this value.
    """
    # Only keep columns with regions and values:
    df_lsoa = df_lsoa.reset_index()
    df_lsoa = df_lsoa[['lsoa', col]]
    if load_msoa:
        # Match LSOA to MSOA:
        df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
        df_lsoa = pd.merge(
            df_lsoa,
            df_lsoa_to_msoa[['lsoa11nm', 'msoa11cd']],
            left_on='lsoa', right_on='lsoa11nm', how='left'
            )
        # Split off just the MSOA and the colour bands:
        df_msoa_values = df_lsoa[['msoa11cd', col]].copy()
        # Remove duplicates:
        df_msoa_values = df_msoa_values.drop_duplicates()

        # Which MSOA appear multiple times?
        mask_msoa_multi = df_msoa_values['msoa11cd'].duplicated(keep=False)

        # Pick out MSOA that appear only once in this list.
        # These can be used directly.
        selected_msoa = df_msoa_values[~mask_msoa_multi]
        # Load MSOA geometry:
        path_to_msoa = os.path.join('data', 'outline_msoa11cds.geojson')
        gdf_msoa = geopandas.read_file(path_to_msoa)
        # Convert to British National Grid:
        gdf_msoa = gdf_msoa.to_crs('EPSG:27700')
        # Columns: MSOA11CD, MSOA11NM, geometry.
        # Limit to only selected MSOA:
        gdf_msoa = pd.merge(
            gdf_msoa, selected_msoa,
            left_on='MSOA11CD', right_on='msoa11cd', how='right'
            )
        gdf_msoa = gdf_msoa[['geometry', col]]

        # Pick out MSOA that appear more than once in the value list.
        # These must be replaced with their constituent LSOA.
        mask_lsoa = df_lsoa['msoa11cd'].isin(
            df_msoa_values.loc[mask_msoa_multi, 'msoa11cd'])
        # Find the LSOA that go into these MSOA:
        selected_lsoa = df_lsoa[mask_lsoa]
        # Load LSOA geometry:
        gdf_lsoa = load_lsoa_gdf()
        # Columns: MSOA11CD, MSOA11NM, geometry.
        # Limit to only selected MSOA:
        gdf_lsoa = pd.merge(
            gdf_lsoa, selected_lsoa,
            left_on='LSOA11NM', right_on='lsoa', how='right'
            )
        gdf_lsoa = gdf_lsoa[['geometry', col]]

        # Stack selected geometry:
        gdf = pd.concat((gdf_msoa, gdf_lsoa))
        gdf.index = range(len(gdf))
    else:
        # Load LSOA geometry:
        gdf = load_lsoa_gdf()
        # Merge in column:
        gdf = pd.merge(gdf, df_lsoa,
                       left_on='LSOA11NM', right_on='lsoa', how='right')
        gdf.index = range(len(gdf))

    # Convert to British National Grid:
    gdf = gdf.to_crs('EPSG:27700')

    # Make geometry valid:
    gdf['geometry'] = [
        make_valid(g) if g is not None else g
        for g in gdf['geometry'].values
        ]

    # Dissolve by value:
    # I have no idea why, but using sort=False in the following line
    # gives unexpected results in the map. e.g. areas that the data
    # says should be exactly zero will show up as other colours.
    # Maybe filling in holes in geometry? Maybe incorrect sorting?
    gdf = gdf.dissolve(by=col)
    gdf = gdf.reset_index()
    # Remove the NaN polygon:
    gdf = gdf[gdf[col] != 'rubbish']

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

