"""
Set up colour bars for maps.
"""
import numpy as np
import os
import pandas as pd
from matplotlib.pyplot import get_cmap


def load_colour_limits(outcome):
    """
    Load in colour limits from file.

    Dictionary contents: vmin, vmax, step_size.
    """
    df = pd.read_csv(os.path.join('utilities', 'colour_limits.csv'), index_col=0)

    # Column names for this outcome measure:
    col = outcome
    col_diff = f'{outcome}_diff'

    # Colour limit dictionaries:
    dict_colours = df[col].to_dict()
    dict_colours_diff = df[col_diff].to_dict()

    return dict_colours, dict_colours_diff


def make_colourbar_display_string(cmap_name, char_line='█', n_lines=20):
    """
    Make a long string with LaTeX-formatted colours from a cmap.

    e.g. have a string of these blocks ████████████████████ with
    each block a different colour. Sample the colours from the
    colour map.
    """
    try:
        # Matplotlib colourmap:
        cmap = get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = get_cmap(f'cmr.{cmap_name}')

    # Get colours:
    colours = cmap(np.linspace(0.0, 1.0, n_lines))
    # Convert tuples to strings:
    colours = (colours * 255).astype(int)
    # Drop the alpha or the colour won't display right!
    colours = ['#%02x%02x%02x' % tuple(c[:-1]) for c in colours]

    line_str = '$'
    for c in colours:
        # s = f"<font color='{c}'>{char_line}</font>"
        s = '\\textcolor{' + f'{c}' + '}{' + f'{char_line}' + '}'
        line_str += s
    line_str += '$'
    return line_str


def make_colour_list(cmap_name='viridis', n_colours=101, remove_white=True):
    """
    Pick out a list of rgba strings from a colour map.
    """
    # Get colour values:
    try:
        # Matplotlib colourmap:
        cmap = get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = get_cmap(f'cmr.{cmap_name}')

    cbands = np.linspace(0.0, 1.0, n_colours)
    colour_list = cmap(cbands)
    # # Convert tuples to strings:
    # Use format_float_positional to stop tiny floats being printed
    # with scientific notation.
    colour_list = np.array([
        'rgba(' +
        # ','.join([f'{np.format_float_positional(c1, precision=100)}'
        ', '.join([f'{int(np.round(c1 * 255, 0))}'
                  for c1 in c]) +
        ')' for c in colour_list
        ])

    if remove_white:
        # Plotly doesn't seem to handle white well so remove it:
        colour_list = [c for c in colour_list if c != 'rgba(1.,1.,1.,1.)']
    return colour_list


# #####################################
# ##### CONTOUR MAPS COLOUR SETUP #####
# #####################################

def make_contour_edge_values(v_min, v_max, step_size):
    # Make a new column for the colours.
    v_bands = np.arange(v_min, v_max + step_size, step_size)

    # If there are negative and positive values,
    # make an extra contour around zero that's really small.
    if ((np.sign(v_min) == -1) & (np.sign(v_max) == 1)):
        # Remove existing zero if it exists:
        ind_z = np.where(abs(v_bands) < step_size * 0.01)[0]
        if len(ind_z) > 0:
            ind_z = ind_z[0]
            v_bands = np.append(v_bands[:ind_z], v_bands[ind_z+1:])
        # Add a zero-ish band.
        ind = np.where(v_bands >= -0.0)[0][0]
        zero_size = step_size * 0.01
        v_bands_z = np.append(v_bands[:ind], [-zero_size, zero_size])
        v_bands_z = np.append(v_bands_z, v_bands[ind:])
        v_bands = v_bands_z
    else:
        pass
    return v_bands


def make_v_bands_str(v_bands, v_name='v'):
    """Turn contour ranges into formatted strings."""
    v_min = v_bands[0]
    v_max = v_bands[-1]

    v_bands_str = [f'{v_name} < {v_min:.3f}']
    for i, band in enumerate(v_bands[:-1]):
        if band != v_bands[i+1]:
            b = f'{band:.3f} <= {v_name} < {v_bands[i+1]:.3f}'
        else:
            # Update zeroish name:
            b = f'{band:.3f}'
        v_bands_str.append(b)
    v_bands_str.append(f'{v_max:.3f} <= {v_name}')

    v_bands_str = np.array(v_bands_str)
    return v_bands_str


def make_colour_map_dict(v_bands_str, cmap_name='viridis'):
    # Get colour values:
    colour_list = make_colour_list(
        cmap_name, n_colours=len(v_bands_str), remove_white=False)

    # # Sample the colour list:
    # colour_map = [(c, colour_list[i]) for i, c in enumerate(v_bands_str)]

    # # # Set over and under colours:
    # # colour_list[0] = 'black'
    # # colour_list[-1] = 'LimeGreen'

    # Return as dict to track which colours are for which bands:
    colour_map = dict(zip(v_bands_str, colour_list))
    return colour_map


def add_infinity_bounds(v_bands, v_min, v_max, step_size):
    # Add an extra bound at either end (for the "to infinity" bit):
    v_bands_for_cs = np.append(v_min - step_size, v_bands)
    v_bands_for_cs = np.append(v_bands_for_cs, v_max + step_size)
    return v_bands_for_cs


def normalise_bounds(v_bands_for_cs):
    # Normalise the data bounds:
    bounds = (
        (np.array(v_bands_for_cs) - np.min(v_bands_for_cs)) /
        (np.max(v_bands_for_cs) - np.min(v_bands_for_cs))
    )
    # Need separate data values and colourbar values.
    # e.g. translate 32 in the data means colour 0.76 on the colourmap.

    return bounds


def create_colour_scale_for_plotly(colours, bounds_for_cs):
    # Create a colour scale from these colours.
    # To get the discrete colourmap (i.e. no continuous gradient of
    # colour made between the defined colours),
    # double up the bounds so that colour A explicitly ends where
    # colour B starts.
    colourscale = []
    for i in range(len(colours)):
        colourscale += [
            [bounds_for_cs[i], colours[i]],
            [bounds_for_cs[i+1], colours[i]]
            ]
    return colourscale
