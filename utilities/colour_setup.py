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
        ','.join([f'{np.format_float_positional(c1, precision=100)}'
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

def set_up_colours(
        scenario_dict,
        v_name='v',
        cmap_name='inferno',
        cmap_diff_name='RdBu'
        ):
    """
    max ever displayed:

    utility:
    max times: > 0.300,

    utility shift:
    min times: 0.100 < 0.150, 0.150 < 0.200, 0.200 < 0.250,
    max times: <0.000, 0.000 - < 0.050, 0.050 < 0.100,

    mrs shift:
    min times: <0.000,
    max times: <0.000, 0.000 - < 0.050, 0.050 < 0.100,

    mrs 0-2:
    min times: 0.250 - 0.0300, > 0.300,
    max times: 0.250 - 0.300, > 0.300


    colour scales sometimes bug out, return to default colourbar
    when the precision here isn't enough decimal places.
    """
    # Define shared colour scales:
    cbar_dict = {
        'utility': {
            'scenario': {
                'vmin': 0.3,
                'vmax': 0.6,
                'step_size': 0.05,
                'cmap_name': cmap_name
            },
            'diff': {
                'vmin': -0.05,
                'vmax': 0.05,
                'step_size': 0.01,
                'cmap_name': cmap_diff_name
            },
        },
        'utility_shift': {
            'scenario': {
                'vmin': 0.0,
                'vmax': 0.15,
                'step_size': 0.025,
                'cmap_name': cmap_name
            },
            'diff': {
                'vmin': -0.040,
                'vmax': 0.040,
                'step_size': 0.010,
                'cmap_name': cmap_diff_name
            },
        },
        'mrs_shift': {
            'scenario': {
                'vmin': -0.5,
                'vmax': 0.0,
                'step_size': 0.1,
                'cmap_name': f'{cmap_name}_r'  # lower numbers are better
            },
            'diff': {
                'vmin': -0.2,
                'vmax': 0.2,
                'step_size': 0.05,
                'cmap_name': f'{cmap_diff_name}_r'  # lower numbers are better
            },
        },
        'mrs_0-2': {
            'scenario': {
                'vmin': 0.30,
                'vmax': 0.70,
                'step_size': 0.05,
                'cmap_name': cmap_name
            },
            'diff': {
                'vmin': -0.15,
                'vmax': 0.15,
                'step_size': 0.05,
                'cmap_name': cmap_diff_name
            },
        }
    }
    if scenario_dict['scenario_type'].startswith('diff'):
        scen = 'diff'
    else:
        scen = 'scenario'

    v_min = cbar_dict[scenario_dict['outcome_type']][scen]['vmin']
    v_max = cbar_dict[scenario_dict['outcome_type']][scen]['vmax']
    step_size = cbar_dict[scenario_dict['outcome_type']][scen]['step_size']
    cmap_name = cbar_dict[scenario_dict['outcome_type']][scen]['cmap_name']

    if cmap_name.endswith('_r_r'):
        # Remove the double reverse reverse.
        cmap_name = cmap_name[:-2]

    # Make a new column for the colours.
    v_bands = np.arange(v_min, v_max + step_size, step_size)
    if 'diff' in scen:
        # Remove existing zero:
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
        v_bands_str = make_v_bands_str(v_bands, v_name=v_name)

        # Update zeroish name:
        v_bands_str[ind+1] = '0.0'
    else:
        v_bands_str = make_v_bands_str(v_bands, v_name=v_name)

    colour_map = make_colour_map_dict(v_bands_str, cmap_name)

    # Link bands to colours via v_bands_str:
    colours = []
    for v in v_bands_str:
        colours.append(colour_map[v])

    # Add an extra bound at either end (for the "to infinity" bit):
    v_bands_for_cs = np.append(v_min - step_size, v_bands)
    v_bands_for_cs = np.append(v_bands_for_cs, v_max + step_size)
    # Normalise the data bounds:
    bounds = (
        (np.array(v_bands_for_cs) - np.min(v_bands_for_cs)) /
        (np.max(v_bands_for_cs) - np.min(v_bands_for_cs))
    )
    # Add extra bounds so that there's a tiny space at either end
    # for the under/over colours.
    # bounds_for_cs = [bounds[0], bounds[0] + 1e-7, *bounds[1:-1], bounds[-1] - 1e-7, bounds[-1]]
    bounds_for_cs = bounds

    # Need separate data values and colourbar values.
    # e.g. translate 32 in the data means colour 0.76 on the colourmap.

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

    colour_dict = {
        'scen': scen,
        'v_min': v_min,
        'v_max': v_max,
        'step_size': step_size,
        'cmap_name': cmap_name,
        'v_bands': v_bands,
        'v_bands_str': v_bands_str,
        'colour_map': colour_map,
        'colour_scale': colourscale,
        'bounds_for_colour_scale': bounds_for_cs,
        # 'zero_label': '0.0',
        # 'zero_colour':
    }
    return colour_dict


def make_colour_map_dict(v_bands_str, cmap_name='viridis'):
    # Get colour values:
    colour_list = make_colour_list(
        cmap_name, n_colours=len(v_bands_str), remove_white=False)

    # Sample the colour list:
    colour_map = [(c, colour_list[i]) for i, c in enumerate(v_bands_str)]

    # # Set over and under colours:
    # colour_list[0] = 'black'
    # colour_list[-1] = 'LimeGreen'

    # Return as dict to track which colours are for which bands:
    colour_map = dict(zip(v_bands_str, colour_list))
    return colour_map


def make_v_bands_str(v_bands, v_name='v'):
    """Turn contour ranges into formatted strings."""
    v_min = v_bands[0]
    v_max = v_bands[-1]

    v_bands_str = [f'{v_name} < {v_min:.3f}']
    for i, band in enumerate(v_bands[:-1]):
        b = f'{band:.3f} <= {v_name} < {v_bands[i+1]:.3f}'
        v_bands_str.append(b)
    v_bands_str.append(f'{v_max:.3f} <= {v_name}')

    v_bands_str = np.array(v_bands_str)
    return v_bands_str
