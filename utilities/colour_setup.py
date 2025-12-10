"""
Set up colour bars for maps.
"""
import streamlit as st
import numpy as np
import os
import pandas as pd
from matplotlib.pyplot import get_cmap
import cmasher as cmr  # for extra cmaps in plt.get_cmap()

from utilities.utils import set_rerun_map


def load_colour_limits(outcome: str):
    """
    Load in colour limits for outcome maps from file.

    Inputs
    ------
    outcome - str. Name of the outcome type to load limits for.
              Either 'utility_shift' or 'mrs_0-2'.

    Returns
    -------
    dict_colours      - dict. Contains vmin and vmax for single
                        outcome map.
    dict_colours_diff - dict. Contains vmin and vmax for difference
                        outcome map.
    """
    df = pd.read_csv(os.path.join('utilities', 'colour_limits.csv'),
                     index_col=0)

    # Column names for this outcome measure:
    col = outcome
    col_diff = f'{outcome}_diff'

    # Colour limit dictionaries:
    dict_colours = df[col].to_dict()
    dict_colours_diff = df[col_diff].to_dict()

    return dict_colours, dict_colours_diff


def make_colourbar_display_string(
        cmap_name: str, char_line: str = '█', n_lines: int = 20):
    """
    Make a long string with LaTeX-formatted colours from a cmap.

    e.g. have a string of these blocks ████████████████████ with
    each block a different colour. Sample the colours from the
    colour map.

    Inputs
    ------
    cmap_name - str. Name of the plt or cmasher colour map to draw.
    char_line - str. The line or block character to draw, e.g. █
    n_lines   - int. How many characters to draw.

    Returns
    -------
    line_str - str. A string of length n_lines containing the char_line
               character in a uniform sampling of colours from
               cmap_name.
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


def make_colour_list(
        cmap_name: str = 'viridis',
        n_colours: int = 101,
        remove_white: bool = False,
        vmax: float = 1.0,
        vmin: float = -1.0
        ):
    """
    Pick out a list of rgba strings from part of a colour map.

    The sanity checks here work best for diverging colour maps.
    This function checks the relative values of vmin and vmax to
    check whether to use: the full cmap; only the left side or only
    the right side; or a portion of the cmap where the central value
    of the full cmap is not the central value of the portion.
    e.g. full cmap: █▓▒░░▒▓█, portion ▒░░▒▓██

    Inputs
    ------
    cmap_name    - str. Name of colour map to sample. Can use any
                   pyplot or cmasher name. Diverging cmaps work best.
    n_colours    - int. How many colours to sample.
    remove_white - bool. Whether to remove pure white colour from
                   resulting list. Had some unexpected plotly
                   behaviour with white while testing.
    vmax         - float. Maximum value in the plotting range.
    vmin         - float. Minimum value in the plotting range.

    Returns
    -------
    colour_list - list. List of rgba() colour strings from the sampled
                  portion of the cmap for cmap_name.
    """
    # Get colour values:
    try:
        # Matplotlib colourmap:
        cmap = get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = get_cmap(f'cmr.{cmap_name}')

    # Work out which part of the colour map to sample.
    # If vmax is +ve and vmin is -ve, use both halves of the diverging
    # colourmap. If both are +ve, use only the top half.
    if vmin == -vmax:
        # █▓▒░░▒▓█ Sample the full diverging colourmap.
        bmin = 0.0
        bmax = 1.0
    elif np.sign(vmin) == -np.sign(vmax):
        # vmin is -ve, vmax is +ve, but zero is not halfway.
        if np.abs(vmax) > np.abs(vmin):
            # ▒░░▒▓██ Use the full right-hand-side of the cmap
            # and part of the left-hand-side.
            bmax = 1.0
            bmin = 0.5 * (1.0 - (np.abs(vmin) / np.abs(vmax)))
        else:
            # ██▓▒░░▒ Use the full left-hand-side of the cmap
            # and part of the right-hand-side.
            bmin = 0.0
            bmax = 0.5 * (1.0 + (np.abs(vmax) / np.abs(vmin)))
    elif np.sign(vmax) == 0:
        # ██▓▓▒▒░░ Max value is zero. Use only left-hand-side of cmap.
        bmin = 0.0
        bmax = 0.5
    elif np.sign(vmin) == 0:
        # ░░▒▒▓▓██ Min value is zero. Use only right-hand-side of cmap.
        bmin = 0.5
        bmax = 1.0
    elif np.sign(vmax) == -1:
        # ██▓▓▒▒░░ Both vmin and vmax are -ve.
        # Use the left half of the cmap:
        bmin = 0.0
        bmax = 0.5
    elif np.sign(vmax) == 1:
        # ░░▒▒▓▓██ Both vmin and vmax are +ve.
        # Use the right half of the cmap:
        bmin = 0.5
        bmax = 1.0
    else:
        # This shouldn't happen. Use the full cmap.
        bmin = 0.0
        bmax = 1.0

    cbands = np.linspace(bmin, bmax, n_colours)
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


def make_colour_list_for_plotly_button(*args, **kwargs):
    """
    Wrapper for make_colour_list() for plotly restyle buttons.

    This is used for figures that contain buttons that allow the
    colour scheme to be changed, for example the outcome maps.
    Lots of extra formatting is needed to make the colour list
    understandable to the plotly restyle function.

    For inputs, see make_colour_list().

    Returns
    -------
    c - str. The colour list as a single string in the format that
        plotly restyle can understand.
    """
    c = make_colour_list(*args, **kwargs)

    # Convert this into the very specific string format
    # that plotly needs for restyle colourscale buttons.
    c = np.array([np.linspace(0, 1, len(c)), c], dtype=object).T
    c = str(c)
    # Swap single for double quotes:
    c = c.replace("'", '"')
    # Remove alpha:
    c = c.replace(', 255)', ')')
    c = c.replace('rgba', 'rgb')
    # Sort out comma placement:
    c = c.replace(' "rgb', ', "rgb')
    c = c.replace(']', '],').replace('],]', ']]').replace(']],', ']]')
    # Remove newlines:
    c = c.replace('\n', '')
    return c


def select_colour_maps():
    """
    User input for which colour map to use on outcome maps.

    Returns
    -------
    cmap_selected - str. The chosen colour map name.
    cmap_names    - list. The cmap options given. Used later to draw
                    restyle buttons on outcome maps.
    """
    cmap_names = ['iceburn_r', 'seaweed', 'fusion', 'waterlily']
    # Add the reverse option after each entry. Remove any double
    # reverse reverse _r_r. Result is flat list.
    cmap_names = sum(
        [[c, (c + '_r').replace('_r_r', '')] for c in cmap_names], [])

    cmap_displays = [
        make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
        for cmap_name in cmap_names
        ]

    try:
        cmap_name = st.session_state['cmap_diff_name']
    except KeyError:
        cmap_name = cmap_names[0]
    cmap_ind = cmap_names.index(cmap_name)

    cmap_selected = st.radio(
        'Default colour display for maps',
        cmap_names,
        captions=cmap_displays,
        index=cmap_ind,
        key='cmap_diff_name',
        horizontal=True
    )

    return cmap_selected, cmap_names


def select_colour_limits(
        map_outcome: str,
        vlim_dict: dict,
        scenario_name: str = 'redir',
        scenario_label: str = 'redirection'
        ):
    """
    Draw editable dataframe of limits of actual and plotted data.

    The table has a row for minimum and a row for maximum values.
    The first column prints the actual limits in the data.
    The second editable column is the limits drawn on the
    outcome maps.

           +------+---------+
           | Data | Colours |
    +------+------+---------+
    | Min. |   X  |    ed.  |
    +------+------+---------+
    | Max. |   X  |    ed.  |
    +------+------+---------+

    Inputs
    ------
    map_outcome    - str. The outcome type being shown on the maps.
    vlim_dict      - dict. Contains an entry for each map with the
                     min and max values in the data.
    scenario_name  - str. The key for the not-usual-care scenario.
    scenario_label - str. The label for the not-usual-care scenario.

    Returns
    -------
    d - dict. For each map, the min and max values in the data and the
        user-selected min and max values to plot.
    """
    outcome_label_dict = {
        'utility_shift': 'Utility shift',
        'mrs_0-2': 'mRS <= 2',
        }
    map_outcome_label = outcome_label_dict[map_outcome]

    d = {}
    d['usual_care'] = {
        'title': f'Usual care: {map_outcome_label}',
        }
    diff_name = f'{scenario_name}_minus_usual_care'
    d[diff_name] = {
        'title': f'''
        Benefit of {scenario_label} over usual care:
        {map_outcome_label}''',
        }
    d['pop'] = {
        'title': 'Population density (people per square kilometre)',
        }

    # Default colour limits:
    dict_colours, dict_colours_diff = load_colour_limits(map_outcome)
    d['usual_care']['vmin'] = dict_colours['vmin']
    d['usual_care']['vmax'] = dict_colours['vmax']
    d[diff_name]['vmin'] = dict_colours_diff['vmin']
    d[diff_name]['vmax'] = dict_colours_diff['vmax']
    d['pop']['vmin'] = 0.0
    d['pop']['vmax'] = 100.0

    # Copy over data min/max values:
    for c in d.keys():
        d[c] = d[c] | vlim_dict[c]

    # Set up display:
    st.markdown('__Edit the limits of the colour scales:__')
    i = 0
    cols = st.columns(3)
    for c, c_dict in d.items():
        arr = np.array([
            [c_dict['data_min'], c_dict['vmin']],
            [c_dict['data_max'], c_dict['vmax']]
            ])
        df = pd.DataFrame(arr, columns=['Map data', 'Colour scale'],
                          index=['Minimum', 'Maximum'])
        with cols[i]:
            st.markdown(c_dict['title'])
            df = st.data_editor(
                df, disabled=['Map data'],
                key=f'{c}_colour_setup',
                on_change=set_rerun_map
                )
        d[c]['vmin'] = df.loc['Minimum', 'Colour scale']
        d[c]['vmax'] = df.loc['Maximum', 'Colour scale']
        i += 1

    return d
