"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for colour maps

def select_parameters():
    # Set up scenarios
    scenarios = {
        'process_time_call_ambulance': {
            'values': [0, 60, 120, 180],
            'name': 'Time to call ambulance',
            'default': 60  # 1  # index for 60
        },
        'process_time_ambulance_response': {
            'values': [15, 30, 45],
            'name': 'Ambulance response time',
            'default': 30  # 1  # index for 30
        },
        'process_ambulance_on_scene_duration': {
            'values': [20, 30, 40],
            'name': 'Time ambulance is on scene',
            'default': 20  # 0  # index for 20
        },
        'process_msu_dispatch': {
            'values': [0, 15, 30],
            'name': 'MSU dispatch time',
            'default': 15  # 1  # index for 15
        },
        'process_msu_thrombolysis': {
            'values': [15, 30, 45],
            'name': 'MSU IVT time',
            'default': 30  # 1  # index for 30
        },
        'process_msu_on_scene_post_thrombolysis': {
            'values': [15, 30],
            'name': 'MSU on scene post IVT time',
            'default': 15  # 0  # index for 15
        },
        'process_time_arrival_to_needle': {
            'values': [30, 45],
            'name': 'Hospital arrival to IVT time',
            'default': 30  # 0  # index for 30
        },
        'transfer_time_delay': {
            'values': [30, 60, 90],
            'name': 'Door-in to door-out (for transfer to MT)',
            'default': 60  # 1  # index for 60
        },
        'process_time_arrival_to_puncture': {
            'values': [30, 45, 60],
            'name': 'Hospital arrival to MT time (for in-hospital IVT+MT)',
            'default': 60  # 2  # index for 60
        },
        'process_time_transfer_arrival_to_puncture': {
            'values': [30, 45, 60],
            'name': 'Hospital arrival to MT time (for transfers)',
            'default': 60  # 2  # index for 60
        },
        'process_time_msu_arrival_to_puncture': {
            'values': [30, 45, 60],
            'name': 'Hospital arrival to MT time (for MSU arrivals)',
            'default': 60  # 2  # index for 60
        },
    }

    input_dict = {}
    for key, s_dict in scenarios.items():
        input_dict[key] = st.select_slider(
            s_dict['name'],
            s_dict['values'],
            value=s_dict['default'],
            key=key
            )

    return input_dict


@st.cache_data
def load_scenario_list():
    df = pd.read_csv('./data/scenario_list_england.csv')
    return df


@st.cache_data
def load_scenario_results():
    df = pd.read_csv('./data/scenario_results_england.csv')
    return df


def find_scenario_id(input_dict):
    # Import the file of all scenario parameter combinations:
    df = load_scenario_list()

    # Find the row of this dataframe that matches the input dict:
    mask_list = [df[key] == value for key, value in input_dict.items()]
    mask = np.all(mask_list, axis=0)

    # Pick out this row only:
    id = df.loc[mask, 'Scenario'].values[0]
    return id


def find_scenario_results(id):
    # Import the file of all scenario results:
    df = load_scenario_results()

    # Find the row of this dataframe with that scenario ID:
    row = df.loc[df['Scenario'] == id]

    # Rename any 'utilility' to 'utility:
    new_cols = []
    for c in row.columns:
        c = c.replace('utilility', 'utility')
        new_cols.append(c)
    row.columns = new_cols

    # Convert to dictionary:
    row = row.to_dict(orient='records')[0]

    return row


def select_scenario(containers=[]):
    if len(containers) == 0:
        containers = [st.container() for i in range(4)]

    # Outcome type input:
    with containers[0]:
        outcome_type_str = st.radio(
            'Outcome measure',
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

    # # Scenario input:
    # with containers[1]:    
    #     scenario_type_str = st.radio(
    #         'Scenario',
    #         ['Drip-and-ship', 'Mothership', 'MSU'],
    #         # horizontal=True
    #     )
    # # Match the input string to the file name string:
    # scenario_type_dict = {
    #     'Drip-and-ship': 'drip_ship',
    #     'Mothership': 'mothership',
    #     'MSU': 'msu'
    # }
    # scenario_type = scenario_type_dict[scenario_type_str]

    # Treatment type:
    with containers[1]:
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
    with containers[2]:
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

    scenario_dict = {}
    scenario_dict['outcome_type_str'] = outcome_type_str
    scenario_dict['outcome_type'] = outcome_type
    # scenario_dict['scenario_type_str'] = scenario_type_str
    # scenario_dict['scenario_type'] = scenario_type
    scenario_dict['treatment_type_str'] = treatment_type_str
    scenario_dict['treatment_type'] = treatment_type
    scenario_dict['stroke_type_str'] = stroke_type_str
    scenario_dict['stroke_type'] = stroke_type
    return scenario_dict


def convert_lsoa_to_msoa_results(df_lsoa):
    # Convert LSOA to MSOA:
    df_lsoa_to_msoa = pd.read_csv('data/lsoa_to_msoa.csv')
    df_lsoa = df_lsoa.reset_index()
    df_msoa = pd.merge(
        df_lsoa,
        df_lsoa_to_msoa[['lsoa11nm', 'msoa11cd', 'msoa11nm']],
        left_on='lsoa', right_on='lsoa11nm', how='left'
        )
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_msoa = df_msoa.drop([
        'lsoa', 'nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit',
        'nearest_msu_unit', 'lsoa11nm', 'msoa11nm'
        ], axis='columns')
    # Aggregate by MSOA:
    df_msoa = df_msoa.groupby('msoa11cd').mean()
    # df_msoa = df_msoa.set_index('msoa11cd')
    # Merge the MSOA names back in and set the index to (msoa_code, msoa):
    df_msoa = df_msoa.reset_index()
    df_msoa = pd.merge(
        df_msoa, df_lsoa_to_msoa[['msoa11cd', 'msoa11nm']],
        left_on='msoa11cd', right_on='msoa11cd', how='left'
        )
    # Remove duplicate rows:
    df_msoa = df_msoa.drop_duplicates()
    df_msoa = df_msoa.rename(columns={'msoa11cd': 'msoa_code', 'msoa11nm': 'msoa'})
    df_msoa = df_msoa.set_index(['msoa', 'msoa_code'])

    return df_msoa


def set_up_colours(scenario_dict):
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
    
    """
    # Define shared colour scales:
    cbar_dict = {
        'utility': {
            'scenario': {
                'vmin': 0.3,
                'vmax': 0.6,
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
                'vmax': 0.25,
                'step_size': 0.025,
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
                'vmin': -0.5,
                'vmax': 0.0,
                'step_size': 0.1,
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
                'vmin': 0.30,
                'vmax': 0.70,
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
    if scenario_dict['scenario_type'].startswith('diff'):
        scen = 'diff'
    else:
        scen = 'scenario'

    v_min = cbar_dict[scenario_dict['outcome_type']][scen]['vmin']
    v_max = cbar_dict[scenario_dict['outcome_type']][scen]['vmax']
    step_size = cbar_dict[scenario_dict['outcome_type']][scen]['step_size']
    cmap_name = cbar_dict[scenario_dict['outcome_type']][scen]['cmap_name']

    # Make a new column for the colours.
    v_bands = np.arange(v_min, v_max + step_size, step_size)
    v_bands_str = make_v_bands_str(v_bands)
    colour_map = make_colour_map_dict(v_bands_str, cmap_name)

    colour_dict = {
        'v_min': v_min,
        'v_max': v_max,
        'step_size': step_size,
        'cmap_name': cmap_name,
        'v_bands': v_bands,
        'v_bands_str': v_bands_str,
        'colour_map': colour_map,
    }
    return colour_dict


def make_colour_map_dict(v_bands_str, cmap_name='viridis'):
    # Get colour values:
    cmap = plt.get_cmap(cmap_name)
    cbands = np.linspace(0.0, 1.0, len(v_bands_str))
    colour_list = cmap(cbands)
    # # Convert tuples to strings:
    colour_list = np.array([
        f'rgba{tuple(c)}' for c in colour_list])
    # Sample the colour list:
    colour_map = [(c, colour_list[i]) for i, c in enumerate(v_bands_str)]

    # # Set over and under colours:
    # colour_list[0] = 'black'
    # colour_list[-1] = 'LimeGreen'

    # Return as dict to track which colours are for which bands:
    colour_map = dict(zip(v_bands_str, colour_list))
    return colour_map


def make_v_bands_str(v_bands):
    """Turn contour ranges into formatted strings."""
    v_min = v_bands[0]
    v_max = v_bands[-1]

    v_bands_str = [f'v < {v_min:.3f}']
    for i, band in enumerate(v_bands[:-1]):
        b = f'{band:.3f} <= v < {v_bands[i+1]:.3f}'
        v_bands_str.append(b)
    v_bands_str.append(f'{v_max:.3f} <= v')

    v_bands_str = np.array(v_bands_str)
    return v_bands_str
