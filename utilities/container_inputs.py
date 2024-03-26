"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st
import pandas as pd
import numpy as np


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
            'name': 'Transfer time delay (for MT)',
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
    df = pd.read_csv('./data/scenario_list.csv')
    return df


@st.cache_data
def load_scenario_results():
    df = pd.read_csv('./data/scenario_results.csv')
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
