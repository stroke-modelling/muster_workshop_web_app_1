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
        'process_time_call_ambulance': [0, 60, 120, 180],
        'process_time_ambulance_response': [15, 30, 45],
        'process_ambulance_on_scene_duration': [20, 30, 40],
        'process_msu_dispatch': [0, 15, 30],
        'process_msu_thrombolysis': [15, 30, 45],
        'process_msu_on_scene_post_thrombolysis': [15, 30],
        'process_time_arrival_to_needle': [30, 45],
        'transfer_time_delay': [30, 60, 90],
        'process_time_arrival_to_puncture': [30, 45, 60],
        'process_time_transfer_arrival_to_puncture': [30, 45, 60],
        'process_time_msu_arrival_to_puncture': [30, 45, 60]    
    }
    rename_dict = {
        'process_time_call_ambulance': 'Time to call ambulance',
        'process_time_ambulance_response': 'Ambulance response time',
        'process_ambulance_on_scene_duration': 'Time ambulance is on scene',
        'process_msu_dispatch': 'MSU dispatch time',
        'process_msu_thrombolysis': 'MSU IVT time',
        'process_msu_on_scene_post_thrombolysis': 'MSU on scene post IVT time',
        'process_time_arrival_to_needle': 'Hospital arrival to IVT time',
        'transfer_time_delay': 'Transfer time delay (for MT)',
        'process_time_arrival_to_puncture': 'Hospital arrival to MT time (for in-hospital IVT+MT)',
        'process_time_transfer_arrival_to_puncture': 'Hospital arrival to MT time (for transfers)',
        'process_time_msu_arrival_to_puncture': 'Hospital arrival to MT time (for MSU arrivals)',
        }

    input_dict = {}
    for key, value_list in scenarios.items():
        label = rename_dict[key]
        input_dict[key] = st.selectbox(label, value_list, key=key)

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

    # Convert to dictionary:
    row = row.to_dict(orient='records')

    return row
