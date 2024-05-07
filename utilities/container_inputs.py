"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for colour maps
import cmasher as cmr  # for additional colour maps

from stroke_maps.catchment import Catchment  # for unit services


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


def select_parameters_map():
    """

    TO DO another day - set these reference values up in fixed_params.
    Default values from median onset to arrival times document
    (Mike Allen, 23rd April 2024):
    onset_to_call: 79,
    call_to_ambulance_arrival_time: 18,
    ambulance_on_scene_time: 29,
    """
    # Set up scenarios
    inputs_shared = {
        # Shared
        'process_time_call_ambulance': {
            'name': 'Time to call ambulance',
            'default': 79
        },
    }
    inputs_standard = {
        # Standard ambulance pathway
        'process_time_ambulance_response': {
            'name': 'Ambulance response time',
            'default': 18
        },
        'process_ambulance_on_scene_duration': {
            'name': 'Time ambulance is on scene',
            'default': 29
        },
        'process_time_arrival_to_needle': {
            'name': 'Hospital arrival to IVT time',
            'default': 30
        },
        'process_time_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for in-hospital IVT+MT)',
            'default': 60
        },
    }
    inputs_transfer = {
        # Transfer required
        'transfer_time_delay': {
            'name': 'Door-in to door-out (for transfer to MT)',
            'default': 60
        },
        'process_time_transfer_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for transfers)',
            'default': 60
        },
    }
    inputs_msu = {
        # MSU
        'process_msu_dispatch': {
            'name': 'MSU dispatch time',
            'default': 15
        },
        'process_msu_thrombolysis': {
            'name': 'MSU IVT time',
            'default': 30
        },
        'process_msu_on_scene_post_thrombolysis': {
            'name': 'MSU on scene post IVT time',
            'default': 15
        },
        'process_time_msu_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for MSU arrivals)',
            'default': 60
        },
        'scale_msu_travel_times': {
            'name': 'Scale factor for MSU travel speed',
            'default': 1.0
        },
    }

    dicts = {
        'Shared': inputs_shared,
        'Standard pathway': inputs_standard,
        'Transfer required': inputs_transfer,
        'Mobile Stroke Unit': inputs_msu
        }

    input_dict = {}
    for heading, i_dict in dicts.items():
        st.markdown(f'## {heading}')
        for key, s_dict in i_dict.items():
            input_dict[key] = st.number_input(
                s_dict['name'],
                value=s_dict['default'],
                help=f"Reference value: {s_dict['default']}",
                key=key
                )

    # Write an example for how the MSU speed affects the timings.
    time_not_msu = 20.0
    time_msu = 20.0 * input_dict['scale_msu_travel_times']

    example_str = ''.join([
        'For example, with a scale factor of ',
        f'{input_dict["scale_msu_travel_times"]}, '
        f'a journey that takes {time_not_msu:.0f} minutes ',
        f'in a normal ambulance would take {time_msu:.0f} minutes ',
        'in a Mobile Stroke Unit vehicle.'
        ])
    st.markdown(example_str)

    return input_dict


def select_parameters_optimist():
    """

    TO DO another day - set these reference values up in fixed_params.
    Default values from median onset to arrival times document
    (Mike Allen, 23rd April 2024):
    onset_to_call: 79,
    call_to_ambulance_arrival_time: 18,
    ambulance_on_scene_time: 29,
    """
    cols = st.columns([1, 1, 1, 5, 1])

    with cols[0]:
        container_shared = st.container(border=True)
    with cols[1]:
        container_on_scene = st.container(border=True)
    with cols[2]:
        container_diagnostic = st.container(border=True)
    with cols[3]:
        container_unit = st.container(border=True)
    with cols[4]:
        container_transfer = st.container(border=True)

    container_occ = st.container()


    input_dict = {}
    # Set up scenarios
    with container_shared:
        st.markdown('Shared')
        input_dict['process_time_call_ambulance'] = st.number_input(
            'Time to call ambulance',
            value=79,
            help=f"Reference value: {79}",
            min_value=0,
            max_value=1440,
            step=1
            )

    with container_on_scene:
        st.markdown('On scene')
        input_dict['process_time_ambulance_response'] = st.number_input(
            'Ambulance response time',
            value=18,
            help=f"Reference value: {18}",
            min_value=0,
            max_value=1440,
            step=1
            )
        input_dict['process_ambulance_on_scene_duration'] = st.number_input(
            'Time ambulance is on scene',
            value=29,
            help=f"Reference value: {29}",
            min_value=0,
            max_value=1440,
            step=1
            )

    with container_diagnostic:
        st.markdown('Diagnostic')
        input_dict['process_ambulance_on_scene_diagnostic_duration'] = st.number_input(
            'Extra time on scene for diagnostic',
            value=10,
            help=f"Reference value: {10}",
            min_value=0,
            max_value=1440,
            step=1
            )

    with container_unit:
        st.markdown('Unit')
        cols_unit = st.columns(3)
        with cols_unit[0]:
            input_dict['process_time_arrival_to_needle'] = st.number_input(
                'Hospital arrival to IVT time',
                value=30,
                help=f"Reference value: {30}",
                min_value=0,
                max_value=1440,
                step=1
                )
        with cols_unit[1]:
            container_unit_with_mt = st.container(border=True)
        with container_unit_with_mt:
            input_dict['process_time_arrival_to_puncture'] = st.number_input(
                'Hospital arrival to MT time (for in-hospital IVT+MT)',
                value=60,
                help=f"Reference value: {60}",
                min_value=0,
                max_value=1440,
                step=1
                )
        with cols_unit[2]:
            container_unit_without_mt = st.container(border=True)
        with container_unit_without_mt:
            input_dict['transfer_time_delay'] = st.number_input(
                'Door-in to door-out (for transfer to MT)',
                value=60,
                help=f"Reference value: {60}",
                min_value=0,
                max_value=1440,
                step=1
                )

    with container_transfer:
        st.markdown('Transfer unit')
        input_dict['process_time_transfer_arrival_to_puncture'] = st.number_input(
            'Hospital arrival to MT time (for transfers)',
            value=60,
            help=f"Reference value: {60}",
            min_value=0,
            max_value=1440,
            step=1
            )

    inputs_occlusion = {
        'prop_nlvo': {
            'name': 'Proportion of population with nLVO',
            'default': 0.65,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
            'container': container_occ
        },
        'prop_lvo': {
            'name': 'Proportion of population with LVO',
            'default': 0.35,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
            'container': container_occ
        }
    }
    inputs_redirection = {
        'sensitivity': {
            'name': 'Sensitivity (proportion of LVO diagnosed as LVO)',
            'default': 0.66,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
            'container': container_occ
        },
        'specificity': {
            'name': 'Specificity (proportion of nLVO diagnosed as nLVO)',
            'default': 0.87,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
            'container': container_occ
        },
    }

    dicts = {
        'Occlusion types': inputs_occlusion,
        'Redirection': inputs_redirection
        }

    with container_occ:
        for heading, i_dict in dicts.items():
            st.markdown(f'### {heading}')
            for key, s_dict in i_dict.items():
                    input_dict[key] = st.number_input(
                        s_dict['name'],
                        value=s_dict['default'],
                        help=f"Reference value: {s_dict['default']}",
                        min_value=s_dict['min_value'],
                        max_value=s_dict['max_value'],
                        step=s_dict['step'],
                        key=key
                        )

    return input_dict

def select_parameters_optimist_OLD():
    """
    This version creates a long list of number inputs.

    TO DO another day - set these reference values up in fixed_params.
    Default values from median onset to arrival times document
    (Mike Allen, 23rd April 2024):
    onset_to_call: 79,
    call_to_ambulance_arrival_time: 18,
    ambulance_on_scene_time: 29,
    """
    # Set up scenarios
    inputs_shared = {
        # Shared
        'process_time_call_ambulance': {
            'name': 'Time to call ambulance',
            'default': 79,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
    }
    inputs_standard = {
        # Standard ambulance pathway
        'process_time_ambulance_response': {
            'name': 'Ambulance response time',
            'default': 18,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
        'process_ambulance_on_scene_duration': {
            'name': 'Time ambulance is on scene',
            'default': 29,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
        'process_ambulance_on_scene_diagnostic_duration': {
            'name': 'Extra time on scene for diagnostic',
            'default': 10,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
        'process_time_arrival_to_needle': {
            'name': 'Hospital arrival to IVT time',
            'default': 30,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
        'process_time_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for in-hospital IVT+MT)',
            'default': 60,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
    }
    inputs_transfer = {
        # Transfer required
        'transfer_time_delay': {
            'name': 'Door-in to door-out (for transfer to MT)',
            'default': 60,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
        'process_time_transfer_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for transfers)',
            'default': 60,
            'min_value': 0,
            'max_value': 1440,
            'step': 1,
        },
    }
    inputs_occlusion = {
        'prop_nlvo': {
            'name': 'Proportion of population with nLVO',
            'default': 0.65,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        },
        'prop_lvo': {
            'name': 'Proportion of population with LVO',
            'default': 0.35,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        }
    }
    inputs_redirection = {
        'sensitivity': {
            'name': 'Sensitivity (proportion of LVO diagnosed as LVO)',
            'default': 0.66,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        },
        'specificity': {
            'name': 'Specificity (proportion of nLVO diagnosed as nLVO)',
            'default': 0.87,
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        },
    }

    dicts = {
        'Shared': inputs_shared,
        'Standard pathway': inputs_standard,
        'Transfer required': inputs_transfer,
        'Occlusion types': inputs_occlusion,
        'Redirection': inputs_redirection
        }

    input_dict = {}
    for heading, i_dict in dicts.items():
        st.markdown(f'## {heading}')
        for key, s_dict in i_dict.items():
            input_dict[key] = st.number_input(
                s_dict['name'],
                value=s_dict['default'],
                help=f"Reference value: {s_dict['default']}",
                min_value=s_dict['min_value'],
                max_value=s_dict['max_value'],
                step=s_dict['step'],
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


def select_stroke_unit_services(use_msu=True):
    df_unit_services, df_unit_services_full, cols_use = (
        import_stroke_unit_services(use_msu))

    # Display and store any changes from the user:
    df_unit_services = st.data_editor(
        df_unit_services,
        disabled=['postcode', 'stroke_team', 'isdn'],
        height=180  # limit height to show fewer rows
        )

    df_unit_services, df_unit_services_full = update_stroke_unit_services(
        df_unit_services, df_unit_services_full, cols_use)
    return df_unit_services, df_unit_services_full


def import_stroke_unit_services(use_msu=True):
    # Set up stroke unit services (IVT, MT, MSU).
    catchment = Catchment()
    df_unit_services = catchment.get_unit_services()
    # Remove Wales:
    df_unit_services = df_unit_services.loc[df_unit_services['region_type'] != 'LHB'].copy()
    df_unit_services_full = df_unit_services.copy()
    # Limit which columns to show:
    cols_to_keep = [
        'stroke_team',
        'use_ivt',
        'use_mt',
        # 'use_msu',
        # 'transfer_unit_postcode',  # to add back in later if stroke-maps replaces geography_processing class
        # 'region',
        # 'icb',
        'isdn'
    ]
    if use_msu:
        cols_to_keep.append('use_msu')
    df_unit_services = df_unit_services[cols_to_keep]
    # Change 1/0 columns to bool for formatting:
    cols_use = ['use_ivt', 'use_mt']
    if use_msu:
        cols_use.append('use_msu')
    df_unit_services[cols_use] = df_unit_services[cols_use].astype(bool)
    # Sort by ISDN name for nicer display:
    df_unit_services = df_unit_services.sort_values('isdn')

    # Update James Cook University Hospital to have MSU by default:
    if 'use_msu' in df_unit_services.columns:
        df_unit_services.at['TS43BW', 'use_msu'] = True
    return df_unit_services, df_unit_services_full, cols_use


def update_stroke_unit_services(
        df_unit_services,
        df_unit_services_full,
        cols_use
        ):
    # Restore dtypes:
    df_unit_services[cols_use] = df_unit_services[cols_use].astype(int)

    # Update the full data (for maps) with the changes:
    cols_to_merge = cols_use  # + ['transfer_unit_postcode']
    df_unit_services_full = df_unit_services_full.drop(
        cols_to_merge, axis='columns')
    df_unit_services_full = pd.merge(
        df_unit_services_full,
        df_unit_services[cols_to_merge].copy(),
        left_index=True, right_index=True, how='left'
        )

    # Rename columns to match what the rest of the model here wants.
    df_unit_services.index.name = 'Postcode'
    df_unit_services = df_unit_services.rename(columns={
        'use_ivt': 'Use_IVT',
        'use_mt': 'Use_MT',
        'use_msu': 'Use_MSU',
    })
    return df_unit_services, df_unit_services_full


def select_scenario(
        scenarios=['outcome', 'treatment', 'stroke'],
        containers=[],
        use_combo_stroke_types=False
        ):
    if len(containers) == 0:
        containers = [st.container() for i in range(3)]

    # Store results in here:
    scenario_dict = {}
    if 'outcome' in scenarios:
        outcome_type, outcome_type_str = select_outcome_type(containers[0])
        scenario_dict['outcome_type_str'] = outcome_type_str
        scenario_dict['outcome_type'] = outcome_type
    if 'treatment' in scenarios:
        treatment_type, treatment_type_str = select_treatment_type(containers[1])
        scenario_dict['treatment_type_str'] = treatment_type_str
        scenario_dict['treatment_type'] = treatment_type
    if 'stroke' in scenarios:
        stroke_type, stroke_type_str = select_stroke_type(
            containers[2], use_combo_stroke_types)
        scenario_dict['stroke_type_str'] = stroke_type_str
        scenario_dict['stroke_type'] = stroke_type

    # scenario_dict['scenario_type_str'] = scenario_type_str
    # scenario_dict['scenario_type'] = scenario_type
    return scenario_dict


def select_outcome_type(container=None):
    """
    """
    if container is None:
        container = st.container()

    # Outcome type input:
    with container:
        outcome_type_str = st.radio(
            'Outcome measure',
            [
                # 'Utility',
                'Added utility',
                # 'Mean shift in mRS',
                'mRS <= 2'
                ],
            index=1,  # 'added utility' as default
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
    return outcome_type, outcome_type_str


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


def select_treatment_type(container=None):
    if container is None:
        container = st.container()

    # Treatment type:
    with container:
        treatment_type_str = st.radio(
            'Treatment type',
            ['IVT', 'MT', 'IVT & MT'],
            index=2,  # IVT & MT as default
            # horizontal=True
            )
    # Match the input string to the file name string:
    treatment_type_dict = {
        'IVT': 'ivt',
        'MT': 'mt',
        'IVT & MT': 'ivt_mt'
    }
    treatment_type = treatment_type_dict[treatment_type_str]
    return treatment_type, treatment_type_str


def select_stroke_type(container=None, use_combo_stroke_types=False):
    if container is None:
        container = st.container()

    options = ['LVO', 'nLVO']
    if use_combo_stroke_types:
        options += ['Combined']

    # Stroke type:
    with container:
        stroke_type_str = st.radio(
            'Stroke type',
            options,
            # horizontal=True
            )
    # Match the input string to the file name string:
    stroke_type_dict = {
        'LVO': 'lvo',
        'nLVO': 'nlvo',
        'Combined': 'combo'
    }
    stroke_type = stroke_type_dict[stroke_type_str]
    return stroke_type, stroke_type_str


def set_up_colours(scenario_dict, v_name='v', cmap_name='inferno', cmap_diff_name='RdBu'):
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
                'vmin': -0.05,
                'vmax': 0.05,
                'step_size': 0.025,
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
    try:
        # Matplotlib colourmap:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = plt.get_cmap(f'cmr.{cmap_name}')

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


def make_colourbar_display_string(cmap_name, char_line='█', n_lines=20):
    try:
        # Matplotlib colourmap:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        # CMasher colourmap:
        cmap = plt.get_cmap(f'cmr.{cmap_name}')

    # Get colours:
    colours = cmap(np.linspace(0.0, 1.0, n_lines))
    # Convert tuples to strings:
    colours = (colours * 255).astype(int)
    # Drop the alpha or the colour won't be right!
    colours = ['#%02x%02x%02x' % tuple(c[:-1]) for c in colours]

    line_str = '$'
    for c in colours:
        # s = f"<font color='{c}'>{char_line}</font>"
        s = '\\textcolor{' + f'{c}' + '}{' + f'{char_line}' + '}'
        line_str += s
    line_str += '$'
    return line_str
