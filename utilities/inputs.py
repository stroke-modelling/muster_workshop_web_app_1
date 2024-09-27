"""
All of the content for the Inputs section.
"""
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for colour maps
import cmasher as cmr  # for additional colour maps
from importlib_resources import files

import stroke_maps.load_data
from utilities.colour_setup import make_colourbar_display_string


def write_text_from_file(filename, head_lines_to_skip=0):
    """
    Write text from 'filename' into streamlit.
    Skip a few lines at the top of the file using head_lines_to_skip.
    """
    # Open the file and read in the contents,
    # skipping a few lines at the top if required.
    with open(filename, 'r', encoding="utf-8") as f:
        text_to_print = f.readlines()[head_lines_to_skip:]

    # Turn the list of all of the lines into one long string
    # by joining them up with an empty '' string in between each pair.
    text_to_print = ''.join(text_to_print)

    # Write the text in streamlit.
    st.markdown(f"""{text_to_print}""")


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
            'default': 60
        },
    }
    inputs_standard = {
        # Standard ambulance pathway
        'process_time_ambulance_response': {
            'name': 'Ambulance response time',
            'default': 20
        },
        'process_ambulance_on_scene_duration': {
            'name': 'Time ambulance is on scene',
            'default': 30
        },
        'process_time_arrival_to_needle': {
            'name': 'Hospital arrival to IVT time',
            'default': 40
        },
        'process_time_arrival_to_puncture': {
            'name': 'Hospital arrival to MT time (for in-hospital IVT+MT)',
            'default': 90
        },
    }
    inputs_transfer = {
        # Transfer required
        'transfer_time_delay': {
            'name': 'Door-in to door-out (for transfer to MT)',
            'default': 90
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
        'process_msu_on_scene_no_thrombolysis': {
            'name': 'MSU on scene (no thrombolysis given)',
            'default': 15
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


# def select_parameters_optimist():
#     """

#     TO DO another day - set these reference values up in fixed_params.
#     Default values from median onset to arrival times document
#     (Mike Allen, 23rd April 2024):
#     onset_to_call: 79,
#     call_to_ambulance_arrival_time: 18,
#     ambulance_on_scene_time: 29,
#     """
#     cols = st.columns([1, 1, 1, 5, 1])

#     with cols[0]:
#         container_shared = st.container(border=True)
#     with cols[1]:
#         container_on_scene = st.container(border=True)
#     with cols[2]:
#         container_diagnostic = st.container(border=True)
#     with cols[3]:
#         container_unit = st.container(border=True)
#     with cols[4]:
#         container_transfer = st.container(border=True)

#     container_occ = st.container()


#     input_dict = {}
#     # Set up scenarios
#     with container_shared:
#         st.markdown('Shared')
#         input_dict['process_time_call_ambulance'] = st.number_input(
#             'Time to call ambulance',
#             value=79,
#             help=f"Reference value: {79}",
#             min_value=0,
#             max_value=1440,
#             step=1
#             )

#     with container_on_scene:
#         st.markdown('On scene')
#         input_dict['process_time_ambulance_response'] = st.number_input(
#             'Ambulance response time',
#             value=18,
#             help=f"Reference value: {18}",
#             min_value=0,
#             max_value=1440,
#             step=1
#             )
#         input_dict['process_ambulance_on_scene_duration'] = st.number_input(
#             'Time ambulance is on scene',
#             value=29,
#             help=f"Reference value: {29}",
#             min_value=0,
#             max_value=1440,
#             step=1
#             )

#     with container_diagnostic:
#         st.markdown('Diagnostic')
#         input_dict['process_ambulance_on_scene_diagnostic_duration'] = st.number_input(
#             'Extra time on scene for diagnostic',
#             value=10,
#             help=f"Reference value: {10}",
#             min_value=0,
#             max_value=1440,
#             step=1
#             )

#     with container_unit:
#         st.markdown('Unit')
#         cols_unit = st.columns(3)
#         with cols_unit[0]:
#             input_dict['process_time_arrival_to_needle'] = st.number_input(
#                 'Hospital arrival to IVT time',
#                 value=30,
#                 help=f"Reference value: {30}",
#                 min_value=0,
#                 max_value=1440,
#                 step=1
#                 )
#         with cols_unit[1]:
#             container_unit_with_mt = st.container(border=True)
#         with container_unit_with_mt:
#             input_dict['process_time_arrival_to_puncture'] = st.number_input(
#                 'Hospital arrival to MT time (for in-hospital IVT+MT)',
#                 value=60,
#                 help=f"Reference value: {60}",
#                 min_value=0,
#                 max_value=1440,
#                 step=1
#                 )
#         with cols_unit[2]:
#             container_unit_without_mt = st.container(border=True)
#         with container_unit_without_mt:
#             input_dict['transfer_time_delay'] = st.number_input(
#                 'Door-in to door-out (for transfer to MT)',
#                 value=60,
#                 help=f"Reference value: {60}",
#                 min_value=0,
#                 max_value=1440,
#                 step=1
#                 )

#     with container_transfer:
#         st.markdown('Transfer unit')
#         input_dict['process_time_transfer_arrival_to_puncture'] = st.number_input(
#             'Hospital arrival to MT time (for transfers)',
#             value=60,
#             help=f"Reference value: {60}",
#             min_value=0,
#             max_value=1440,
#             step=1
#             )

#     inputs_occlusion = {
#         'prop_nlvo': {
#             'name': 'Proportion of population with nLVO',
#             'default': 0.65,
#             'min_value': 0.0,
#             'max_value': 1.0,
#             'step': 0.01,
#             'container': container_occ
#         },
#         'prop_lvo': {
#             'name': 'Proportion of population with LVO',
#             'default': 0.35,
#             'min_value': 0.0,
#             'max_value': 1.0,
#             'step': 0.01,
#             'container': container_occ
#         }
#     }
#     inputs_redirection = {
#         'sensitivity': {
#             'name': 'Sensitivity (proportion of LVO diagnosed as LVO)',
#             'default': 0.66,
#             'min_value': 0.0,
#             'max_value': 1.0,
#             'step': 0.01,
#             'container': container_occ
#         },
#         'specificity': {
#             'name': 'Specificity (proportion of nLVO diagnosed as nLVO)',
#             'default': 0.87,
#             'min_value': 0.0,
#             'max_value': 1.0,
#             'step': 0.01,
#             'container': container_occ
#         },
#     }

#     dicts = {
#         'Occlusion types': inputs_occlusion,
#         'Redirection': inputs_redirection
#         }

#     with container_occ:
#         for heading, i_dict in dicts.items():
#             st.markdown(f'### {heading}')
#             for key, s_dict in i_dict.items():
#                     input_dict[key] = st.number_input(
#                         s_dict['name'],
#                         value=s_dict['default'],
#                         help=f"Reference value: {s_dict['default']}",
#                         min_value=s_dict['min_value'],
#                         max_value=s_dict['max_value'],
#                         step=s_dict['step'],
#                         key=key
#                         )

#     return input_dict

def select_parameters_pathway_optimist():
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
    dicts = {
        'Shared': inputs_shared,
        'Standard pathway': inputs_standard,
        'Transfer required': inputs_transfer,
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


def select_parameters_population_optimist():
    inputs_occlusion = {
        # MADE UP these default values
        'prop_nlvo': {
            'name': 'Proportion of population considered for redirection who have nLVO',
            'default': 0.10,  # INVENTED
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        },
        'prop_lvo': {
            'name': 'Proportion of population considered for redirection who have LVO',
            'default': 0.90,  # INVENTED
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.01,
        }
    }
    # inputs_redirection_considered = {
    #     # Made up these default numbers.
    #     'prop_nlvo_redirection_considered': {
    #         'name': 'nLVO proportion considered for redirection',
    #         'default': 0.42,  # INVENTED
    #         'min_value': 0.0,
    #         'max_value': 1.0,
    #         'step': 0.01,
    #     },
    #     'prop_lvo_redirection_considered': {
    #         'name': 'LVO proportion considered for redirection',
    #         'default': 0.42,  # INVENTED
    #         'min_value': 0.0,
    #         'max_value': 1.0,
    #         'step': 0.01,
    #     },
    # }
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
        'Occlusion types': inputs_occlusion,
        # 'Redirection considered': inputs_redirection_considered,
        'Redirection approved': inputs_redirection
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

    # # Now calculate the proportions of the "redirection considered" group
    # # that are nLVO and LVO.
    # input_dict['prop_redirection_considered'] = (
    #     (input_dict['prop_nlvo'] *
    #      input_dict['prop_nlvo_redirection_considered']) +
    #     (input_dict['prop_lvo'] *
    #      input_dict['prop_lvo_redirection_considered'])
    # )
    # input_dict['prop_redirection_considered_nlvo'] = (
    #     (input_dict['prop_nlvo'] *
    #      input_dict['prop_nlvo_redirection_considered']) /
    #     input_dict['prop_redirection_considered']
    # )
    # input_dict['prop_redirection_considered_lvo'] = (
    #     (input_dict['prop_lvo'] *
    #      input_dict['prop_lvo_redirection_considered']) /
    #     input_dict['prop_redirection_considered']
    # )

    return input_dict


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


# def select_stroke_unit_services_broad(use_msu=True):
# WORK IN PROGRESS AND BROKEN!!
#     df_unit_services, df_unit_services_full, cols_use = (
#         import_stroke_unit_services(use_msu))

#     # Display and store any changes from the user:
#     df_unit_services = st.data_editor(
#         df_unit_services,
#         disabled=['postcode', 'stroke_team', 'isdn'],
#         height=180  # limit height to show fewer rows
#         )
    
#     # Select either:
#     # + MSU at all IVT-only units
#     # + MSU at all MT units
#     # + MSU at all IVT and/or MT units
#     add_all_ivt = st.button('Place MSU at all IVT-only units')
#     add_all_mt = st.button('Place MSU at all MT units')
#     add_all = st.button('Place MSU at all units')
#     remove_all_ivt = st.button('Remove MSU from all IVT-only units')
#     remove_all_mt = st.button('Remove MSU from all MT units')
#     remove_all = st.button('Remove MSU from all units')

#     units_ivt_bool = (
#         (df_unit_services['use_ivt'] == 1) &
#         (df_unit_services['use_mt'] == 0)
#     )
#     units_mt_bool = (
#         (df_unit_services['use_mt'] == 1)
#     )
#     if add_all_ivt:
#         df_unit_services.loc[units_ivt_bool, 'use_msu'] = 1
#     if add_all_mt:
#         df_unit_services.loc[units_mt_bool, 'use_msu'] = 1
#     if add_all:
#         df_unit_services['use_msu'] = 1
#     if remove_all_ivt:
#         df_unit_services.loc[units_ivt_bool, 'use_msu'] = 0
#     if remove_all_mt:
#         df_unit_services.loc[units_mt_bool, 'use_msu'] = 0
#     if remove_all:
#         df_unit_services['use_msu'] = 0

#     # For display:
#     df_unit_services['use_msu'] = df_unit_services['use_msu'].astype(bool)

#     df_unit_services, df_unit_services_full = update_stroke_unit_services(
#         df_unit_services, df_unit_services_full, cols_use)
#     return df_unit_services, df_unit_services_full


def import_stroke_unit_services(use_msu=True):
    # Set up stroke unit services (IVT, MT, MSU).
    df_unit_services = stroke_maps.load_data.stroke_unit_region_lookup()
    # Remove stroke units that don't offer IVT or MT:
    mask = (
        (df_unit_services['use_ivt'] == 1) |
        (df_unit_services['use_mt'] == 1)
    )
    df_unit_services = df_unit_services.loc[mask].copy()
    # Limit to England:
    mask = df_unit_services['country'] == 'England'
    df_unit_services = df_unit_services.loc[mask].copy()
    # Remove Wales:
    df_unit_services = df_unit_services.loc[
        df_unit_services['region_type'] != 'LHB'].copy()
    df_unit_services_full = df_unit_services.copy()
    # Limit which columns to show:
    cols_to_keep = [
        'stroke_team',
        'use_ivt',
        'use_mt',
        # 'region',
        # 'icb',
        'isdn'
    ]
    df_unit_services = df_unit_services[cols_to_keep]
    # Change 1/0 columns to bool for formatting:
    cols_use = ['use_ivt', 'use_mt']
    if use_msu:
        df_unit_services['use_msu'] = df_unit_services['use_mt'].copy()
        cols_use.append('use_msu')
    df_unit_services[cols_use] = df_unit_services[cols_use].astype(bool)
    # Sort by ISDN name for nicer display:
    df_unit_services = df_unit_services.sort_values('isdn')

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


def select_outcome_type():
    """
    """
    # Outcome type input:
    outcome_type_str = st.radio(
        'Outcome measure',
        [
            # 'Utility',
            'Added utility',
            # 'Mean shift in mRS',
            'mRS <= 2'
            ],
        index=0,  # 'added utility' as default
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


def select_treatment_type():
    # Treatment type:
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


def select_stroke_type(use_combo_stroke_types=False):
    options = ['LVO', 'nLVO']
    if use_combo_stroke_types:
        options += ['Combined']

    # Stroke type:
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


def select_colour_maps(cmap_names, cmap_diff_names):
    """
    User inputs.
    """
    # cmap_displays = [
    #     make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
    #     for cmap_name in cmap_names
    #     ]
    cmap_diff_displays = [
        make_colourbar_display_string(cmap_name, char_line='█', n_lines=15)
        for cmap_name in cmap_diff_names
        ]

    try:
        # cmap_name = st.session_state['cmap_name']
        cmap_diff_name = st.session_state['cmap_diff_name']
    except KeyError:
        # cmap_name = cmap_names[0]
        cmap_diff_name = cmap_diff_names[0]
    # cmap_ind = cmap_names.index(cmap_name)
    cmap_diff_ind = cmap_diff_names.index(cmap_diff_name)

    # cmap_name = st.radio(
    #     'Colour display for "usual care" map',
    #     cmap_names,
    #     captions=cmap_displays,
    #     index=cmap_ind,
    #     key='cmap_name'
    # )

    cmap_diff_name = st.radio(
        'Default colour display for difference map',
        cmap_diff_names,
        captions=cmap_diff_displays,
        index=cmap_diff_ind,
        key='cmap_diff_name'
    )
    cmap_name = cmap_diff_name

    return cmap_name, cmap_diff_name


def load_region_lists(df_unit_services_full):
    """
    # Nearest units from IVT units in df_unit_services,
    # ISDN and ICB from the reference data.
    """

    # Load region data:
    df_regions = stroke_maps.load_data.region_lookup()
    df_regions = df_regions.reset_index()
    # Only keep English regions:
    mask = df_regions['region_code'].str.contains('E')
    df_regions = df_regions.loc[mask]

    # Lists of ICBs and ISDNs without repeats:
    icb_list = sorted(list(set(df_regions['icb'])))
    isdn_list = sorted(list(set(df_regions['isdn'])))

    # Load ambulance service data:
    df_lsoa_ambo = stroke_maps.load_data.ambulance_lsoa_lookup()
    # List of ambulance services without repeats:
    ambo_list = sorted(list(set(df_lsoa_ambo['ambo22'])))
    # Drop Wales:
    ambo_list.remove('WAST')

    # Find list of units offering IVT.
    # Use names not postcodes here to match ICB and ISDN names
    # and have nicer display on the app.
    mask = df_unit_services_full['use_ivt'] == 1
    nearest_ivt_unit_names_list = sorted(df_unit_services_full.loc[mask, 'stroke_team'])

    # Key for region type, value for list of options.
    region_options_dict = {
        'ISDN': isdn_list,
        'ICB': icb_list,
        'Nearest unit': nearest_ivt_unit_names_list,
        'Ambulance service': ambo_list
    }

    return region_options_dict
