"""
MUSTER results display app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this example, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st
import pandas as pd

# Custom functions:
import utilities.calculations as calc
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.plot_mrs_dists as mrs
from utilities.maps_raster import make_raster_from_vectors, \
    set_up_raster_transform
# Containers:
import utilities.inputs as inputs
import utilities.colour_setup as colour_setup


@st.cache_data
def main_calculations(input_dict, df_unit_services):
    # Times to treatment:
    geo = calc.calculate_geography(df_unit_services)
    # Travel times for each LSOA:
    df_travel_times = geo.combined_data[
        [c for c in geo.combined_data.columns if 'time' in c] +
        ['transfer_required', 'LSOA']
        ]
    df_travel_times = df_travel_times.set_index('LSOA')

    # Add travel times to the pathway timings to get treatment times.
    df_outcome_uc = calc.make_outcome_inputs_usual_care(
        input_dict, df_travel_times)
    df_outcome_msu = calc.make_outcome_inputs_msu(
        input_dict, df_travel_times)
    dict_outcome_inputs = {
        'usual_care': df_outcome_uc,
        'msu': df_outcome_msu,
    }

    # Process LSOA and calculate outcomes:
    df_lsoa, df_mrs = calc.calculate_outcomes(
        dict_outcome_inputs, df_unit_services, geo.combined_data)

    # Calculate diff - msu minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['msu', 'usual_care']
        )
    df_mrs = calc.combine_results_by_diff(
        df_mrs,
        scenario_types=['msu', 'usual_care'],
        combine_mrs_dists=True
        )

    # Place probabilities of death into st.session_state['df_lsoa'] data
    # so that they are displayed in the results tables.
    cols_probs_of_death = [
        'usual_care_lvo_ivt_mrs_dists_noncum_6',
        'usual_care_lvo_ivt_mt_mrs_dists_noncum_6',
        'usual_care_lvo_mt_mrs_dists_noncum_6',
        'usual_care_nlvo_ivt_mrs_dists_noncum_6',
        'msu_lvo_ivt_mrs_dists_noncum_6',
        'msu_lvo_ivt_mt_mrs_dists_noncum_6',
        'msu_lvo_mt_mrs_dists_noncum_6',
        'msu_nlvo_ivt_mrs_dists_noncum_6',
    ]
    df_lsoa = pd.merge(
        df_lsoa, df_mrs[cols_probs_of_death],
        left_index=True, right_index=True, how='left'
    )

    df_icb, df_isdn, df_nearest_ivt, df_ambo = calc.group_results_by_region(
        df_lsoa, df_unit_services)

    return df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt, df_ambo


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )


# #####################################
# ########## CONTAINER SETUP ##########
# #####################################
# Make containers:
# +-----------------------------------------------+
# |                container_intro                |
# +-------------------------+---------------------+
# |      container_map      | container_mrs_dists |
# +-------------------------+---------------------+
# |              container_map_inputs             |
# +-----------------------------------------------+
# |            container_results_tables           |
# +-----------------------------------------------+

# Sidebar:
# form
#   +--------------------------+
#   |     container_inputs     |
#   +--------------------------+
#   |  container_unit_services |
#   +--------------------------+
# /form
# v Accessibility & advanced options
#   +--------------------------+
#   | container_select_outcome |
#   +--------------------------+
#   |  container_select_cmap   |
#   +--------------------------+

container_intro = st.container()
tab_inputs, tab_results = st.tabs(['Inputs', 'Results'])
with tab_inputs:
    container_inputs = st.container()
    # container_timeline_plot = st.container()
    container_unit_services = st.container()

with container_inputs:
    container_inputs_top = st.container()
    (
        container_inputs_standard,
        container_timeline_standard,
        container_inputs_msu,
        container_timeline_msu
    ) = st.columns(4)

with tab_results:
    container_rerun = st.container()
    container_map, container_mrs_dists_etc = st.columns([2, 1])
    # Convert the map container to empty so that the placeholder map
    # is replaced once the real map is ready.
    with container_map:
        container_map = st.empty()
    # Convert mRS dists to empty so that re-running a fragment replaces
    # the bars rather than displays the new plot in addition.
    with container_mrs_dists_etc:
        container_mrs_dists = st.empty()
    container_map_inputs = st.container(border=True)
    with container_map_inputs:
        st.markdown('__Plot options__')
        (container_input_treatment,
         container_input_stroke_type,
         container_input_region_type,
         container_input_mrs_region) = st.columns(4)
    with container_input_mrs_region:
        container_input_mrs_region = st.empty()
    with st.expander('Full data tables'):
        container_results_tables = st.container()

with st.sidebar:
    with st.expander('Accessibility & advanced options'):
        container_select_outcome = st.container()
        container_select_cmap = st.container()

with container_intro:
    st.markdown('# Benefit in outcomes from Mobile Stroke Units')


# #################################
# ########## USER INPUTS ##########
# #################################

# These affect the data in all tables and all plots.

# ----- Pathway timings and stroke units -----
input_dict = {}
with container_inputs_top:
    st.markdown('## Pathway inputs')
    input_dict['process_time_call_ambulance'] = st.number_input(
        'Time to call ambulance',
        value=60,
        help=f"Reference value: {60}",
        # key=key
        )
with container_inputs_standard:
    st.markdown('### Standard pathway')
    input_dict = inputs.select_parameters_map(input_dict)
with container_inputs_msu:
    st.markdown('### Mobile Stroke Unit')
    input_dict = inputs.select_parameters_msu(input_dict)

with container_unit_services:
    st.header('Stroke unit services')
    st.markdown('Update which services the stroke units provide:')
    df_unit_services, df_unit_services_full = (
        inputs.select_stroke_unit_services())

# These do not change the underlying data,
# but do change what is shown in the plots.

# ----- Stroke type, treatment, outcome -----
with container_select_outcome:
    st.markdown('### Alternative outcome measures')
    outcome_type, outcome_type_str = inputs.select_outcome_type()
with container_input_treatment:
    treatment_type, treatment_type_str = inputs.select_treatment_type()
with container_input_stroke_type:
    stroke_type, stroke_type_str = (
        inputs.select_stroke_type(use_combo_stroke_types=False))

# ----- Regions to draw -----
# Name of the column in the geojson that labels the shapes:
with container_input_region_type:
    outline_name = st.radio(
        'Region type to draw on maps',
        ['None', 'ISDN', 'ICB', 'Nearest service', 'Ambulance service']
        )

# ----- Colourmap selection -----
cmap_names = [
    'cosmic', 'viridis', 'inferno', 'neutral'
    ]
cmap_diff_names = [
    'iceburn_r', 'seaweed', 'fusion', 'waterlily'
    ]
with container_select_cmap:
    st.markdown('### Colour schemes')
    cmap_name, cmap_diff_name = inputs.select_colour_maps(
        cmap_names, cmap_diff_names)
# If we're showing mRS scores then flip the colour maps:
if outcome_type == 'mrs_shift':
    cmap_name += '_r'
    cmap_diff_name += '_r'
    # Remove any double reverse reverse.
    if cmap_name.endswith('_r_r'):
        cmap_name = cmap_name[:-4]
    if cmap_diff_name.endswith('_r_r'):
        cmap_diff_name = cmap_diff_name[:-4]



# TIME LINE
# import utilities.plot_timeline as timeline
# times_dicts = timeline.build_data_for_timeline(input_dict, use_drip_ship=True, use_mothership=True, use_msu=False)


def build_time_dicts(pathway_dict):
    # Pre-hospital "usual care":
    time_dict_prehosp_usual_care = {'onset': 0}
    prehosp_keys = [
        'process_time_call_ambulance',
        'process_time_ambulance_response',
        'process_ambulance_on_scene_duration',
        ]
    for key in prehosp_keys:
        time_dict_prehosp_usual_care[key] = pathway_dict[key]

    # MSU dispatch:
    time_dict_msu_dispatch = {'onset': 0}
    msu_dispatch_keys = [
        'process_time_call_ambulance',
        'process_msu_dispatch',
        ]
    for key in msu_dispatch_keys:
        time_dict_msu_dispatch[key] = pathway_dict[key]

    # MSU:
    time_dict_prehosp_msu_ivt = {'msu_arrival_on_scene': 0}
    prehosp_msu_ivt_keys = [
        'process_msu_thrombolysis',
        'process_msu_on_scene_post_thrombolysis',
        ]
    for key in prehosp_msu_ivt_keys:
        time_dict_prehosp_msu_ivt[key] = pathway_dict[key]

    # MSU, no thrombolysis:
    time_dict_prehosp_msu_no_ivt = {'msu_arrival_on_scene': 0}
    prehosp_msu_no_ivt_keys = [
        'process_msu_on_scene_no_thrombolysis',
        ]
    for key in prehosp_msu_no_ivt_keys:
        time_dict_prehosp_msu_no_ivt[key] = pathway_dict[key]

    # Transfer to MT unit from MSU:
    time_dict_mt_transfer_unit_from_msu = {'arrival_ivt_mt': 0}
    time_dict_mt_transfer_unit_from_msu['arrival_to_puncture'] = (
        pathway_dict['process_time_msu_arrival_to_puncture'])

    # IVT-only unit:
    time_dict_ivt_only_unit = {'arrival_ivt_only': 0}
    time_dict_ivt_only_unit['arrival_to_needle'] = (
        pathway_dict['process_time_arrival_to_needle'])
    time_dict_ivt_only_unit['needle_to_door_out'] = (
        pathway_dict['transfer_time_delay'] -
        pathway_dict['process_time_arrival_to_needle']
    )

    # MT transfer unit:
    time_dict_mt_transfer_unit = {'arrival_ivt_mt': 0}
    time_dict_mt_transfer_unit['arrival_to_puncture'] = (
        pathway_dict['process_time_transfer_arrival_to_puncture'])

    # IVT and MT unit:
    time_dict_ivt_mt_unit = {'arrival_ivt_mt': 0}
    time_dict_ivt_mt_unit['arrival_to_needle'] = (
        pathway_dict['process_time_arrival_to_needle'])
    time_dict_ivt_mt_unit['needle_to_puncture'] = (
        pathway_dict['process_time_arrival_to_puncture'] -
        pathway_dict['process_time_arrival_to_needle']
    )

    time_dicts = {
        'prehosp_usual_care': time_dict_prehosp_usual_care,
        'msu_dispatch': time_dict_msu_dispatch,
        'prehosp_msu_ivt': time_dict_prehosp_msu_ivt,
        'prehosp_msu_no_ivt': time_dict_prehosp_msu_no_ivt,
        'mt_transfer_from_msu': time_dict_mt_transfer_unit_from_msu,
        'ivt_only_unit': time_dict_ivt_only_unit,
        'mt_transfer_unit': time_dict_mt_transfer_unit,
        'ivt_mt_unit': time_dict_ivt_mt_unit,
    }
    return time_dicts


time_dicts = build_time_dicts(input_dict)

# Emoji unicode reference:
# 🔧 \U0001f527
# 🏥 \U0001f3e5
# 🚑 \U0001f691
# 💉 \U0001f489
# ☎ \U0000260E
timeline_display_dict = {
    'onset': {
        'emoji': '',
        'text': 'Onset',
    },
    'process_time_call_ambulance': {
        'emoji': '\U0000260E',
        'text': 'Call<br>ambulance',
    },
    'process_time_ambulance_response': {
        'emoji': '\U0001f691',
        'text': 'Ambulance<br>arrives on scene',
    },
    'process_ambulance_on_scene_duration': {
        'emoji': '\U0001f691',
        'text': 'Ambulance<br>leaves',
    },
    'arrival_ivt_only': {
        'emoji': '\U0001f3e5',
        'text': 'Arrival<br>IVT unit',
    },
    'arrival_ivt_mt': {
        'emoji': '\U0001f3e5',
        'text': 'Arrival<br>MT unit',
    },
    'arrival_to_needle': {
        'emoji': '\U0001f489',
        'text': '<b><span style="color:red">IVT</span></b>',
    },
    'needle_to_door_out': {
        'emoji': '\U0001f691',
        'text': 'Ambulance<br>transfer<br>begins',
    },
    'needle_to_puncture': {
        'emoji': '\U0001f527',
        'text': '<b><span style="color:red">MT</span></b>',
    },
    'arrival_to_puncture': {
        'emoji': '\U0001f527',
        'text': '<b><span style="color:red">MT</span></b>',
    },
    'process_msu_dispatch': {
        'emoji': '\U0001f691',
        'text': 'MSU<br>leaves base'
    },
    'msu_arrival_on_scene': {
        'emoji': '\U0001f691',
        'text': 'MSU arrives<br>on scene'
    },
    'process_msu_thrombolysis': {
        'emoji': '\U0001f489',
        'text': '<b><span style="color:red">IVT</span></b>',
    },
    'process_msu_on_scene_post_thrombolysis': {
        'emoji': '\U0001f691',
        'text': 'MSU<br>leaves scene'
    },
    'process_msu_on_scene_no_thrombolysis': {
        'emoji': '\U0001f691',
        'text': 'MSU<br>leaves scene'
    },
    'nearest_ivt_time': {
        'emoji': '',
        'text': '',
    },
    'nearest_mt_time': {
        'emoji': '',
        'text': '',
    },
    'transfer_time': {
        'emoji': '',
        'text': '',
    },
    }


import plotly.graph_objects as go
import numpy as np

def plot_timeline(
        time_dicts,
        timeline_display_dict,
        y_vals,
        y_labels,
        time_offsets=[],
        tmax=None,
        tmin=None,
        ):
    if len(time_offsets) == 0:
        time_offsets = [0] * len(time_dicts.keys())

    # Pre-hospital timelines
    fig = go.Figure()


    # # Draw box
    # fig.add_trace(go.Scatter(
    #     y=[0, 100, 100, 0],
    #     x=[0, 0, -2, -2],
    #     fill="toself",
    #     hoverinfo='skip',
    #     mode='lines',
    #     line=dict(color="RoyalBlue", width=3),
    #     showlegend=False
    #     ))

    # Assume the keys are in the required order:
    time_names = list(time_dicts.keys())

    for i, time_name in enumerate(time_names):
        # Pick out this dict:
        time_dict = time_dicts[time_name]
        # Convert the discrete times into cumulative times:
        cum_times = np.cumsum(list(time_dict.values()))
        # Pick out the emoji and text labels to plot:
        emoji_here = [timeline_display_dict[key]['emoji']
                      for key in list(time_dict.keys())]
        labels_here = [f'{timeline_display_dict[key]["text"]}'  #<br><br><br>'
                       for key in list(time_dict.keys())]
        time_offset = time_offsets[time_name]

        fig.add_trace(go.Scatter(
            y=time_offset + cum_times,
            x=[y_vals[i]]*len(cum_times),
            mode='lines+markers+text',
            text=emoji_here,
            marker=dict(symbol='line-ew', size=10,
                        line_width=2, line_color='grey'),
            line_color='grey',
            textposition='middle center',
            textfont=dict(size=24),
            name=time_name,
            showlegend=False,
            hoverinfo='skip'  # 'y'
        ))
        fig.add_trace(go.Scatter(
            y=time_offset + cum_times,
            x=[y_vals[i] + 0.2]*len(cum_times),
            mode='text',
            text=labels_here,
            textposition='middle right',
            # textfont=dict(size=24)
            showlegend=False,
            hoverinfo='skip'
        ))

        # Sneaky extra scatter marker for hover text:
        y_sneaky = np.array([np.mean([cum_times[i], cum_times[i+1]]) for i in range(len(cum_times) - 1)])
        y_diffs = [f'{d}    ' for d in np.diff(cum_times)]
        fig.add_trace(go.Scatter(
            y=time_offset + y_sneaky,
            x=[y_vals[i]]*len(y_sneaky),
            mode='text',
            text=y_diffs,
            line_color='grey',
            textposition='middle left',
            textfont=dict(size=18),
            showlegend=False,
            hoverinfo='skip'  # 'y'
        ))

    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=[],  # y_vals,
        # ticktext=y_labels
    ))
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=[],
        zeroline=False
    ))
    fig.update_layout(xaxis_range=[min(y_vals) - 0.5, max(y_vals) + 1.0])
    # fig.update_layout(yaxis_title='Time (minutes)')

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(
        # autosize=False,
        # width=500,
        height=tmax*2.0,
        yaxis=dict(
            range=[tmax, tmin],  # Default x-axis zoom.
        ),
        # Make the default cursor setting pan instead of zoom box:
        dragmode='pan'
    )
    # fig.update_yaxes(autorange="reversed")
    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            'zoom',
            # 'pan',
            'select',
            # 'zoomIn',
            # 'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=plotly_config
        )


gap_between_chunks = 45

time_offsets = {
    'prehosp_usual_care': 0,
    'ivt_only_unit': gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values()),
    'mt_transfer_unit': gap_between_chunks * 2.0 + sum(time_dicts['prehosp_usual_care'].values()) + sum(time_dicts['ivt_only_unit'].values()),
    'ivt_mt_unit': gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values()),
    'msu_dispatch': 0,
    'prehosp_msu_ivt': gap_between_chunks + sum(time_dicts['msu_dispatch'].values()),
    'prehosp_msu_no_ivt': gap_between_chunks + sum(time_dicts['msu_dispatch'].values()),
    'mt_transfer_from_msu': gap_between_chunks * 2.0 + sum(time_dicts['msu_dispatch'].values()) + max([sum(time_dicts['prehosp_msu_ivt'].values()), sum(time_dicts['prehosp_msu_no_ivt'].values())]),
}
tmax = max(
    [time_offsets[k] + sum(time_dicts[k].values()) for k in time_dicts.keys()]
) + gap_between_chunks

time_keys_standard = [
    'prehosp_usual_care',
    'ivt_only_unit',
    'mt_transfer_unit',
    'ivt_mt_unit',
]
time_dicts_standard = dict([(k, time_dicts[k]) for k in time_keys_standard])
time_offsets_standard = dict([(k, time_offsets[k]) for k in time_keys_standard])


time_keys_msu = [
    'msu_dispatch',
    'prehosp_msu_ivt',
    'prehosp_msu_no_ivt',
    'mt_transfer_from_msu',
]
time_dicts_msu = dict([(k, time_dicts[k]) for k in time_keys_msu])
time_offsets_msu = dict([(k, time_offsets[k]) for k in time_keys_msu])


with container_timeline_standard:
    plot_timeline(
        time_dicts_standard,
        timeline_display_dict,
        y_vals=[0.5, 0, 0, 1],
        y_labels=[
            'Usual care',
            'IVT-only unit',
            'Transfer to MT unit',
            'IVT & MT unit'
            ],
        time_offsets=time_offsets_standard,
        tmax=tmax,
        tmin=-gap_between_chunks
        )
with container_timeline_msu:
    plot_timeline(
        time_dicts_msu, timeline_display_dict, y_vals=[0.5, 0, 1, 0.5],
        y_labels=[
            'MSU dispatch', 'MSU (IVT)', 'MSU (no IVT)',
            'Transfer to MT unit (from MSU)'
            ],
        time_offsets=time_offsets_msu,
        tmax=tmax,
        tmin=-gap_between_chunks
    )

st.stop()


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################
# While the main calculations are happening, display a blank map.
# Later, when the calculations are finished, replace with the actual map.
with container_map:
    plot_maps.plotly_blank_maps(['', ''], n_blank=2)

try:
    inputs_changed = (
        (st.session_state['input_dict'] != input_dict) |
        (st.session_state['df_unit_services']['Use_IVT'] != df_unit_services['Use_IVT']).any() |
        (st.session_state['df_unit_services']['Use_MT'] != df_unit_services['Use_MT']).any() |
        (st.session_state['df_unit_services']['Use_MSU'] != df_unit_services['Use_MSU']).any()
    )
except KeyError:
    # First run of the app.
    inputs_changed = False

with container_rerun:
    if st.button('Calculate results'):
        st.session_state['input_dict'] = input_dict
        st.session_state['df_unit_services'] = df_unit_services
        (
            st.session_state['df_lsoa'],
            st.session_state['df_mrs'],
            st.session_state['df_icb'],
            st.session_state['df_isdn'],
            st.session_state['df_nearest_ivt'],
            st.session_state['df_ambo']
        ) = main_calculations(input_dict, df_unit_services)
    else:
        if inputs_changed:
            with container_rerun:
                st.warning('Inputs have changed! The results currently being shown are for the previous set of inputs. Use the "calculate results" button to update the results.', icon='⚠️')


if 'df_lsoa' in st.session_state.keys():
    pass
else:
    # This hasn't been created yet and so the results cannot be drawn.
    st.stop()

# #########################################
# ########## RESULTS - FULL DATA ##########
# #########################################
with container_results_tables:
    results_tabs = st.tabs([
        'Results by IVT unit catchment',
        'Results by ISDN',
        'Results by ICB',
        'Results by ambulance service',
        'Full results by LSOA'
        ])

    # Set some columns to bool for nicer display:
    cols_bool = ['transfer_required', 'England']
    for col in cols_bool:
        for df in [st.session_state['df_icb'], st.session_state['df_isdn'], st.session_state['df_nearest_ivt'], st.session_state['df_ambo'], st.session_state['df_lsoa']]:
            df[col] = df[col].astype(bool)

    with results_tabs[0]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each IVT unit catchment area.'
            ]))
        st.dataframe(st.session_state['df_nearest_ivt'])

    with results_tabs[1]:
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.dataframe(st.session_state['df_isdn'])

    with results_tabs[2]:
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.dataframe(st.session_state['df_icb'])

    with results_tabs[3]:
        st.markdown(''.join([
            'Results are the mean values of all LSOA ',
            'in each ambulance service.'
            ]))
        st.dataframe(st.session_state['df_ambo'])

    with results_tabs[4]:
        st.dataframe(st.session_state['df_lsoa'])


# #########################################
# ########## RESULTS - mRS DISTS ##########
# #########################################

# st.session_state['df_mrs'] column names are in the format:
# `usual_care_lvo_ivt_mt_mrs_dists_X`, for X from 0 to 6, i.e.
# '{scenario}_{occlusion}_{treatment}_{dist}_{X}' with these options:
#
# +---------------------------+------------+------------+------------------+
# | Scenarios                 | Occlusions | Treatments | Dist types       |
# +---------------------------+------------+------------+------------------+
# | usual_care                | nlvo       | ivt        | mrs_dists        |
# | msu                       | lvo        | mt         | mrs_dists_noncum |
# | diff_msu_minus_usual_care |            | ivt_mt     |                  |
# +---------------------------+------------+------------+------------------+
#
# There is not a separate column for "no treatment" to save space.

# Limit the mRS data to only LSOA that benefit from an MSU,
# i.e. remove anything where the added utility of MSU is not better
# than the added utility of usual care.
d_str = 'diff_msu_minus_usual_care'

if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # This data doesn't exist so show no LSOAs.
    lsoa_to_keep = []
elif ((stroke_type == 'nlvo') & ('mt' in treatment_type)):
    # Use IVT-only data:
    c1 = f'{d_str}_{stroke_type}_ivt_{outcome_type}'
    lsoa_to_keep = st.session_state['df_lsoa'].index[(st.session_state['df_lsoa'][c1] > 0.0)]
else:
    # Look up the data normally.
    c1 = f'{d_str}_{stroke_type}_{treatment_type}_{outcome_type}'
    lsoa_to_keep = st.session_state['df_lsoa'].index[(st.session_state['df_lsoa'][c1] > 0.0)]

# mRS distributions that meet the criteria:
df_mrs_to_plot = st.session_state['df_mrs'][st.session_state['df_mrs'].index.isin(lsoa_to_keep)]

with container_mrs_dists_etc:
    st.markdown(''.join([
        'mRS distributions shown for only LSOA who would benefit ',
        'from an MSU (i.e. "added utility" for "MSU" scenario ',
        'is better than for "usual care" scenario).'
        ]))


# Select mRS distribution region.
# Select a region based on what's actually in the data,
# not by guessing in advance which IVT units are included for example.
region_options_dict = inputs.load_region_lists(df_unit_services_full)
bar_options = ['National']
for key, region_list in region_options_dict.items():
    bar_options += [f'{key}: {v}' for v in region_list]

# Which mRS distributions will be shown on the bars:
scenario_mrs = ['usual_care', 'msu']

# Keep this in its own fragment so that choosing a new region
# to plot doesn't re-run the maps too.


@st.fragment
def display_mrs_dists():
    # User input:
    bar_option = st.selectbox('Region for mRS distributions', bar_options)

    mrs_lists_dict, region_selected, col_pretty = (
        mrs.setup_for_mrs_dist_bars(
            bar_option,
            stroke_type,
            treatment_type,
            stroke_type_str,
            treatment_type_str,
            st.session_state['df_lsoa'][['nearest_ivt_unit', 'nearest_ivt_unit_name']],
            df_mrs_to_plot,
            input_dict,
            scenarios=scenario_mrs
            ))

    mrs.plot_mrs_bars(
        mrs_lists_dict, title_text=f'{region_selected}<br>{col_pretty}')


with container_mrs_dists:
    display_mrs_dists()


# ####################################
# ########## SETUP FOR MAPS ##########
# ####################################
# Keep this below the results above because the map creation is slow.

# ----- Set up geodataframe -----
gdf = maps.load_lsoa_gdf()

# Merge in outcomes data:
gdf = pd.merge(
    gdf, st.session_state['df_lsoa'],
    left_on='LSOA11NM', right_on='lsoa', how='left'
    )


# ----- Find data for colours -----

# st.session_state['df_lsoa'] column names are in the format:
# `usual_care_lvo_ivt_mt_utility_shift`, i.e.
# '{scenario}_{occlusion}_{treatment}_{outcome}' with these options:
#
# +---------------------------+------------+------------+---------------+
# | Scenarios                 | Occlusions | Treatments | Outcomes      |
# +---------------------------+------------+------------+---------------+
# | usual_care                | nlvo       | ivt        | utility_shift |
# | msu                       | lvo        | mt         | mrs_shift     |
# | diff_msu_minus_usual_care |            | ivt_mt     | mrs_0-2       |
# +---------------------------+------------+------------+---------------+
#
# There is not a separate column for "no treatment" to save space.

# Find the names of the columns that contain the data
# that will be shown in the colour maps.
if ((stroke_type == 'nlvo') & (treatment_type == 'mt')):
    # Use no-treatment data.
    # Set this to something that doesn't exist so it fails the try.
    column_colours = None
    column_colours_diff = None
else:
    # If this is nLVO with IVT and MT, look up the data for
    # nLVO with IVT only.
    using_nlvo_ivt_mt = ((stroke_type == 'nlvo') & ('mt' in treatment_type))
    t = 'ivt' if using_nlvo_ivt_mt else treatment_type

    column_colours = '_'.join([
        'usual_care', stroke_type, t, outcome_type])
    column_colours_diff = '_'.join([
        'diff_msu_minus_usual_care', stroke_type, t, outcome_type])

# Pick out the columns of data for the colours:
try:
    vals_for_colours = gdf[column_colours]
    vals_for_colours_diff = gdf[column_colours_diff]
except KeyError:
    # Those columns don't exist in the data.
    # This should only happen for nLVO treated with MT only.
    vals_for_colours = [0] * len(gdf)
    vals_for_colours_diff = [0] * len(gdf)
    # Note: this works for now because expect always no change
    # for added utility and added mrs<=2 with no treatment.


# ----- Convert vectors to raster -----
# Set up parameters for conversion to raster:
transform_dict = set_up_raster_transform(gdf, pixel_size=2000)
# Burn geometries for left-hand map...
burned_lhs = make_raster_from_vectors(
    gdf['geometry'],
    vals_for_colours,
    transform_dict['height'],
    transform_dict['width'],
    transform_dict['transform']
)
# ... and right-hand map:
burned_rhs = make_raster_from_vectors(
    gdf['geometry'],
    vals_for_colours_diff,
    transform_dict['height'],
    transform_dict['width'],
    transform_dict['transform']
)


# ----- Set up colours -----
# Load colour limits info (vmin, vmax, step_size):
dict_colours, dict_colours_diff = (
    colour_setup.load_colour_limits(outcome_type))
# Load colour map colours:
dict_colours['cmap'] = colour_setup.make_colour_list(
    cmap_name,
    vmin=dict_colours['vmin'],
    vmax=dict_colours['vmax']
    )
dict_colours_diff['cmap'] = colour_setup.make_colour_list(
    cmap_diff_name,
    vmin=dict_colours_diff['vmin'],
    vmax=dict_colours_diff['vmax']
    )
# Colour bar titles:
dict_colours['title'] = f'{outcome_type_str}'
dict_colours_diff['title'] = (
    f'{outcome_type_str}: Benefit of MSU over usual care')


# ----- Region outlines -----
if outline_name == 'None':
    outline_names_col = None
    gdf_catchment_lhs = None
    gdf_catchment_rhs = None
else:
    outline_names_col, gdf_catchment_lhs, gdf_catchment_rhs = (
        calc.load_or_calculate_region_outlines(outline_name, st.session_state['df_lsoa']))


# ----- Process geography for plotting -----
# Convert gdf polygons to xy cartesian coordinates:
gdfs_to_convert = [gdf_catchment_lhs, gdf_catchment_rhs]
for gdf in gdfs_to_convert:
    if gdf is None:
        pass
    else:
        x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
        gdf['x'] = x_list
        gdf['y'] = y_list


# ----- Stroke units -----
# Stroke unit scatter markers:
traces_units = plot_maps.create_stroke_team_markers(df_unit_services_full)
# Which subplots to draw which units on:
# Each entry is [row number, column number].
# In plotly, the first row is 1 and first column is 1.
# The order in which they are drawn (and so which markers appear
# on top) is the order of this dictionary.
unit_subplot_dict = {
    'msu': [[1, 2]],        # second map only
    'ivt': [[1, 1]],        # first map only
    'mt': [[1, 1], [1, 2]]  # both maps
}


# ----- Plot -----
# Display names:
subplot_titles = [
    'Usual care',
    'Benefit of MSU over usual care'
]
with container_map:
    plot_maps.plotly_many_heatmaps(
        burned_lhs,
        burned_rhs,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        outline_names_col,
        outline_name,
        traces_units,
        unit_subplot_dict,
        subplot_titles=subplot_titles,
        dict_colours=dict_colours,
        dict_colours_diff=dict_colours_diff,
        transform_dict=transform_dict,
        )
