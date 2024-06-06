"""
Extra notebook to show the OPTIMIST app workings in more detail.
"""
# ----- Imports -----
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Custom functions:
import utilities.calculations as calc
import utilities.container_inputs as inputs
from utilities.utils import load_reference_mrs_dists
import utilities.plot_mrs_dists as mrs
import utilities.maps as maps
import utilities.plot_maps as plot_maps


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='OPTIMIST',
    page_icon=':ambulance:',
    layout='wide'
    )

st.markdown('# OPTIMIST app step-by-step')
st.markdown(
'''
The app does three main things:
+ Calculate __times to treatment__.
+ Use those times to calculate __outcomes__ for all patient types.
+ Find results for a __population__ of stroke patients by combining results of different patient types.
'''
)

with st.sidebar:
    container_inputs = st.container()

# #################################
# ########## USER INPUTS ##########
# #################################

# These affect the data in all tables and all plots.
with container_inputs:
    with st.form('Model setup'):
        st.markdown('### Pathway inputs')
        # input_dict = inputs.select_parameters_optimist_OLD()
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
            'Transfer required': inputs_transfer
            }

        pathway_dict = {}
        for heading, i_dict in dicts.items():
            st.markdown(f'## {heading}')
            for key, s_dict in i_dict.items():
                pathway_dict[key] = st.number_input(
                    s_dict['name'],
                    value=s_dict['default'],
                    help=f"Reference value: {s_dict['default']}",
                    min_value=s_dict['min_value'],
                    max_value=s_dict['max_value'],
                    step=s_dict['step'],
                    key=key
                    )
        # Add in some blank values for the travel times:
        for key in ['nearest_ivt_time', 'transfer_time', 'nearest_mt_time']:
            pathway_dict[key] = np.NaN

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services(use_msu=False))

        # Button for completing the form
        # (so script only re-runs once it is pressed, allows changes
        # to multiple widgets at once.)
        submitted = st.form_submit_button('Submit')

# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################

# Process LSOA and calculate outcomes:
df_lsoa, df_mrs = calc.calculate_outcomes(
    pathway_dict, df_unit_services, use_msu=False, use_mothership=True)


# ##############################
# ########## TIMELINE ##########
# ##############################

st.markdown('## Pathway timings')
st.markdown(
'''
Inputs that affect this:
+ Anything related to "time"
'''
)
st.markdown('### Timeline without travel')


# from utilities.plot_timeline import build_data_for_timeline, draw_timeline
# # Gather cumulative times and nicer-formatted cumulative time labels:
# # (times_dicts, times_cum_dicts, times_cum_label_dicts
# # times_dicts = build_data_for_timeline(pathway_dict)

# # Drip-and-ship:
# # Times since onset:
# times_keys_drip_ship = [
#     'process_time_call_ambulance',
#     'process_time_ambulance_response',
#     'process_ambulance_on_scene_duration',
# ]
# # Times since IVT unit arrival:
#     'process_time_arrival_to_needle',
#     'transfer_time_delay',
#     'transfer_time',
#     'process_time_transfer_arrival_to_puncture',
# ]
# times_values_drip_ship = [params_dict[k] for k in times_keys_drip_ship]
# times_dict_drip_ship = time_dict | dict(zip(times_keys_drip_ship, times_values_drip_ship))
# times_dicts['drip_ship'] = times_dict_drip_ship

# # st.write(times_dicts)
# # st.write(times_cum_dicts)
# # st.write(times_cum_label_dicts)

# # draw_timeline(times_cum_dicts, times_cum_label_dicts)
# st.stop()

# "Usual care" is either go to an IVT unit and then an MT unit,
# or go to the nearest unit which happens to have IVT and MT.
# "Redirect" adds the extra time for diagnostic and can still
# have either of those two options.

# Pre-hospital "usual care":
time_dict_prehosp_usual_care = {'onset': 0}
prehosp_keys = [
    'process_time_call_ambulance',
    'process_time_ambulance_response',
    'process_ambulance_on_scene_duration',
    ]
for key in prehosp_keys:
    time_dict_prehosp_usual_care[key] = pathway_dict[key]
# Extra time for redirection:
time_dict_prehosp_redirect = {'onset': 0}
# Keep this order so that on the timeline plots, the ambulance doesn't
# say it leaves before doing the diagnostic.
prehosp_redirect_keys = [
    'process_time_call_ambulance',
    'process_time_ambulance_response',
    'process_ambulance_on_scene_diagnostic_duration',
    'process_ambulance_on_scene_duration',
    ]
for key in prehosp_redirect_keys:
    time_dict_prehosp_redirect[key] = pathway_dict[key]

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

# Emoji unicode reference:
# 🔧 \U0001f527
# 🏥 \U0001f3e5
# 🚑 \U0001f691
# 💉 \U0001f489
# \U0000260E
emoji_dict = {
    'onset': '',
    'process_time_call_ambulance': '\U0000260E',
    'process_time_ambulance_response': '\U0001f691',
    'process_ambulance_on_scene_diagnostic_duration': '\U0000260E',
    'process_ambulance_on_scene_duration': '\U0001f691',
    'arrival_ivt_only': '\U0001f3e5',
    'arrival_ivt_mt': '\U0001f3e5',
    'arrival_to_needle': '\U0001f489',
    'needle_to_door_out': '\U0001f691',
    'needle_to_puncture': '\U0001f527',
    'arrival_to_puncture': '\U0001f527',
    # 'MSU<br>leaves base': '\U0001f691',
    # 'MSU<br>arrives on scene': '\U0001f691',
    # 'MSU<br>leaves scene': '\U0001f691',
    'nearest_ivt_time': '',
    'nearest_mt_time': '',
    'transfer_time': ''
    }
display_text_dict = {
    'onset': 'Onset',
    'process_time_call_ambulance': 'Call<br>ambulance',
    'process_time_ambulance_response': 'Ambulance<br>arrives<br>on scene',
    'process_ambulance_on_scene_diagnostic_duration': 'Extra time<br>for<br>diagnostic',
    'process_ambulance_on_scene_duration': 'Ambulance<br>leaves',
    'arrival_ivt_only': 'Arrival<br>IVT unit',
    'arrival_ivt_mt': 'Arrival<br>MT unit',
    'arrival_to_needle': '<b><span style="color:red">IVT</span></b>',
    'needle_to_door_out': 'Ambulance<br>transfer<br>begins',
    'needle_to_puncture': '<b><span style="color:red">MT</span></b>',
    'arrival_to_puncture': '<b><span style="color:red">MT</span></b>',
    # 'MSU<br>leaves base': '\U0001f691',
    # 'MSU<br>arrives on scene': '\U0001f691',
    # 'MSU<br>leaves scene': '\U0001f691',
    'nearest_ivt_time': '',
    'nearest_mt_time': '',
    'transfer_time': ''
    }

cols = st.columns([6, 4], gap='large')
# Keep an empty middle column to adjust the gap between plots.
with cols[0]:
    container_timeline_prehosp = st.container()
with cols[1]:
    container_timeline_info = st.container()

# Pre-hospital timelines
fig = go.Figure()

y_vals = [0, -1, -3, -4, -5]

time_dict = time_dict_prehosp_usual_care
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[0]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Usual care',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[0]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

time_dict = time_dict_prehosp_redirect
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[1]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Redirection',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[1]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

# fig.update_layout(yaxis=dict(
#     tickmode='array',
#     tickvals=y_vals,
#     ticktext=['Usual care', 'Redirection']
# ))
# fig.update_layout(yaxis_range=[-0.25, 0.15])
# fig.update_layout(xaxis_title='Time (minutes)')

# with container_timeline_prehosp:
#     st.plotly_chart(fig, use_container_width=True)


# # Stroke unit timelines
# fig = go.Figure()

# y_vals = []

time_dict = time_dict_ivt_only_unit
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[2]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='IVT-only unit',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[2]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

time_dict = time_dict_mt_transfer_unit
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[3]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Transfer to MT unit',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[3]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

time_dict = time_dict_ivt_mt_unit
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[4]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='IVT & MT unit',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[4]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(yaxis=dict(
    tickmode='array',
    tickvals=y_vals,
    ticktext=['Usual care', 'Redirection', 'IVT-only unit', 'Transfer to MT unit', 'IVT & MT unit']
))
fig.update_layout(yaxis_range=[y_vals[-1] - 0.5, y_vals[0] + 1])
fig.update_layout(xaxis_title='Time (minutes)')

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.update_layout(
    # autosize=False,
    # width=500,
    height=700
)

with container_timeline_prehosp:
    st.plotly_chart(fig, use_container_width=True)



### CALCULATE TIMES TO TREATMENT 

# Usual care:
time_to_ivt_without_travel_usual_care = (
    pathway_dict['process_time_call_ambulance'] +
    pathway_dict['process_time_ambulance_response'] +
    pathway_dict['process_ambulance_on_scene_duration'] +
    pathway_dict['process_time_arrival_to_needle']
)

time_to_mt_no_transfer_without_travel_usual_care = (
    pathway_dict['process_time_call_ambulance'] +
    pathway_dict['process_time_ambulance_response'] +
    pathway_dict['process_ambulance_on_scene_duration'] +
    pathway_dict['process_time_arrival_to_puncture']
)

time_to_mt_with_transfer_without_travel_usual_care = (
    pathway_dict['process_time_call_ambulance'] +
    pathway_dict['process_time_ambulance_response'] +
    pathway_dict['process_ambulance_on_scene_duration'] +
    pathway_dict['transfer_time_delay'] +
    pathway_dict['process_time_transfer_arrival_to_puncture']
)

# Redirect:
time_to_ivt_without_travel_redirect = (
    time_to_ivt_without_travel_usual_care +
    pathway_dict['process_ambulance_on_scene_diagnostic_duration']
)

time_to_mt_no_transfer_without_travel_redirect = (
    time_to_mt_no_transfer_without_travel_usual_care +
    pathway_dict['process_ambulance_on_scene_diagnostic_duration']
)

time_to_mt_with_transfer_without_travel_redirect = (
    time_to_mt_with_transfer_without_travel_usual_care +
    pathway_dict['process_ambulance_on_scene_diagnostic_duration']
)


# TO DO - currently the model completely skips the process_ambulance_on_scene_diagnostic_duration bit!
arr_times = [
    [time_to_ivt_without_travel_usual_care,
     time_to_mt_no_transfer_without_travel_usual_care,
     time_to_mt_with_transfer_without_travel_usual_care
     ],
    [time_to_ivt_without_travel_redirect,
     time_to_mt_no_transfer_without_travel_redirect,
     time_to_mt_with_transfer_without_travel_redirect
    ]
    ]
df_times = pd.DataFrame(
    arr_times,
    columns=['IVT', 'MT (no transfer)', 'MT (after transfer)'],
    index=['Usual care', 'Redirected']
)

with container_timeline_info:
    st.markdown('The pathway timings add up to these treatment times (__excluding__ travel times):')
    st.table(df_times)



# ##################################
# ########## TIME COMPARISON #######
# ##################################

st.markdown('### Treatment times with travel')
st.markdown(
'''
Inputs that affect this:
+ Which stroke units provide IVT and MT.
'''
)

st.markdown('Take the shared pathway times and add on the travel times to the chosen stroke units.')

selected_unit = st.selectbox(
    'Transfer unit',
    sorted(list(set(df_lsoa['transfer_unit_name'])))
    )

# Limit to just one region:
mask = df_lsoa['transfer_unit_name'].isin([selected_unit])#, 'PL68DH', 'TQ27AA'])
df_lsoa = df_lsoa.loc[mask].copy()
mask_mrs = df_mrs.index.isin(df_lsoa.index)
df_mrs = df_mrs.loc[mask_mrs].copy()

# st.markdown('These tables are for Anna\'s reference, ignore them please')
# st.write(df_lsoa)
# st.write(df_mrs)

# Usual care times:
col_ivt_times_usual_care = 'drip_ship_ivt_time'
col_mt_times_usual_care = 'drip_ship_mt_time'
# Redirect times:
col_ivt_times_redirect = 'mothership_ivt_time'
col_mt_times_redirect = 'mothership_mt_time'


# ############################
# ########## TIME MAPS #######
# ############################

# Load in LSOA, limit to only selected, assign colours by time, show four subplots (IVT/MT usual/redirect).
# Hover to show the LSOA name please.
# Shared colour scales.

tmin = 150
tmax = 360
tstep = 30
cmap_name = 'inferno_r'
subplot_titles = ['Usual care - time to IVT', 'Redirect - time to IVT',
                  'Usual care - time to MT', 'Redirect - time to MT']
cmap_titles = ['Time'] * 4
merge_polygons_bool = False
region_type = 'LSOA'
use_discrete_cmap = False

plot_the_maps_please = st.checkbox('Plot maps')
if plot_the_maps_please:
    gdf_usual_ivt, colour_dict_usual_ivt = maps.create_colour_gdf_demog(
        df_lsoa[col_ivt_times_usual_care],
        col_ivt_times_usual_care,
        tmin,
        tmax,
        tstep,
        cmap_name=cmap_name,
        cbar_title=cmap_titles[0],
        merge_polygons_bool=merge_polygons_bool,
        region_type=region_type,
        use_discrete_cmap=use_discrete_cmap
        )
    gdf_usual_mt, colour_dict_usual_mt = maps.create_colour_gdf_demog(
        df_lsoa[col_mt_times_usual_care],
        col_mt_times_usual_care,
        tmin,
        tmax,
        tstep,
        cmap_name=cmap_name,
        cbar_title=cmap_titles[0],
        merge_polygons_bool=merge_polygons_bool,
        region_type=region_type,
        use_discrete_cmap=use_discrete_cmap
        )
    gdf_redirect_ivt, colour_dict_redirect_ivt = maps.create_colour_gdf_demog(
        df_lsoa[col_ivt_times_redirect],
        col_ivt_times_redirect,
        tmin,
        tmax,
        tstep,
        cmap_name=cmap_name,
        cbar_title=cmap_titles[0],
        merge_polygons_bool=merge_polygons_bool,
        region_type=region_type,
        use_discrete_cmap=use_discrete_cmap
        )
    gdf_redirect_mt, colour_dict_redirect_mt = maps.create_colour_gdf_demog(
        df_lsoa[col_mt_times_redirect],
        col_mt_times_redirect,
        tmin,
        tmax,
        tstep,
        cmap_name=cmap_name,
        cbar_title=cmap_titles[0],
        merge_polygons_bool=merge_polygons_bool,
        region_type=region_type,
        use_discrete_cmap=use_discrete_cmap
        )

    # ----- Process geography for plotting -----
    # Convert gdf polygons to xy cartesian coordinates:
    gdfs_to_convert = [gdf_usual_ivt, gdf_redirect_ivt, gdf_usual_mt, gdf_redirect_mt]
    for gdf in gdfs_to_convert:
        if gdf is None:
            pass
        else:
            x_list, y_list = maps.convert_shapely_polys_into_xy(gdf)
            gdf['x'] = x_list
            gdf['y'] = y_list

    plot_maps.plotly_time_maps(
        [gdf_usual_ivt, gdf_redirect_ivt, gdf_usual_mt, gdf_redirect_mt],
        [colour_dict_usual_ivt, colour_dict_redirect_ivt, colour_dict_usual_mt, colour_dict_redirect_mt],
        subplot_titles=subplot_titles,
        use_discrete_cmap=use_discrete_cmap
        )

# ##################################
# ########## TIME SCATTER ##########
# ##################################
fig = go.Figure()

# Add background diagonal line to show parity:
time_min = np.min([df_lsoa[[col_ivt_times_redirect, col_ivt_times_usual_care, col_mt_times_redirect, col_mt_times_usual_care]]])
time_max = np.max([df_lsoa[[col_ivt_times_redirect, col_ivt_times_usual_care, col_mt_times_redirect, col_mt_times_usual_care]]])
fig.add_trace(go.Scatter(
    x=[time_min, time_max],
    y=[time_min, time_max],
    mode='lines',
    line_color='grey',
    showlegend=False,
    hoverinfo='skip'
))


# Plot connecting lines from markers to diagonal line:
x_lines_ivt = np.stack((df_lsoa[col_ivt_times_usual_care],
                        df_lsoa[col_ivt_times_usual_care],
                        [None]*len(df_lsoa)), axis=-1).flatten()
y_lines_ivt = np.stack((df_lsoa[col_ivt_times_usual_care],
                        df_lsoa[col_ivt_times_redirect],
                        [None]*len(df_lsoa)), axis=-1).flatten()
x_lines_mt = np.stack((df_lsoa[col_mt_times_usual_care],
                       df_lsoa[col_mt_times_usual_care],
                       [None]*len(df_lsoa)), axis=-1).flatten()
y_lines_mt = np.stack((df_lsoa[col_mt_times_usual_care],
                       df_lsoa[col_mt_times_redirect],
                       [None]*len(df_lsoa)), axis=-1).flatten()
fig.add_trace(go.Scatter(
    x=x_lines_ivt,
    y=y_lines_ivt,
    line_color='grey',
    line_width=0.5,
    mode='lines',
    showlegend=False,
    hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=x_lines_mt,
    y=y_lines_mt,
    line_color='grey',
    line_width=0.5,
    mode='lines',
    showlegend=False,
    hoverinfo='skip'
))

# Plot lines between IVT and MT:
x_lines = np.stack((df_lsoa[col_ivt_times_usual_care],
                    df_lsoa[col_mt_times_usual_care],
                    [None]*len(df_lsoa)), axis=-1).flatten()
y_lines = np.stack((df_lsoa[col_ivt_times_redirect],
                    df_lsoa[col_mt_times_redirect],
                    [None]*len(df_lsoa)), axis=-1).flatten()
# fig.add_trace(go.Scatter(
#     x=x_lines,
#     y=y_lines,
#     mode='lines',
#     line_color='Salmon',
#     line_width=0.5,
#     showlegend=False,
#     hoverinfo='skip'
# ))
# Plot markers for IVT:
fig.add_trace(go.Scatter(
    x=df_lsoa[col_ivt_times_usual_care],
    y=df_lsoa[col_ivt_times_redirect],
    customdata=np.stack((df_lsoa.index.values, df_lsoa['nearest_ivt_unit']), axis=-1),
    # customdata=[df_lsoa.index.values],
    marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
    mode='markers',
    name='IVT',
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>' +
        'Usual care IVT: %{x}<br>' +
        'Redirection IVT: %{y}' +
        '<extra></extra>'
    )
))
# Plot markers for MT:
fig.add_trace(go.Scatter(
    x=df_lsoa[col_mt_times_usual_care],
    y=df_lsoa[col_mt_times_redirect],
    customdata=np.stack((df_lsoa.index.values, df_lsoa['nearest_ivt_unit']), axis=-1),
    marker=dict(symbol='square', color='cyan', line=dict(color='black', width=0.5)),
    mode='markers',
    name='MT',
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>' +
        'Usual care MT: %{x}<br>' +
        'Redirection MT: %{y}' +
        '<extra></extra>'
    )
))


fig.add_annotation(
    xref="x domain",
    yref="y domain",
    x=0.25,
    y=0.95,
    showarrow=False,
    text='"Usual care"<br>is <b>faster</b><br>than "redirection"',
    # arrowhead=2,
)

fig.add_annotation(
    xref="x domain",
    yref="y domain",
    x=0.75,
    y=0.05,
    showarrow=False,
    text='"Usual care"<br>is <b>slower</b><br>than "redirection"',
    # arrowhead=2,
)

# Equivalent to pyplot set_aspect='equal':
fig.update_yaxes(scaleanchor='x', scaleratio=1)

# Axis ticks every 30min:
xticks = np.arange(0, time_max+30, 30)
yticks = np.arange(0, time_max+30, 30)
xticklabels = [f'{t//60:.0f}h {int(round(t%60, 0)):02}min' for t in xticks]
yticklabels = [f'{t//60:.0f}h {int(round(t%60, 0)):02}min' for t in yticks]

fig.update_layout(xaxis=dict(
    tickmode='array',
    tickvals=xticks,
    ticktext=xticklabels
))
fig.update_layout(yaxis=dict(
    tickmode='array',
    tickvals=yticks,
    ticktext=yticklabels
))
# Axis limits:
fig.update_layout(xaxis_range=[time_min, time_max])
fig.update_layout(yaxis_range=[time_min, time_max])

# Axis labels:
fig.update_layout(xaxis_title='Time to treatment with<br>usual care')
fig.update_layout(yaxis_title='Time to treatment with<br>redirection')

# Show vertical grid lines (makes it obvious where the labels are)
fig.update_xaxes(showgrid=True)  # , gridwidth=1, gridcolor='LimeGreen')

st.plotly_chart(fig)

# Pick out some extreme values:
time_ivt_diff = (df_lsoa[col_ivt_times_redirect] -
                 df_lsoa[col_ivt_times_usual_care])
time_mt_diff = (df_lsoa[col_mt_times_redirect] -
                df_lsoa[col_mt_times_usual_care])
# Redirection is best when time_ivt_diff is small (preferably zero) *and*
# time_mt_diff is minimum (preferably below zero).
time_metric = time_ivt_diff + time_mt_diff
inds_best = np.where(time_metric == np.min(time_metric))[0]
lsoa_best = df_lsoa.index.values[inds_best]
st.write('LSOA(s) with best time tradeoff: ', ', '.join(lsoa_best), '.')
# Redirection is worst when ivt diff is large (+ve) and mt diff is large
# (less negative).
inds_worst = np.where(time_metric == np.max(time_metric))[0]
lsoa_worst = df_lsoa.index.values[inds_worst]
st.write('LSOA(s) with worst time tradeoff: ', ', '.join(lsoa_worst), '.')

# #######################
# ##### LSOA CHOICE #####
# #######################

st.markdown('### Results for one LSOA')

# Pick out an LSOA:
# lsoa_name = 'East Devon 005D'
lsoa_name = st.selectbox('LSOA', df_lsoa.index)

# Write the stroke unit names:
st.markdown(f'Nearest IVT unit: {df_lsoa.loc[lsoa_name, "nearest_ivt_unit_name"]}')
st.markdown(f'Nearest MT unit: {df_lsoa.loc[lsoa_name, "nearest_mt_unit_name"]}')
st.markdown(f'Transfer unit: {df_lsoa.loc[lsoa_name, "transfer_unit_name"]}')

st.markdown('__Treatment times__')
transfer_required = (
    True if df_lsoa.loc[lsoa_name, 'transfer_required'] == True else False)
str_transfer_required = (
    '' if transfer_required else 'not ')
st.markdown(f'A transfer is {str_transfer_required} required for MT.')


cols = st.columns(3)
with cols[0]:
    st.markdown('Treatment times without travel:')
    st.table(df_times)  # Treatment times from earlier

# Travel times:
travel_times = [
    df_lsoa.loc[lsoa_name, 'nearest_ivt_time'],
    df_lsoa.loc[lsoa_name, 'nearest_mt_time'],
    df_lsoa.loc[lsoa_name, 'transfer_time'],
    ]
df_travel_times = pd.Series(
    travel_times, 
    index=['Time to nearest IVT unit', 'Time to nearest MT unit', 'Time for transfer'],
    name='Travel times'
    )
with cols[1]:
    st.markdown('Travel times:')
    st.table(df_travel_times)

arr_times = [
    [df_lsoa.loc[lsoa_name, 'drip_ship_ivt_time'],
     df_lsoa.loc[lsoa_name, 'drip_ship_mt_time'],
     ],
    [df_lsoa.loc[lsoa_name, 'drip_ship_ivt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     df_lsoa.loc[lsoa_name, 'drip_ship_mt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     ],
    [df_lsoa.loc[lsoa_name, 'mothership_ivt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     df_lsoa.loc[lsoa_name, 'mothership_mt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     ],
    ]
df_times = pd.DataFrame(
    arr_times,
    columns=['IVT', 'MT'],
    index=['Usual care', 'Redirection rejected', 'Redirection approved']
)
with cols[2]:
    st.markdown('Treatment times with travel:')
    st.table(df_times)


# Draw timeline for this LSOA:
dict_transfer = (
        time_dict_ivt_only_unit |
        {'transfer_time': df_lsoa.loc[lsoa_name, 'transfer_time']} |
        time_dict_mt_transfer_unit
    )
dict_no_transfer = time_dict_ivt_mt_unit
if transfer_required:
    dict2 = dict_transfer
else:
    dict2 = dict_no_transfer

time_dict_here_usual_care = (
    time_dict_prehosp_usual_care |
    {'nearest_ivt_time': df_lsoa.loc[lsoa_name, 'nearest_ivt_time']} |
    dict2
)
time_dict_here_redirect_but_not = (
    time_dict_prehosp_redirect |
    {'nearest_ivt_time': df_lsoa.loc[lsoa_name, 'nearest_ivt_time']} |
    dict_transfer
)
time_dict_here_redirect_but_yes = (
    time_dict_prehosp_redirect |
    {'nearest_mt_time': df_lsoa.loc[lsoa_name, 'nearest_mt_time']} |
    dict_no_transfer
)

# Pre-hospital timelines
fig = go.Figure()

y_vals = [0, -1, -2]

time_dict = time_dict_here_usual_care
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[0]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Usual care',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[0]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

time_dict = time_dict_here_redirect_but_not
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[1]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Redirection',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[1]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

time_dict = time_dict_here_redirect_but_yes
cum_times = np.cumsum(list(time_dict.values()))
emoji_here = [emoji_dict[key] for key in list(time_dict.keys())]
labels_here = [f'{display_text_dict[key]}<br><br><br>' for key in list(time_dict.keys())]
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[2]]*len(cum_times),
    mode='lines+markers+text',
    text=emoji_here,
    marker=dict(symbol='line-ns', size=10, line_width=2, line_color='grey'),
    line_color='grey',
    textposition='top center',
    textfont=dict(size=24),
    name='Redirection',
    showlegend=False,
    hoverinfo='x'
))
fig.add_trace(go.Scatter(
    x=cum_times,
    y=[y_vals[2]]*len(cum_times),
    mode='text',
    text=labels_here,
    textposition='top center',
    # textfont=dict(size=24)
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(yaxis=dict(
    tickmode='array',
    tickvals=y_vals,
    ticktext=['Usual care', 'Redirection rejected', 'Redirection approved']
))
fig.update_layout(yaxis_range=[y_vals[-1] - 0.5, y_vals[0] + 1])
fig.update_layout(xaxis_title='Time (minutes)')

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# fig.update_layout(
#     # autosize=False,
#     # width=500,
#     height=700
# )

st.plotly_chart(fig, use_container_width=True)


# #####################
# ##### mRS DISTS #####
# #####################

st.markdown('## Outcome model')
st.markdown('## mRS distributions')
st.markdown('Using the times to treatment in the selected LSOA.')

# Store mRS dists in here:
mrs_dists = {}
# Import no-treatment mRS dists from file:
dist_dict = load_reference_mrs_dists()
mrs_dists['nlvo_no_treatment_noncum'] = dist_dict['nlvo_no_treatment_noncum']
mrs_dists['nlvo_no_treatment'] = dist_dict['nlvo_no_treatment']
mrs_dists['lvo_no_treatment_noncum'] = dist_dict['lvo_no_treatment_noncum']
mrs_dists['lvo_no_treatment'] = dist_dict['lvo_no_treatment']
# Usual care and redirect mRS dists:
for sce in ['drip_ship', 'mothership']:
    key = 'usual_care' if sce == 'drip_ship' else 'redirect'
    for occ in ['nlvo', 'lvo']:
        for tre in ['ivt', 'mt', 'ivt_mt']:
            try:
                dist = np.array(df_mrs.loc[lsoa_name, [f'{occ}_{sce}_{tre}_mrs_dists_{i}' for i in range(7)]].to_list())
                dist_noncum = np.array(df_mrs.loc[lsoa_name, [f'{occ}_{sce}_{tre}_mrs_dists_noncum_{i}' for i in range(7)]].to_list())
                mrs_dists[f'{occ}_{tre}_{key}_noncum'] = dist_noncum
                mrs_dists[f'{occ}_{tre}_{key}'] = dist
            except KeyError:
                pass


def plot_mrs_bars_here(occ, tre):
    # Place all data and setup for plot into this dictionary.
    # The keys are used for the legend labels.
    display0 = 'Usual care'
    display1 = 'Redirection'

    mrs_lists_dict = {
        'No treatment': {
            'noncum': mrs_dists[f'{occ}_no_treatment_noncum'],
            'cum': mrs_dists[f'{occ}_no_treatment'],
            'std': None,
            'colour': 'grey',
            'linestyle': 'dot',
        },
        display0: {
            'noncum': mrs_dists[f'{occ}_{tre}_usual_care_noncum'],
            'cum': mrs_dists[f'{occ}_{tre}_usual_care'],
            'std': None,
            'colour': '#0072b2',
            'linestyle': 'dash',
        },
        display1: {
            'noncum': mrs_dists[f'{occ}_{tre}_redirect_noncum'],
            'cum': mrs_dists[f'{occ}_{tre}_redirect'],
            'std': None,
            'colour': '#56b4e9',
            'linestyle': 'dashdot',
        },
    }

    mrs.plot_mrs_bars(mrs_lists_dict)

cols = st.columns(3)
with cols[0]:
    st.write('nLVO + IVT')
    plot_mrs_bars_here('nlvo', 'ivt')
with cols[1]:
    st.write('LVO + IVT')
    plot_mrs_bars_here('lvo', 'ivt')
with cols[2]:
    st.write('LVO + MT')
    plot_mrs_bars_here('lvo', 'mt')
# with cols[3]:
#     st.write('LVO + (IVT & MT)')
#     plot_mrs_bars_here('lvo', 'ivt_mt')
#     st.write('This one will match one of the other LVO ones exactly.')
if (df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_mt_utility_shift'] == df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_utility_shift']):
    usual_care_ivt_mt_uses = 'IVT-only'
else:
    usual_care_ivt_mt_uses = 'MT-only'

if (df_lsoa.loc[lsoa_name, 'lvo_mothership_ivt_mt_utility_shift'] == df_lsoa.loc[lsoa_name, 'lvo_mothership_ivt_utility_shift']):
    redirect_ivt_mt_uses = 'IVT-only'
else:
    redirect_ivt_mt_uses = 'MT-only'

str_lvo_treatment = ''.join([
    'The LVO with IVT & MT distributions use whichever of the separate ',
    'IVT and MT distributions has the better average utility. ',
    'In this case, the "usual care" scenario uses the ',
    f'{usual_care_ivt_mt_uses} distribution and the "redirect" scenario ',
    f'uses the {redirect_ivt_mt_uses} distribution.'
    ])
st.markdown(str_lvo_treatment)

# ###########################
# ##### UTILITY SCATTER #####
# ###########################

st.markdown('### Average utility')
# util_dict = {}
# 'nlvo_no_treatment_utility'
# 'lvo_no_treatment_utility'

cols_shift = [
    'nlvo_drip_ship_ivt_utility_shift',
    'lvo_drip_ship_ivt_utility_shift',
    'lvo_drip_ship_mt_utility_shift',
    'lvo_drip_ship_ivt_mt_utility_shift',
    'nlvo_mothership_ivt_utility_shift',
    'lvo_mothership_ivt_utility_shift',
    'lvo_mothership_mt_utility_shift',
    'lvo_mothership_ivt_mt_utility_shift',
]

fig = go.Figure()

# Add background diagonal line to show parity:
util_min = np.min(df_lsoa.loc[lsoa_name, cols_shift])
util_max = np.max(df_lsoa.loc[lsoa_name, cols_shift])
# Add buffer:
util_min -= abs(util_min) * 0.1
util_max += abs(util_max) * 0.1

fig.add_trace(go.Scatter(
    x=[util_min, util_max],
    y=[util_min, util_max],
    mode='lines',
    line_color='grey',
    showlegend=False,
    hoverinfo='skip'
))

# Plot markers:
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'nlvo_drip_ship_ivt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'nlvo_mothership_ivt_utility_shift']],
    # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
    mode='markers',
    name='nLVO IVT',
))
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'lvo_mothership_ivt_utility_shift']],
    # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
    mode='markers',
    name='LVO IVT',
))
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_mt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'lvo_mothership_mt_utility_shift']],
    # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
    mode='markers',
    name='LVO MT',
))
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_mt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'lvo_mothership_ivt_mt_utility_shift']],
    marker=dict(symbol='square', color='rgba(0, 0, 0, 0)', size=10, line=dict(color='grey', width=1)),
    mode='markers',
    name='LVO IVT & MT',
))

# Plot connecting lines from markers to diagonal line:
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'nlvo_drip_ship_ivt_utility_shift'], df_lsoa.loc[lsoa_name, 'nlvo_drip_ship_ivt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'nlvo_drip_ship_ivt_utility_shift'], df_lsoa.loc[lsoa_name, 'nlvo_mothership_ivt_utility_shift']],
    line_color='grey',
    mode='lines',
    showlegend=False,
    hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_utility_shift'], df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_ivt_utility_shift'], df_lsoa.loc[lsoa_name, 'lvo_mothership_ivt_utility_shift']],
    line_color='grey',
    mode='lines',
    showlegend=False,
    hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_mt_utility_shift'], df_lsoa.loc[lsoa_name, 'lvo_drip_ship_mt_utility_shift']],
    y=[df_lsoa.loc[lsoa_name, 'lvo_drip_ship_mt_utility_shift'], df_lsoa.loc[lsoa_name, 'lvo_mothership_mt_utility_shift']],
    line_color='grey',
    mode='lines',
    showlegend=False,
    hoverinfo='skip'
))


fig.add_annotation(
    xref="x domain",
    yref="y domain",
    x=0.25,
    y=0.95,
    showarrow=False,
    text='"Usual care"<br>gives <b>worse</b> utility<br>than "redirection"',
    # arrowhead=2,
)

fig.add_annotation(
    xref="x domain",
    yref="y domain",
    x=0.75,
    y=0.05,
    showarrow=False,
    text='"Usual care"<br>gives <b>better</b> utility <br>than "redirection"',
    # arrowhead=2,
)

# Equivalent to pyplot set_aspect='equal':
fig.update_yaxes(scaleanchor='x', scaleratio=1)

# Axis limits:
fig.update_layout(xaxis_range=[util_min, util_max])
fig.update_layout(yaxis_range=[util_min, util_max])

# Axis labels:
fig.update_layout(xaxis_title='Utility shift with<br>usual care')
fig.update_layout(yaxis_title='Utility shift with<br>redirection')

# # Show vertical grid lines (makes it obvious where the labels are)
# fig.update_xaxes(showgrid=True)  # , gridwidth=1, gridcolor='LimeGreen')

st.plotly_chart(fig)



# ######################
# ##### POPULATION #####
# ######################

st.write('## Population')
st.markdown(
'''
Inputs that affect this:
+ The following ones:
'''
)

pie_cols = st.columns([1, 3])
with pie_cols[0]:
    container_occ = st.container()
with pie_cols[1]:
    container_sunburst = st.container()
input_dict = {}
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

# Proportion nLVO and LVO:
prop_nlvo = input_dict['prop_nlvo']
prop_lvo = input_dict['prop_lvo']
# Sensitivity and specificity:
prop_lvo_redirected = input_dict['sensitivity']
prop_nlvo_redirected = (1.0 - input_dict['specificity'])

# How many people are being treated?
pie_dict = {
    'nLVO': {
        'value': prop_nlvo,
        'parent': '',
        'pattern_shape': '',
    },
    'LVO': {
        'value': prop_lvo,
        'parent': '',
        'pattern_shape': '',
    },
    'nLVO - usual care': {
        'value': prop_nlvo * (1.0 - prop_nlvo_redirected),
        'parent': 'nLVO',
        'pattern_shape': '',
    },
    'nLVO - redirected': {
        'value': prop_nlvo * prop_nlvo_redirected,
        'parent': 'nLVO',
        'pattern_shape': '/',
    },
    'LVO - usual care': {
        'value': prop_lvo * (1.0 - prop_lvo_redirected),
        'parent': 'LVO',
        'pattern_shape': '',
    },
    'LVO - redirected': {
        'value': prop_lvo * prop_lvo_redirected,
        'parent': 'LVO',
        'pattern_shape': '/',
    },
}

# Plot sunburst:
fig = go.Figure()
fig.add_trace(go.Sunburst(
    labels=list(pie_dict.keys()),
    parents=[pie_dict[key]['parent'] for key in list(pie_dict.keys())],
    values=[pie_dict[key]['value'] for key in list(pie_dict.keys())],
    marker=dict(
        pattern=dict(
            shape=[pie_dict[key]['pattern_shape'] for key in list(pie_dict.keys())],
            solidity=0.9,
            )
        ),
    branchvalues='total'
))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

with container_sunburst:
    st.plotly_chart(fig)

# #############################
# ##### COMBINE mRS DISTS #####
# #############################

st.markdown('Assume for now that everyone gets treatment.')

st.markdown('Sum the mRS distributions in the proportions from the pie chart.')

st.markdown('Calculate benefit of redirection for each occlusion and treatment type.')


mrs_dists['nlvo_ivt_combo_noncum'] = (
    ((1.0 - prop_nlvo_redirected) * mrs_dists["nlvo_ivt_usual_care_noncum"]) +
    (prop_nlvo_redirected * mrs_dists["nlvo_ivt_redirect_noncum"])
)
mrs_dists['lvo_ivt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["lvo_ivt_usual_care_noncum"]) +
    (prop_lvo_redirected * mrs_dists["lvo_ivt_redirect_noncum"])
)
mrs_dists['lvo_mt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["lvo_mt_usual_care_noncum"]) +
    (prop_lvo_redirected * mrs_dists["lvo_mt_redirect_noncum"])
)
mrs_dists['lvo_ivt_mt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["lvo_ivt_mt_usual_care_noncum"]) +
    (prop_lvo_redirected * mrs_dists["lvo_ivt_mt_redirect_noncum"])
)

st.markdown('### Example: combining "usual care" and "redirected" data')
st.markdown('Options for bar chart mRS distributions:')
occ = st.radio('Occlusion type', ['lvo', 'nlvo'])
tre = st.radio('Treatment type', ['ivt', 'mt'], index=1)
prop = prop_lvo_redirected if occ == 'lvo' else prop_nlvo_redirected

plot_dists = {}
# Step 1: Just the mRS dists.
plot_dists['step1_usual'] = mrs_dists[f'{occ}_{tre}_usual_care_noncum']
plot_dists['step1_redir'] = mrs_dists[f'{occ}_{tre}_redirect_noncum']
# Step 2: mRS dists scaled.
plot_dists['step2_usual'] = (
    (1.0 - prop) * mrs_dists[f'{occ}_{tre}_usual_care_noncum'])
plot_dists['step2_redir'] = prop * mrs_dists[f'{occ}_{tre}_redirect_noncum']
# Step 3: Place mRS dists side-by-side.
plot_dists['step3'] = np.append(plot_dists['step2_usual'], plot_dists['step2_redir'])
# Step 4: Rearrange the mRS bins.
plot_dists['step4']  = np.stack((plot_dists['step2_usual'], plot_dists['step2_redir']), axis=-1).flatten()
# Step 5: Done.
plot_dists['step5']  = np.sum(plot_dists['step4'].reshape(7, 2), axis=1)


# Define built-in colours for mRS bands:
# Colours as of 16th January 2023:
# (the first six are from seaborn-colorblind)
colour_list = [
    "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9",
    "DarkSlateGray"  # mRS=6
    ]

fig = go.Figure()

for i, width in enumerate(plot_dists['step1_usual']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[0],
        marker=dict(color=colour_list[i]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=True,
        width=0.7  # skinniness of bars
))
for i, width in enumerate(plot_dists['step1_redir']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[-1],
        marker=dict(color=colour_list[i]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=False,
        width=0.7  # skinniness of bars
))


for i, width in enumerate(plot_dists['step2_usual']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[-3],
        marker=dict(color=colour_list[i]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=False,
        width=0.7  # skinniness of bars
))
for i, width in enumerate(plot_dists['step2_redir']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[-4],
        marker=dict(color=colour_list[i]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=False,
        width=0.7  # skinniness of bars
))


for i, width in enumerate(plot_dists['step3']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[-6],
        marker=dict(color=colour_list[i % 7]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=False,
        width=0.7  # skinniness of bars
))

# for i, width in enumerate(plot_dists['step4']):
#     fig.add_trace(go.Bar(
#         x=[width],
#         y=[-8],
#         marker=dict(color=colour_list[int(0.5*i)]),
#         orientation='h',
#         name=f'mRS={i}',
#         showlegend=False,
#         width=0.7  # skinniness of bars
# ))

for i, width in enumerate(plot_dists['step5']):
    fig.add_trace(go.Bar(
        x=[width],
        y=[-8],
        marker=dict(color=colour_list[i]),
        orientation='h',
        name=f'mRS={i}',
        showlegend=False,
        width=0.7  # skinniness of bars
))

# Change the bar mode
fig.update_layout(barmode='stack')
# Format legend:
fig.update_layout(legend=dict(
    orientation='h',      # horizontal
    traceorder='normal',  # Show mRS=0 on left
))
# Y tick labels:
yticks = [1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
yticklabels = [
    '<b>Step 1</b>', 'Usual care', 'Redirected',
    '<b>Step 2</b>', f'Scaled usual care (x{(1.0 - prop):.3f})', f'Scaled redirected (x{prop:.3f})',
    '<b>Step 3</b>', 'Summed scaled usual care<br>and redirected',
    '<b>Step 4</b>', 'Sorted summed scaled<br>usual care and redirected'
    ]
fig.update_layout(yaxis=dict(
    tickmode='array',
    tickvals=yticks,
    ticktext=yticklabels
))
fig.update_layout(yaxis_range=[-8.5, 1.5])

st.plotly_chart(fig)


st.markdown('### Combining geographical regions')

st.markdown('Weight LSOA by number of admissions to stroke units.')