"""
Extra notebook to show the OPTIMIST app workings in more detail.
"""
# ----- Imports -----
import streamlit as st
import numpy as np
import pandas as pd

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

st.markdown('## Timeline')
st.markdown('The pathway timings add up to these treatment times (__excluding__ travel times):')

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
st.dataframe(df_times)



# ##################################
# ########## TIME COMPARISON #######
# ##################################

st.markdown('## Time comparison')
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

st.markdown('These tables are for Anna\'s reference, ignore them please')
st.write(df_lsoa)
st.write(df_mrs)

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

plot_the_maps_please = False
if plot_the_maps_please:
    plot_maps.plotly_time_maps(
        [gdf_usual_ivt, gdf_redirect_ivt, gdf_usual_mt, gdf_redirect_mt],
        [colour_dict_usual_ivt, colour_dict_redirect_ivt, colour_dict_usual_mt, colour_dict_redirect_mt],
        subplot_titles=subplot_titles,
        use_discrete_cmap=use_discrete_cmap
        )

# ##################################
# ########## TIME SCATTER ##########
# ##################################
import plotly.graph_objects as go
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
fig.add_trace(go.Scatter(
    x=x_lines,
    y=y_lines,
    mode='lines',
    line_color='Salmon',
    line_width=0.5,
    showlegend=False,
    hoverinfo='skip'
))
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
    marker=dict(symbol='square', color='FireBrick', line=dict(color='black', width=0.5)),
    mode='markers',
    name='MT',
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>' +
        'Usual care MT: %{x}<br>' +
        'Redirection MT: %{y}' +
        '<extra></extra>'
    )
))

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

st.markdown('## Results for one LSOA')

# Pick out an LSOA:
# lsoa_name = 'East Devon 005D'
lsoa_name = st.selectbox('LSOA', df_lsoa.index)

st.markdown('### Treatment times')
str_transfer_required = (
    '' if df_lsoa.loc[lsoa_name, 'transfer_required'] == True else 'not ')
st.markdown(f'A transfer is {str_transfer_required} required for MT.')

arr_times = [
    [df_lsoa.loc[lsoa_name, 'drip_ship_ivt_time'],
     df_lsoa.loc[lsoa_name, 'drip_ship_mt_time'],
     ],
    [df_lsoa.loc[lsoa_name, 'mothership_ivt_time'],
     df_lsoa.loc[lsoa_name, 'mothership_mt_time'],
     ]
    ]
df_times = pd.DataFrame(
    arr_times,
    columns=['IVT', 'MT'],
    index=['Usual care', 'Redirected']
)
st.dataframe(df_times)

# #####################
# ##### mRS DISTS #####
# #####################

st.markdown('### mRS distributions')

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

st.write('## Population here')

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
tre = st.radio('Treatment type', ['ivt', 'mt'])
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
