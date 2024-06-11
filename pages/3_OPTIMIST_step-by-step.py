"""
Extra notebook to show the OPTIMIST app workings in more detail.

"Usual care" is either go to an IVT unit and then an MT unit,
or go to the nearest unit which happens to have IVT and MT.
"Redirect" adds the extra time for diagnostic and can still
have either of those two options.

# TO DO - currently the model completely skips the process_ambulance_on_scene_diagnostic_duration bit!
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
        pathway_dict = inputs.select_parameters_pathway_optimist()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        df_unit_services, df_unit_services_full = (
            inputs.select_stroke_unit_services(use_msu=False))

        # Button for completing the form
        # (so script only re-runs once it is pressed, allows changes
        # to multiple widgets at once.)
        submitted = st.form_submit_button('Submit')

cols = st.columns([6, 4], gap='large')
# Keep an empty middle column to adjust the gap between plots.
with cols[0]:
    container_timeline_plot = st.container()
with cols[1]:
    container_timeline_info = st.container()

pie_cols = st.columns([1, 3])
with pie_cols[0]:
    container_occ = st.container()
with pie_cols[1]:
    container_sunburst = st.container()

with container_occ:
    population_dict = inputs.select_parameters_population_optimist()


# #######################################
# ########## MAIN CALCULATIONS ##########
# #######################################

# Process LSOA and calculate outcomes:
# df_lsoa, df_mrs = calc.calculate_outcomes(
#     pathway_dict, df_unit_services)
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
    df_outcome_ra = calc.make_outcome_inputs_redirection_approved(
        input_dict, df_travel_times)
    df_outcome_rr = calc.make_outcome_inputs_redirection_rejected(
        input_dict, df_travel_times)
    dict_outcome_inputs = {
        'usual_care': df_outcome_uc,
        'redirection_approved': df_outcome_ra,
        'redirection_rejected': df_outcome_rr,
    }

    # Process LSOA and calculate outcomes:
    df_lsoa, df_mrs = calc.calculate_outcomes(
        dict_outcome_inputs, df_unit_services, geo.combined_data)

    # Extra calculations for redirection:
    # Combine redirection rejected and approved results in
    # proportions given by specificity and sensitivity.
    # This creates columns labelled "redirection_approved".
    redirect_dict = {
        'sensitivity': input_dict['sensitivity'],
        'specificity': input_dict['specificity'],
    }
    df_lsoa = calc.combine_results_by_redirection(df_lsoa, redirect_dict)
    df_mrs = calc.combine_results_by_redirection(
        df_mrs, redirect_dict, combine_mrs_dists=True)

    # Make combined nLVO + LVO data in the proportions given:
    # Combine for "usual care":
    prop_dict = {
        'nlvo': input_dict['prop_nlvo'],
        'lvo': input_dict['prop_lvo']
    }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['usual_care'])
    df_mrs = calc.combine_results_by_occlusion_type(
        df_mrs, prop_dict, combine_mrs_dists=True,
        scenario_list=['usual_care'])
    # Combine for redirection considered:
    prop_dict = {
        'nlvo': input_dict['prop_redirection_considered_nlvo'],
        'lvo': input_dict['prop_redirection_considered_lvo']
    }
    df_lsoa = calc.combine_results_by_occlusion_type(
        df_lsoa, prop_dict, scenario_list=['redirection_considered'])
    df_mrs = calc.combine_results_by_occlusion_type(
        df_mrs, prop_dict, combine_mrs_dists=True,
        scenario_list=['redirection_considered'])
    # Don't calculate the separate redirection approved/rejected bits.

    # Calculate diff - redirect minus usual care:
    df_lsoa = calc.combine_results_by_diff(
        df_lsoa,
        scenario_types=['redirection_considered', 'usual_care']
        )
    df_mrs = calc.combine_results_by_diff(
        df_mrs,
        scenario_types=['redirection_considered', 'usual_care'],
        combine_mrs_dists=True
        )

    df_icb, df_isdn, df_nearest_ivt = calc.group_results_by_region(
        df_lsoa, df_unit_services)

    return df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt

df_lsoa, df_mrs, df_icb, df_isdn, df_nearest_ivt = main_calculations(
    pathway_dict | population_dict,
    df_unit_services
    )

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

    time_dicts = {
        'prehosp_usual_care': time_dict_prehosp_usual_care,
        'prehosp_redirect': time_dict_prehosp_redirect,
        'ivt_only_unit': time_dict_ivt_only_unit,
        'mt_transfer_unit': time_dict_mt_transfer_unit,
        'ivt_mt_unit': time_dict_ivt_mt_unit,
    }
    return time_dicts


time_dicts = build_time_dicts(pathway_dict)

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
        'text': 'Ambulance<br>arrives<br>on scene',
    },
    'process_ambulance_on_scene_diagnostic_duration': {
        'emoji': '\U0000260E',
        'text':  'Extra time<br>for<br>diagnostic',
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
    # 'MSU<br>leaves base': {
        # 'emoji': '\U0001f691',
        # 'text': 
    # },
    # 'MSU<br>arrives on scene': {
        # 'emoji': '\U0001f691',
        # 'text': 
    # },
    # 'MSU<br>leaves scene': {
        # 'emoji': '\U0001f691',
        # 'text': 
    # },
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



def plot_timeline(time_dicts, timeline_display_dict, y_vals, y_labels):
    # Pre-hospital timelines
    fig = go.Figure()

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
        labels_here = [f'{timeline_display_dict[key]["text"]}<br><br><br>'
                       for key in list(time_dict.keys())]
        fig.add_trace(go.Scatter(
            x=cum_times,
            y=[y_vals[i]]*len(cum_times),
            mode='lines+markers+text',
            text=emoji_here,
            marker=dict(symbol='line-ns', size=10,
                        line_width=2, line_color='grey'),
            line_color='grey',
            textposition='top center',
            textfont=dict(size=24),
            name=time_name,
            showlegend=False,
            hoverinfo='x'
        ))
        fig.add_trace(go.Scatter(
            x=cum_times,
            y=[y_vals[i]]*len(cum_times),
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
        ticktext=y_labels
    ))
    fig.update_layout(yaxis_range=[y_vals[-1] - 0.5, y_vals[0] + 1])
    fig.update_layout(xaxis_title='Time (minutes)')

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(
        # autosize=False,
        # width=500,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


with container_timeline_plot:
    plot_timeline(
        time_dicts, timeline_display_dict, y_vals=[0, -1, -3, -4, -5],
        y_labels=[
            'Usual care', 'Redirection',
            'IVT-only unit', 'Transfer to MT unit', 'IVT & MT unit'
            ])


# --- Calculate times to treatment, show table ---

# Usual care:
time_to_ivt_without_travel_usual_care = (
    np.sum(list(time_dicts['prehosp_usual_care'].values())) +
    time_dicts['ivt_only_unit']['arrival_to_needle']
)
time_to_mt_no_transfer_without_travel_usual_care = (
    np.sum(list(time_dicts['prehosp_usual_care'].values())) +
    time_dicts['ivt_mt_unit']['arrival_to_needle'] +
    time_dicts['ivt_mt_unit']['needle_to_puncture']
)
time_to_mt_with_transfer_without_travel_usual_care = (
    np.sum(list(time_dicts['prehosp_usual_care'].values())) +
    np.sum(list(time_dicts['ivt_only_unit'].values())) +
    time_dicts['mt_transfer_unit']['arrival_to_puncture']
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
mask = df_lsoa['transfer_unit_name'].isin([selected_unit])
df_lsoa = df_lsoa.loc[mask].copy()
mask_mrs = df_mrs.index.isin(df_lsoa.index)
df_mrs = df_mrs.loc[mask_mrs].copy()

# st.markdown('These tables are for Anna\'s reference, ignore them please')
# st.write(df_lsoa)
# st.write(df_mrs)


# ############################
# ########## TIME MAPS #######
# ############################

# Load in LSOA, limit to only selected,
# assign colours by time, show four subplots (IVT/MT usual/redirect).
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
        df_lsoa['usual_care_ivt_time'],
        'usual_care_ivt_time',
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
        df_lsoa['usual_care_mt_time'],
        'usual_care_mt_time',
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
        df_lsoa['redirection_approved_ivt_time'],
        'redirection_approved_ivt_time',
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
        df_lsoa['redirection_approved_mt_time'],
        'redirection_approved_mt_time',
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

def scatter_ivt_mt_times(df_lsoa):
    fig = go.Figure()

    # Add background diagonal line to show parity:
    time_min = np.min([df_lsoa[['redirection_approved_ivt_time', 'usual_care_ivt_time',
                                'redirection_approved_mt_time', 'usual_care_mt_time']]])
    time_max = np.max([df_lsoa[['redirection_approved_ivt_time', 'usual_care_ivt_time',
                                'redirection_approved_mt_time', 'usual_care_mt_time']]])
    fig.add_trace(go.Scatter(
        x=[time_min, time_max],
        y=[time_min, time_max],
        mode='lines',
        line_color='grey',
        showlegend=False,
        hoverinfo='skip'
    ))


    # Plot connecting lines from markers to diagonal line:
    x_lines_ivt = np.stack((df_lsoa['usual_care_ivt_time'],
                            df_lsoa['usual_care_ivt_time'],
                            [None]*len(df_lsoa)), axis=-1).flatten()
    y_lines_ivt = np.stack((df_lsoa['usual_care_ivt_time'],
                            df_lsoa['redirection_approved_ivt_time'],
                            [None]*len(df_lsoa)), axis=-1).flatten()
    x_lines_mt = np.stack((df_lsoa['usual_care_mt_time'],
                        df_lsoa['usual_care_mt_time'],
                        [None]*len(df_lsoa)), axis=-1).flatten()
    y_lines_mt = np.stack((df_lsoa['usual_care_mt_time'],
                        df_lsoa['redirection_approved_mt_time'],
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

    # # Plot lines between IVT and MT:
    # x_lines = np.stack((df_lsoa['usual_care_ivt_time'],
    #                     df_lsoa['usual_care_mt_time'],
    #                     [None]*len(df_lsoa)), axis=-1).flatten()
    # y_lines = np.stack((df_lsoa['redirection_approved_ivt_time'],
    #                     df_lsoa['redirection_approved_mt_time'],
    #                     [None]*len(df_lsoa)), axis=-1).flatten()
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
        x=df_lsoa['usual_care_ivt_time'],
        y=df_lsoa['redirection_approved_ivt_time'],
        customdata=np.stack(
            (df_lsoa.index.values, df_lsoa['nearest_ivt_unit']), axis=-1),
        # customdata=[df_lsoa.index.values],
        marker=dict(symbol='circle', color='red',
                    line=dict(color='black', width=0.5)),
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
        x=df_lsoa['usual_care_mt_time'],
        y=df_lsoa['redirection_approved_mt_time'],
        customdata=np.stack(
            (df_lsoa.index.values, df_lsoa['nearest_ivt_unit']), axis=-1),
        marker=dict(symbol='square', color='cyan',
                    line=dict(color='black', width=0.5)),
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

scatter_ivt_mt_times(df_lsoa)

# --- Pick out some extreme values ---
# Calculate time tradeoff:
time_ivt_diff = (df_lsoa['redirection_approved_ivt_time'] -
                 df_lsoa['usual_care_ivt_time'])
time_mt_diff = (df_lsoa['redirection_approved_mt_time'] -
                df_lsoa['usual_care_mt_time'])
time_metric = time_ivt_diff + time_mt_diff
# Redirection is best when time_ivt_diff is small (preferably zero) *and*
# time_mt_diff is minimum (preferably below zero).
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
st.markdown(f'A transfer is {str_transfer_required}required for MT.')


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

# TO DO - adding on the extra diagnostic time shouldn't be necessary if the model is set up correctly ----------------------------------
arr_times = [
    [df_lsoa.loc[lsoa_name, 'usual_care_ivt_time'],
     df_lsoa.loc[lsoa_name, 'usual_care_mt_time'],
     ],
    [df_lsoa.loc[lsoa_name, 'usual_care_ivt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     df_lsoa.loc[lsoa_name, 'usual_care_mt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     ],
    [df_lsoa.loc[lsoa_name, 'redirection_approved_ivt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
     df_lsoa.loc[lsoa_name, 'redirection_approved_mt_time'] + pathway_dict['process_ambulance_on_scene_diagnostic_duration'],
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
        time_dicts['ivt_only_unit'] |
        {'transfer_time': df_lsoa.loc[lsoa_name, 'transfer_time']} |
        time_dicts['mt_transfer_unit']
    )
dict_no_transfer = time_dicts['ivt_mt_unit']
if transfer_required:
    dict2 = dict_transfer
else:
    dict2 = dict_no_transfer

time_travel_dicts = {}
time_travel_dicts['usual_care'] = (
    time_dicts['prehosp_usual_care'] |
    {'nearest_ivt_time': df_lsoa.loc[lsoa_name, 'nearest_ivt_time']} |
    dict2
)
time_travel_dicts['redirect_but_not'] = (
    time_dicts['prehosp_redirect'] |
    {'nearest_ivt_time': df_lsoa.loc[lsoa_name, 'nearest_ivt_time']} |
    dict2
)
time_travel_dicts['redirect_but_yes'] = (
    time_dicts['prehosp_redirect'] |
    {'nearest_mt_time': df_lsoa.loc[lsoa_name, 'nearest_mt_time']} |
    dict_no_transfer
)

plot_timeline(
    time_travel_dicts, timeline_display_dict, y_vals=[0, -1, -2],
    y_labels=['Usual care', 'Redirection rejected', 'Redirection approved']
    )

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
for sce in ['usual_care', 'redirection_approved']:
    key = 'usual_care' if sce == 'usual_care' else 'redirect'
    for occ in ['nlvo', 'lvo']:
        for tre in ['ivt', 'mt', 'ivt_mt']:
            try:
                dist = np.array(df_mrs.loc[lsoa_name, [f'{sce}_{occ}_{tre}_mrs_dists_{i}' for i in range(7)]].to_list())
                dist_noncum = np.array(df_mrs.loc[lsoa_name, [f'{sce}_{occ}_{tre}_mrs_dists_noncum_{i}' for i in range(7)]].to_list())
                mrs_dists[f'{key}_{occ}_{tre}_noncum'] = dist_noncum
                mrs_dists[f'{key}_{occ}_{tre}'] = dist
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
            'noncum': mrs_dists[f'usual_care_{occ}_{tre}_noncum'],
            'cum': mrs_dists[f'usual_care_{occ}_{tre}'],
            'std': None,
            'colour': '#0072b2',
            'linestyle': 'dash',
        },
        display1: {
            'noncum': mrs_dists[f'redirect_{occ}_{tre}_noncum'],
            'cum': mrs_dists[f'redirect_{occ}_{tre}'],
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

# Write which data the LVO with IVT & MT distribution uses.
if (df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_mt_utility_shift'] == df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_utility_shift']):
    usual_care_ivt_mt_uses = 'IVT-only'
else:
    usual_care_ivt_mt_uses = 'MT-only'

if (df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_ivt_mt_utility_shift'] == df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_ivt_utility_shift']):
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


def plot_average_utility(df_lsoa, lsoa_name):
    fig = go.Figure()

    cols_shift = [
        'usual_care_nlvo_ivt_utility_shift',
        'usual_care_lvo_ivt_utility_shift',
        'usual_care_lvo_mt_utility_shift',
        'usual_care_lvo_ivt_mt_utility_shift',
        'redirection_approved_nlvo_ivt_utility_shift',
        'redirection_approved_lvo_ivt_utility_shift',
        'redirection_approved_lvo_mt_utility_shift',
        'redirection_approved_lvo_ivt_mt_utility_shift',
    ]
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
        x=[df_lsoa.loc[lsoa_name, 'usual_care_nlvo_ivt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'redirection_approved_nlvo_ivt_utility_shift']],
        # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
        mode='markers',
        name='nLVO IVT',
    ))
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_ivt_utility_shift']],
        # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
        mode='markers',
        name='LVO IVT',
    ))
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_mt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_mt_utility_shift']],
        # marker=dict(symbol='circle', color='red', line=dict(color='black', width=0.5)),
        mode='markers',
        name='LVO MT',
    ))
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_mt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_ivt_mt_utility_shift']],
        marker=dict(symbol='square', color='rgba(0, 0, 0, 0)',
                    size=10, line=dict(color='grey', width=1)),
        mode='markers',
        name='LVO IVT & MT',
    ))

    # Plot connecting lines from markers to diagonal line:
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_nlvo_ivt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'usual_care_nlvo_ivt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'usual_care_nlvo_ivt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'redirection_approved_nlvo_ivt_utility_shift']],
        line_color='grey',
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_ivt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_ivt_utility_shift']],
        line_color='grey',
        mode='lines',
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_mt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'usual_care_lvo_mt_utility_shift']],
        y=[df_lsoa.loc[lsoa_name, 'usual_care_lvo_mt_utility_shift'],
           df_lsoa.loc[lsoa_name, 'redirection_approved_lvo_mt_utility_shift']],
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


plot_average_utility(df_lsoa, lsoa_name)


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

# Proportion nLVO and LVO:
prop_nlvo = population_dict['prop_nlvo']
prop_lvo = population_dict['prop_lvo']
# Proportion of these who are considered for redirection:
prop_nlvo_redirect_considered = population_dict['prop_nlvo_redirection_considered']
prop_lvo_redirect_considered = population_dict['prop_lvo_redirection_considered']
# Sensitivity and specificity:
prop_lvo_redirected = population_dict['sensitivity']
prop_nlvo_redirected = (1.0 - population_dict['specificity'])

colour_nlvo = 'PaleGoldenrod'
colour_lvo = 'SkyBlue'


# How many people are being treated?
pie_usual_dict = {
    'nLVO': {
        'value': prop_nlvo,
        'parent': '',
        'pattern_shape': '',
        'colour': colour_nlvo,
        'label': 'nLVO',
    },
    'LVO': {
        'value': prop_lvo,
        'parent': '',
        'pattern_shape': '',
        'colour': colour_lvo,
        'label': 'LVO',
    },
}

# Plot sunburst:
fig = go.Figure()
fig.add_trace(go.Sunburst(
    ids=list(pie_usual_dict.keys()),
    parents=[pie_usual_dict[key]['parent'] for key in list(pie_usual_dict.keys())],
    labels=[pie_usual_dict[key]['label'] for key in list(pie_usual_dict.keys())],
    values=[pie_usual_dict[key]['value'] for key in list(pie_usual_dict.keys())],
    marker=dict(
        pattern=dict(
            shape=[pie_usual_dict[key]['pattern_shape'] for key in list(pie_usual_dict.keys())],
            solidity=0.9,
            ),
        colors=[pie_usual_dict[key]['colour'] for key in list(pie_usual_dict.keys())]
        ),
    branchvalues='total'
))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

with container_sunburst:
    st.markdown('__Usual population__:')
    st.plotly_chart(fig)

# How many people are being treated?
pie_dict = {
    'nLVO': {
        'value': prop_nlvo,
        'parent': '',
        'pattern_shape': '',
        'colour': colour_nlvo,
        'label': 'nLVO',
    },
    'LVO': {
        'value': prop_lvo,
        'parent': '',
        'pattern_shape': '',
        'colour': colour_lvo,
        'label': 'LVO',
    },
    'nLVO - usual care': {
        'value': prop_nlvo * (1.0 - prop_nlvo_redirect_considered),
        'parent': 'nLVO',
        'pattern_shape': '',
        'colour': colour_nlvo,
        'label': 'Usual care',
    },
    'nLVO - redirection considered': {
        'value': prop_nlvo * prop_nlvo_redirect_considered,
        'parent': 'nLVO',
        'pattern_shape': '.',
        'colour': 'LimeGreen',
        'label': 'Redirection<br>considered',
    },
    'nLVO - redirection approved': {
        'value': prop_nlvo * prop_nlvo_redirect_considered * prop_nlvo_redirected,
        'parent': 'nLVO - redirection considered',
        'pattern_shape': '.',
        'colour': colour_nlvo,
        'label': 'Approved',
    },
    'nLVO - redirection rejected': {
        'value': prop_nlvo * prop_nlvo_redirect_considered * (1.0 - prop_nlvo_redirected),
        'parent': 'nLVO - redirection considered',
        'pattern_shape': '',
        'colour': colour_nlvo,
        'label': 'Rejected',
    },
    'LVO - usual care': {
        'value': prop_lvo * (1.0 - prop_lvo_redirect_considered),
        'parent': 'LVO',
        'pattern_shape': '',
        'colour': colour_lvo,
        'label': 'Usual care',
    },
    'LVO - redirection considered': {
        'value': prop_lvo * prop_lvo_redirect_considered,
        'parent': 'LVO',
        'pattern_shape': '.',
        'colour': 'LimeGreen',
        'label': 'Redirection<br>considered',
    },
    'LVO - redirection approved': {
        'value': prop_lvo * prop_lvo_redirect_considered * prop_lvo_redirected,
        'parent': 'LVO - redirection considered',
        'pattern_shape': '.',
        'colour': colour_lvo,
        'label': 'Approved',
    },
    'LVO - redirection rejected': {
        'value': prop_lvo * prop_lvo_redirect_considered * (1.0 - prop_lvo_redirected),
        'parent': 'LVO - redirection considered',
        'pattern_shape': '',
        'colour': colour_lvo,
        'label': 'Rejected',
    },
}

# Plot sunburst:
fig = go.Figure()
fig.add_trace(go.Sunburst(
    ids=list(pie_dict.keys()),
    parents=[pie_dict[key]['parent'] for key in list(pie_dict.keys())],
    labels=[pie_dict[key]['label'] for key in list(pie_dict.keys())],
    values=[pie_dict[key]['value'] for key in list(pie_dict.keys())],
    marker=dict(
        pattern=dict(
            shape=[pie_dict[key]['pattern_shape'] for key in list(pie_dict.keys())],
            solidity=0.9,
            ),
        colors=[pie_dict[key]['colour'] for key in list(pie_dict.keys())]
        ),
    branchvalues='total'
))
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

with container_sunburst:
    st.markdown('__SPEEDY population in green__:')
    st.plotly_chart(fig)

# #############################
# ##### COMBINE mRS DISTS #####
# #############################

st.markdown('Assume for now that everyone gets treatment.')

st.markdown('Sum the mRS distributions in the proportions from the pie chart.')

st.markdown('Calculate benefit of redirection for each occlusion and treatment type.')


mrs_dists['nlvo_ivt_combo_noncum'] = (
    ((1.0 - prop_nlvo_redirected) * mrs_dists["usual_care_nlvo_ivt_noncum"]) +
    (prop_nlvo_redirected * mrs_dists["redirect_nlvo_ivt_noncum"])
)
mrs_dists['lvo_ivt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["usual_care_lvo_ivt_noncum"]) +
    (prop_lvo_redirected * mrs_dists["redirect_lvo_ivt_noncum"])
)
mrs_dists['lvo_mt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["usual_care_lvo_mt_noncum"]) +
    (prop_lvo_redirected * mrs_dists["redirect_lvo_mt_noncum"])
)
mrs_dists['lvo_ivt_mt_combo_noncum'] = (
    ((1.0 - prop_lvo_redirected) * mrs_dists["usual_care_lvo_ivt_mt_noncum"]) +
    (prop_lvo_redirected * mrs_dists["redirect_lvo_ivt_mt_noncum"])
)

st.markdown('### Example: combining "usual care" and "redirected" data')
st.markdown('Options for bar chart mRS distributions:')
occ = st.radio('Occlusion type', ['lvo', 'nlvo'])
tre = st.radio('Treatment type', ['ivt', 'mt'], index=1)
prop = prop_lvo_redirected if occ == 'lvo' else prop_nlvo_redirected

plot_dists = {}
# Step 1: Just the mRS dists.
plot_dists['step1_usual'] = mrs_dists[f'usual_care_{occ}_{tre}_noncum']
plot_dists['step1_redir'] = mrs_dists[f'redirect_{occ}_{tre}_noncum']
# Step 2: mRS dists scaled.
plot_dists['step2_usual'] = (
    (1.0 - prop) * mrs_dists[f'usual_care_{occ}_{tre}_noncum'])
plot_dists['step2_redir'] = prop * mrs_dists[f'redirect_{occ}_{tre}_noncum']
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
