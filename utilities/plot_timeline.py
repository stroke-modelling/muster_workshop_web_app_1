"""
Plots a timeline for the patient pathway for cases 1 and 2.
Includes labelled points and emoji.

Initially copied over from the stroke outcome app.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# from outcome_utilities.fixed_params import emoji_text_dict, plotly_colours


def build_data_for_timeline(params_dict, use_drip_ship=True, use_mothership=True, use_msu=False):
    """
    params_dict keys include:
    # Travel times:
    'nearest_ivt_time', 'nearest_mt_time', 'transfer_time',
    'nearest_msu_time',
    # Actual values for reference:
    'drip_ship_ivt_time', 'drip_ship_mt_time',
    'mothership_ivt_time', 'mothership_mt_time',
    'msu_ivt_time', 'msu_mt_time',
    # Input dict:
    'process_time_call_ambulance'
    'process_time_ambulance_response',
    'process_ambulance_on_scene_duration',
    'process_msu_dispatch',
    'process_msu_thrombolysis',
    'process_msu_on_scene_post_thrombolysis',
    'process_time_arrival_to_needle',
    'transfer_time_delay',
    'process_time_arrival_to_puncture',
    'process_time_transfer_arrival_to_puncture',
    'process_time_msu_arrival_to_puncture'
    """
    times_dicts = {}
    # times_cum_dicts = {}
    # times_cum_label_dicts = {}
    # Onset time:
    time_dict = {'onset': 0}

    if use_drip_ship:
        times_keys_drip_ship = [
            'process_time_call_ambulance',
            'process_time_ambulance_response',
            'process_ambulance_on_scene_duration',
            'nearest_ivt_time',
            'process_time_arrival_to_needle',
            'transfer_time_delay',
            'transfer_time',
            'process_time_transfer_arrival_to_puncture',
        ]
        times_values_drip_ship = [params_dict[k] for k in times_keys_drip_ship]
        times_dict_drip_ship = time_dict | dict(zip(times_keys_drip_ship, times_values_drip_ship))
        times_dicts['drip_ship'] = times_dict_drip_ship

        # times_cum_drip_ship = build_cumulative_times(list(times_dict_drip_ship.values()))
        # keys_cum_drip_ship = [
        #     'Onset',
        #     'Call<br>ambulance',
        #     'Ambulance<br>arrives on scene',
        #     'Ambulance<br>leaves scene',
        #     'IVT unit<br>arrival',
        #     'IVT',
        #     'IVT unit<br>departure',
        #     'MT unit<br>arrival',
        #     'MT',
        # ]
        # # Convert to dict:
        # times_cum_dict_drip_ship = dict(
        #     zip(keys_cum_drip_ship, times_cum_drip_ship))
        # # Special case in drip-and-ship -
        # # the time delay for transfer is counted from arrival at unit, not
        # # from time to IVT which is the result of build_cumulative_times().
        # times_cum_dict_drip_ship['IVT unit<br>departure'] = (
        #         times_cum_dict_drip_ship['IVT unit<br>arrival'] +
        #         times_dict_drip_ship['transfer_time_delay']
        #     )
        # times_cum_dict_drip_ship['MT unit<br>arrival'] = (
        #         times_cum_dict_drip_ship['IVT unit<br>departure'] +
        #         times_dict_drip_ship['transfer_time']
        #     )
        # times_cum_dict_drip_ship['MT'] = (
        #         times_cum_dict_drip_ship['MT unit<br>arrival'] +
        #         times_dict_drip_ship['process_time_transfer_arrival_to_puncture']
        #     )
        # times_cum_dicts['drip_ship'] = times_cum_dict_drip_ship
    
        # times_cum_label_dict_drip_ship = (
        #     build_formatted_time_str_lists_for_scenarios(times_cum_dict_drip_ship))
        # times_cum_label_dicts['drip_ship'] = times_cum_label_dict_drip_ship

    
    if use_mothership:
        times_keys_mothership = [
            'process_time_call_ambulance',
            'process_time_ambulance_response',
            'process_ambulance_on_scene_duration',
            'nearest_mt_time',
            'process_time_arrival_to_needle',
            'process_time_arrival_to_puncture',
        ]
        times_values_mothership = [params_dict[k] for k in times_keys_mothership]
        times_dict_mothership = time_dict | dict(zip(times_keys_mothership, times_values_mothership))
        times_dicts['mothership'] = times_dict_mothership

        # times_cum_mothership = build_cumulative_times(list(times_dict_mothership.values()))
        # keys_cum_mothership = [
        #     'Onset',
        #     'Call<br>ambulance',
        #     'Ambulance<br>arrives on scene',
        #     'Ambulance<br>leaves scene',
        #     'MT unit<br>arrival',
        #     'IVT',
        #     'MT',
        # ]
        # # Convert to dict:
        # times_cum_dict_mothership = dict(
        #     zip(keys_cum_mothership, times_cum_mothership))
        # # Special case in mothership -
        # # the time to MT is counted from arrival at unit, not
        # # from time to IVT which is the result of build_cumulative_times().
        # times_cum_dict_mothership['MT'] = (
        #     times_cum_dict_mothership['MT unit<br>arrival'] +
        #     times_dict_mothership['process_time_arrival_to_puncture']
        # )
        # times_cum_dicts['mothership'] = times_cum_dict_mothership

        # times_cum_label_dict_mothership = (
        #     build_formatted_time_str_lists_for_scenarios(times_cum_dict_mothership))
        # times_cum_label_dicts['mothership'] = times_cum_label_dict_mothership


    if use_msu:
        times_keys_msu = [
            'process_time_call_ambulance',
            'process_msu_dispatch',
            'nearest_msu_time',
            'process_msu_thrombolysis',
            'process_msu_on_scene_post_thrombolysis',
            'nearest_mt_time',
            'process_time_msu_arrival_to_puncture',
        ]
        times_values_msu = [params_dict[k] for k in times_keys_msu]
        times_dict_msu = time_dict | dict(zip(times_keys_msu, times_values_msu))
        times_dicts['msu'] = times_dict_msu

        # times_cum_msu = build_cumulative_times(list(times_dict_msu.values()))

        # keys_cum_msu = [
        #     'Onset',
        #     'Call<br>ambulance',
        #     'MSU<br>leaves base',
        #     'MSU<br>arrives on scene',
        #     'IVT',
        #     'MSU<br>leaves scene',
        #     'MT unit<br>arrival',
        #     'MT',
        # ]
        # # Convert to dict:
        # times_cum_dict_msu = dict(
        #     zip(keys_cum_msu, times_cum_msu))
        # times_cum_dicts['msu'] = times_cum_dict_msu

        # times_cum_label_dict_msu = (
        #     build_formatted_time_str_lists_for_scenarios(times_cum_dict_msu))
        # times_cum_label_dicts['msu'] = times_cum_label_dict_msu

    return (times_dicts
            # times_cum_dicts,
            # times_cum_label_dicts
            )


def build_cumulative_times(time_list):
    # Current time t:
    t = 0
    times_cum = []
    for time in time_list:
        if np.isnan(time):
            # Overwrite.
            time = 0
        t += time
        times_cum.append(t)
    return times_cum


def build_formatted_time_str_lists_for_scenarios(
        times_dict
        ):
    keys = list(times_dict.keys())
    values = make_formatted_time_str_list(list(times_dict.values()))
    new_dict = dict(zip(keys, values))
    return new_dict


def make_formatted_time_str_list(times):
    new_times = []
    for t in times:
        try:
            t_new = (f'{int(60*(t/60)//60):2d}hr ' + f'{int(60*(t/60)%60):2d}min')
        except ValueError:
            # t is NaN.
            t_new = '~'
        new_times.append(t_new)
    return new_times


def draw_timeline(times_cum_dicts, times_cum_label_dicts):
    # Emoji unicode reference:
    # 🔧 \U0001f527
    # 🏥 \U0001f3e5
    # 🚑 \U0001f691
    # 💉 \U0001f489
    emoji_dict = {
        'Onset': '',
        'Call<br>ambulance': '\U0000260E',
        'Ambulance<br>arrives on scene': '\U0001f691',
        'Ambulance<br>leaves scene': '\U0001f691',
        'IVT unit<br>arrival': '\U0001f3e5',
        'IVT': '\U0001f489',
        'IVT unit<br>departure': '\U0001f691',
        'MT unit<br>arrival': '\U0001f3e5',
        'MT': '\U0001f527',
        'MSU<br>leaves base': '\U0001f691',
        'MSU<br>arrives on scene': '\U0001f691',
        'MSU<br>leaves scene': '\U0001f691',
        }

    emoji_offset = 0.0
    label_offset = 0.3

    fig = go.Figure()
    y_max = 0.0

    axis_labels = ['Drip & ship', 'Mothership', 'MSU']
    axis_values = np.array([0, 1, 2]) * 0.9
    for i, times_cum_dict in enumerate(times_cum_dicts):
        time_cum_list = list(times_cum_dict.values())
        # Convert from minutes to hours:
        time_cum_list = np.array(time_cum_list) / 60.0

        time_cum_str_list = list(times_cum_label_dicts[i].values())
        labels = list(times_cum_dict.keys())
        emoji_list = [emoji_dict[k] for k in list(times_cum_dict.keys())]

        # --- Plot ---
        # Make new labels list with line breaks removed
        # (for use in the hover label):
        labels_plain = [l.replace('<br>', ' ') for l in labels]
        # Draw straight line along the time axis:
        fig.add_trace(go.Scatter(
            y=time_cum_list,
            x=[axis_values[i]] * len(time_cum_list),
            mode='lines+markers',
            marker=dict(size=6, symbol='line-ew-open'),
            line=dict(color='grey'),    # OK in light and dark mode.
            showlegend=False,
            customdata=np.stack((time_cum_str_list, labels_plain), axis=-1)
        ))
        # "customdata" is not directly plotted in the line above,
        # but the values are made available for the hover label.

        # Update the hover text for the lines:
        fig.update_traces(
            hovertemplate=(
                '%{customdata[1]}'          # Name of this checkpoint
                '<br>' +                    # (line break)
                'Time: %{customdata[0]}' +  # Formatted time string
                '<extra></extra>'           # Remove secondary box.
                )
            )

        # # Add label for each scatter marker
        # for t, time_cum in enumerate(time_cum_list):
        #     # Only show it if it's moved on from the previous:
        #     if t == 0 or time_cum_list[t] > time_cum_list[t-1] or input_type == 'Simple':
        #         if write_under_list[t] is True:
        #             # Add formatted time string to the label.
        #             # (plus a line break, <br>)
        #             text = labels[t] + '<br>'+time_cum_str_list[t]
        #         else:
        #             text = labels[t]

        # Add emoji for each scatter marker
        for t, time in enumerate(time_cum_list):
            label = labels[t]
            time_label = time_cum_str_list[t]
            if labels[t] in ['IVT', 'MT']:
                colour = 'red'
                # label = f'<b>{label}'  # bold
                label += f': {time_label}'  # time
            else:
                colour = None
            # Write the label:
            fig.add_annotation(
                y=time,
                x=axis_values[i] + label_offset,
                text=label,
                showarrow=False,
                font=dict(
                    color=colour,
                    size=14),
                )
            fig.add_annotation(
                y=time,
                x=axis_values[i] + emoji_offset,
                text=emoji_list[t],
                showarrow=False,
                font=dict(
                    # color=time_colour_list[t],
                    size=24),
                # font=dict(color=time_colour_list[t])
                )

        # Update y-axis limit value if necessary:
        if np.max(time_cum_list) > y_max:
            y_max = np.max(time_cum_list)

    # # Set y range:
    fig.update_yaxes(range=[y_max * 1.05, 0 - y_max * 0.025])
    # Set y-axis label
    fig.update_yaxes(title_text='Time since onset (hours)')
    # Change y-axis title font size:
    fig.update_yaxes(title_font_size=10)

    # Set x range:
    fig.update_xaxes(range=[-0.1, 2.5])
    # Set x-axis labels
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=axis_values,
            ticktext=[f'<b>{x}' for x in axis_labels],   # <b> for bold
            side='top'  # Moves the labels to the top of the grid
        ),
    )

    # Remove y=0 and x=0 lines (zeroline) and grid lines:
    fig.update_xaxes(zeroline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, showgrid=False)


    fig_height = 200 * y_max
    fig.update_layout(
        # autosize=False,
        # width=500,
        height=fig_height
    )

    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Disable zoom and pan:
    fig.update_layout(xaxis=dict(fixedrange=True),
                      yaxis=dict(fixedrange=True))

    # Turn off legend click events
    # (default is click on legend item, remove that item from the plot)
    fig.update_layout(legend_itemclick=False)

    # Options for the mode bar.
    # (which doesn't appear on touch devices.)
    plotly_config = {
        # Mode bar always visible:
        # 'displayModeBar': True,
        # Plotly logo in the mode bar:
        'displaylogo': False,
        # Remove the following from the mode bar:
        'modeBarButtonsToRemove': [
            'zoom', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }

    # Write to streamlit:
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def build_time_dicts_muster(pathway_dict):
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


def get_timeline_display_dict():

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
    return timeline_display_dict


def plot_timeline(
        time_dicts,
        timeline_display_dict,
        y_vals,
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