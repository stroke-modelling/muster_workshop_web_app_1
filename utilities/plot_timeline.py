"""
Plots a timeline for the patient pathway for cases 1 and 2.
Includes labelled points and emoji.

Initially copied over from the stroke outcome app.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# from outcome_utilities.fixed_params import emoji_text_dict, plotly_colours


def build_data_for_timeline(params_dict):
    times_dicts = (
        build_time_dicts_for_scenarios(params_dict))
    times_cum_dicts = build_cumulative_time_dicts_for_scenarios(
        *times_dicts
        )
    times_cum_label_dicts = build_formatted_time_str_lists_for_scenarios(
        times_cum_dicts
        )
    return (times_dicts,
            times_cum_dicts,
            times_cum_label_dicts
            )


def build_time_dicts_for_scenarios(params_dict):
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
    # Onset time:
    time_dict = {'onset': 0}

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

    return times_dict_drip_ship, times_dict_mothership, times_dict_msu


def build_cumulative_time_dicts_for_scenarios(
        times_dict_drip_ship,
        times_dict_mothership,
        times_dict_msu
        ):
    times_cum_drip_ship = build_cumulative_times(list(times_dict_drip_ship.values()))
    times_cum_mothership = build_cumulative_times(list(times_dict_mothership.values()))
    times_cum_msu = build_cumulative_times(list(times_dict_msu.values()))

    keys_cum_drip_ship = [
        'Onset',
        'Call<br>ambulance',
        'Ambulance<br>arrives on scene',
        'Ambulance<br>leaves scene',
        'IVT unit<br>arrival',
        'IVT',
        'IVT unit<br>departure',
        'MT unit<br>arrival',
        'MT',
    ]
    keys_cum_mothership = [
        'Onset',
        'Call<br>ambulance',
        'Ambulance<br>arrives on scene',
        'Ambulance<br>leaves scene',
        'MT unit<br>arrival',
        'IVT',
        'MT',
    ]
    keys_cum_msu = [
        'Onset',
        'Call<br>ambulance',
        'MSU<br>leaves base',
        'MSU<br>arrives on scene',
        'IVT',
        'MSU<br>leaves scene',
        'MT unit<br>arrival',
        'MT',
    ]

    # Convert to dict:
    times_cum_dict_drip_ship = dict(
        zip(keys_cum_drip_ship, times_cum_drip_ship))
    times_cum_dict_mothership = dict(
        zip(keys_cum_mothership, times_cum_mothership))
    times_cum_dict_msu = dict(
        zip(keys_cum_msu, times_cum_msu))

    # Special case in drip-and-ship -
    # the time delay for transfer is counted from arrival at unit, not
    # from time to IVT which is the result of build_cumulative_times().
    times_cum_dict_drip_ship['IVT unit<br>departure'] = (
            times_cum_dict_drip_ship['IVT unit<br>arrival'] +
            times_dict_drip_ship['transfer_time_delay']
        )
    times_cum_dict_drip_ship['MT unit<br>arrival'] = (
            times_cum_dict_drip_ship['IVT unit<br>departure'] +
            times_dict_drip_ship['transfer_time']
        )
    times_cum_dict_drip_ship['MT'] = (
            times_cum_dict_drip_ship['MT unit<br>arrival'] +
            times_dict_drip_ship['process_time_transfer_arrival_to_puncture']
        )

    # Special case in mothership -
    # the time to MT is counted from arrival at unit, not
    # from time to IVT which is the result of build_cumulative_times().
    times_cum_dict_mothership['MT'] = (
        times_cum_dict_mothership['MT unit<br>arrival'] +
        times_dict_mothership['process_time_arrival_to_puncture']
    )

    return (times_cum_dict_drip_ship,
            times_cum_dict_mothership,
            times_cum_dict_msu)


def build_cumulative_times(time_list):
    # Current time t:
    t = 0
    times_cum = []
    for time in time_list:
        t += time
        times_cum.append(t)
    return times_cum


def build_formatted_time_str_lists_for_scenarios(
        times_dicts
        ):
    new_dicts = []
    for t, times_dict in enumerate(times_dicts):
        keys = list(times_dict.keys())
        values = make_formatted_time_str_list(list(times_dict.values()))
        new_dict = dict(zip(keys, values))
        new_dicts.append(new_dict)
    return tuple(new_dicts)


def make_formatted_time_str_list(times):
    return [(f'{int(60*(t/60)//60):2d}hr ' + f'{int(60*(t/60)%60):2d}min')
            for t in times]


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
    label_offset = 0.2

    fig = go.Figure()

    axis_labels = ['Drip & ship', 'Mothership', 'MSU']
    axis_values = [2, 1, 0]
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
            x=time_cum_list,
            y=[axis_values[i]] * len(time_cum_list),
            mode='lines+markers',
            marker=dict(size=6, symbol='line-ns-open'),
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
                label += f'<br>{time_label}'  # time
            else:
                colour = None
            # Write the label:
            fig.add_annotation(
                x=time,
                y=axis_values[i] + label_offset,
                text=label,
                showarrow=False,
                font=dict(
                    color=colour,
                    size=10),
                )
            fig.add_annotation(
                x=time,
                y=axis_values[i] + emoji_offset,
                text=emoji_list[t],
                showarrow=False,
                font=dict(
                    # color=time_colour_list[t],
                    size=24),
                # font=dict(color=time_colour_list[t])
                )

        # # Update y-axis limit value if necessary:
        # if time_cumulative > y_max:
        #     y_max = time_cumulative

    # # Set y range:
    # fig.update_yaxes(range=[y_max * 1.05, 0 - y_max * 0.025])
    # Set y-axis label
    fig.update_xaxes(title_text='Time since onset (hours)')
    # Change y-axis title font size:
    fig.update_xaxes(title_font_size=10)

    # Set x range:
    fig.update_yaxes(range=[-0.1, 2.5])
    # Set x-axis labels
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=axis_values,
            ticktext=[f'<b>{x}' for x in axis_labels],   # <b> for bold
            # side='top'  # Moves the labels to the top of the grid
        ),
    )

    # Remove y=0 and x=0 lines (zeroline) and grid lines:
    fig.update_xaxes(zeroline=False, showgrid=False)
    fig.update_yaxes(zeroline=False, showgrid=False)


    # Reduce size of figure by adjusting margins:
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)

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