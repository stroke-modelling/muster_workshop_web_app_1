"""
Plots a timeline for the patient pathway for cases 1 and 2.
Includes labelled points and emoji.

Initially copied over from the stroke outcome app.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utilities.utils import update_plotly_font_sizes, update_plotly_font_sizes


def make_formatted_time_str(t):
    try:
        t_new = (f'{int(60*(t/60)//60):d}hr ' +
                 f'{int(60*(t/60)%60):02d}min')
    except ValueError:
        # t is NaN.
        t_new = '~'
    return t_new


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


def make_treatment_time_df_msu(treatment_times_without_travel):
    # Add strings to show travel times:
    usual_care_time_to_ivt_str = ' '.join([
        f'{treatment_times_without_travel["usual_care_time_to_ivt"]}',
        '+ 🚑 travel to nearest unit'
        ])
    usual_care_mt_no_transfer_str = ' '.join([
        f'{treatment_times_without_travel["usual_care_mt_no_transfer"]}',
        '+ 🚑 travel to nearest unit'
        ])
    usual_care_mt_transfer_str = ' '.join([
        f'{treatment_times_without_travel["usual_care_mt_transfer"]}',
        '+ 🚑 travel to nearest unit',
        '+ 🚑 travel between units'
        ])
    msu_time_to_ivt_str = ' '.join([
        f'{treatment_times_without_travel["msu_time_to_ivt"]}',
        '+ 🚑 travel from MSU base'
        ])
    msu_time_to_mt_str = ' '.join([
        f'{treatment_times_without_travel["msu_time_to_mt"]}',
        '+ 🚑 travel from MSU base',
        '+ 🚑 travel to MT unit'
        ])
    msu_time_to_mt_no_ivt_str = ' '.join([
        f'{treatment_times_without_travel["msu_time_to_mt_no_ivt"]}',
        '+ 🚑 travel from MSU base',
        '+ 🚑 travel to MT unit'
        ])

    # Place these into a dataframe:
    df_treatment_times = pd.DataFrame(
        [[usual_care_time_to_ivt_str, msu_time_to_ivt_str],
         [usual_care_mt_no_transfer_str, msu_time_to_mt_no_ivt_str],
         [usual_care_mt_transfer_str, msu_time_to_mt_str]],
        columns=['Standard pathway', 'Mobile Stroke Unit'],
        index=['Time to IVT', 'Time to MT (fastest)', 'Time to MT (slowest)']
    )
    return df_treatment_times


def make_treatment_time_df_optimist(treatment_times_without_travel):
    t_near = '\n\+ 🚑 travel to nearest unit'
    t_mt = '\n\+ 🚑 travel to MT unit'
    t_trans = '\+ 🚑 travel between units'

    t_dict = {
        'usual_care_time_to_ivt': {
            'tre': 'usual_care_time_to_ivt',
            'trav': [t_near],
        },
        'usual_care_mt_no_transfer': {
            'tre': 'usual_care_time_to_mt_no_transfer',
            'trav': [t_near],
        },
        'usual_care_mt_transfer': {
            'tre': 'usual_care_time_to_mt_transfer',
            'trav': [t_near, t_trans],
        },
        'prehospdiag_rej_time_to_ivt': {
            'tre': 'prehospdiag_time_to_ivt',
            'trav': [t_near],
        },
        'prehospdiag_rej_mt_no_transfer': {
            'tre': 'prehospdiag_time_to_mt_no_transfer',
            'trav': [t_near],
        },
        'prehospdiag_rej_mt_transfer': {
            'tre': 'prehospdiag_time_to_mt_transfer',
            'trav': [t_near, t_trans],
        },
        'prehospdiag_app_time_to_ivt': {
            'tre': 'prehospdiag_time_to_ivt',
            'trav': [t_mt],
        },
        'prehospdiag_app_mt_no_transfer': {
            'tre': 'prehospdiag_time_to_mt_no_transfer',
            'trav': [t_mt],
        },
        'prehospdiag_app_mt_transfer': {
            'tre': 'prehospdiag_time_to_mt_no_transfer',
            'trav': [t_mt],
        },
    }
    s_dict = {}
    for key, t_dict in t_dict.items():
        t = treatment_times_without_travel[t_dict['tre']]
        s = make_formatted_time_str(t)
        s = f"__{s.replace(' ', '__ __')}__"
        s_dict[key] = '  '.join([s] + t_dict['trav'])

    # Manually update the last one:
    s_dict['prehospdiag_app_mt_transfer'] = '\-'

    # Place these into a dataframe:
    r1 = [s_dict['usual_care_time_to_ivt'],
          s_dict['prehospdiag_rej_time_to_ivt'],
          s_dict['prehospdiag_app_time_to_ivt']]
    r2 = [s_dict['usual_care_mt_no_transfer'],
          s_dict['prehospdiag_rej_mt_no_transfer'],
          s_dict['prehospdiag_app_mt_no_transfer']]
    r3 = [s_dict['usual_care_mt_transfer'],
          s_dict['prehospdiag_rej_mt_transfer'],
          s_dict['prehospdiag_app_mt_transfer']]
    cols = ['Usual care', 'Redirection rejected', 'Redirection approved']
    ind = ['Time to IVT 💉', 'Time to MT 🔧  \n(fast)', 'Time to MT 🔧  \n(slow)']
    df_treatment_times = pd.DataFrame([r1, r2, r3], columns=cols, index=ind)
    return df_treatment_times


def build_time_dicts_for_plot_msu(time_dicts, gap_between_chunks=45):
    # Setup for timeline plot.
    # Leave this gap in minutes between separate chunks of pathway:
    # Start each chunk at these offsets:
    time_offsets = {
        'prehosp_usual_care': 0,
        'ivt_only_unit': (
            gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values())
            ),
        'mt_transfer_unit': (
            gap_between_chunks * 2.0 +
            sum(time_dicts['prehosp_usual_care'].values()) +
            sum(time_dicts['ivt_only_unit'].values())
        ),
        'ivt_mt_unit': (
            gap_between_chunks + sum(time_dicts['prehosp_usual_care'].values())
        ),
        'msu_dispatch': 0,
        'prehosp_msu_ivt': (
            gap_between_chunks + sum(time_dicts['msu_dispatch'].values())
        ),
        'prehosp_msu_no_ivt': (
            gap_between_chunks + sum(time_dicts['msu_dispatch'].values())
        ),
        'mt_transfer_from_msu': (
            gap_between_chunks * 2.0 +
            sum(time_dicts['msu_dispatch'].values()) +
            max([
                sum(time_dicts['prehosp_msu_ivt'].values()),
                sum(time_dicts['prehosp_msu_no_ivt'].values())
                ])
        ),
    }
    # Find shared max time for setting same size across multiple plots
    # so that 1 minute always spans the same number of pixels.
    tmax = max(
        [time_offsets[k] + sum(time_dicts[k].values()) for k in time_dicts.keys()]
    ) + gap_between_chunks
    return time_offsets, tmax


def load_timeline_setup(use_col):
    # Columns of emoji, label, and sub-timeline order:
    df_time = pd.read_csv('./data/timeline_display.csv', index_col='name')
    # Limit to the keys we need:
    df_time = df_time[df_time[use_col] == 1]
    df_time = df_time.drop(['optimist', 'muster'], axis='columns')
    return df_time


def load_timeline_chunks(use_col):
    # Columns of emoji, label, and sub-timeline order:
    df_chunks = pd.read_csv('./data/timeline_chunks.csv', index_col='scenario')
    # Limit to the keys we need:
    df_chunks = df_chunks[df_chunks[use_col] == 1]
    df_chunks = df_chunks.drop(['optimist', 'muster'], axis='columns')
    return df_chunks


def set_up_fig_chunks(df_times, t_travel=20, gap=20):
    # Set up chunk widths and spacing:
    cols_prehosp = [c for c in df_times if c.startswith('times_prehospital')]
    cols_first = [c for c in df_times if
                  (c.startswith('times_ivt') | c.startswith('times_mt_unit'))]
    cols_second = ['times_mt_unit_transfer']
    t_max_prehosp = df_times[cols_prehosp].max().max()
    t_max_first = df_times[cols_first].max().max()
    t_max_second = df_times[cols_second].max().max()

    # Set up figure-wide chunk locations:
    tups = (
        ('prehosp', 'Pre-hospital'),
        ('travel_to_first', 'Travel to<br>first unit'),
        ('first_unit', 'First unit'),
        ('travel_transfer', 'Transfer to<br>second unit'),
        ('second_unit', 'Transfer unit'),
        ('treat_times', 'Treatment times<br>without travel'),
    )
    df_chunk_coords = (
        pd.DataFrame(tups, columns=['chunk', 'label']).set_index('chunk'))
    # Size of each chunk:
    df_chunk_coords.loc['prehosp', 'width'] = t_max_prehosp
    df_chunk_coords.loc['travel_to_first', 'width'] = t_travel
    df_chunk_coords.loc['first_unit', 'width'] = t_max_first
    df_chunk_coords.loc['travel_transfer', 'width'] = t_travel
    df_chunk_coords.loc['second_unit', 'width'] = t_max_second
    df_chunk_coords.loc['treat_times', 'width'] = t_travel * 2.0

    # Start time of each chunk:
    chunk_order = df_chunk_coords.index
    for i, chunk in enumerate(chunk_order):
        if i == 0:
            df_chunk_coords.loc[chunk, 'offset'] = 0
        else:
            last_chunk = chunk_order[i-1]
            df_chunk_coords.loc[chunk, 'offset'] = (
                df_chunk_coords.loc[last_chunk, 'offset'] +
                df_chunk_coords.loc[last_chunk, 'width'] +
                gap
            )

    # Spans of background grouping rectangles:
    df_chunk_coords['min'] = df_chunk_coords['offset'].copy()
    df_chunk_coords['max'] = (
        df_chunk_coords['min'] + df_chunk_coords['width']).copy()
    # Alter spans for breathing room:
    df_chunk_coords['min'] -= gap*0.25
    df_chunk_coords['max'] += gap*0.25

    return df_chunk_coords


def draw_timeline(df_pathway_steps, df_treats):
    """Optimist."""
    # Load emoji and labels:
    df_times = load_timeline_setup('optimist')
    df_times = pd.merge(df_times, df_pathway_steps,
                        left_index=True, right_index=True, how='left')

    # Calculate cumulative times for each chunk:
    order_cols = [c for c in df_times.columns if
                  (c.startswith('order_') & ('msu' not in c))]
    for col in order_cols:
        df = df_times[[col, 'value']]
        df = df[df[col].notna()].sort_values(col)
        new_col = col.replace('order_', 'times_')
        df[new_col] = np.cumsum(df['value'])
        df_times = pd.concat((df_times, df[new_col]), axis='columns')

    # Chunk setup for each scenario.
    # Use these to work out which chunk is used in each scenario,
    # e.g. whether first unit is the IVT unit or the MT unit:
    df_chunks = load_timeline_chunks('optimist')
    # Use these to make coordinates for plotting:
    df_chunk_coords = set_up_fig_chunks(df_times, t_travel=20, gap=20)

    # Use this to pick out the right times for each scenario:
    def get_times(df_times, col):
        """Pick out valid times for a chunk/scenario combo."""
        mask = df_times[col].notna()
        df = df_times.loc[mask].sort_values(col)
        df = df[[col, 'emoji', 'label_timeline', 'value']]
        df = df.rename(columns={col: 'cumulative_time'})
        return df

    # ----- Plot setup -----
    # Hover labels for pathway data:
    hovertemplate = (
        '%{customdata[1]}'          # Name of this checkpoint
        '<br>' +                    # (line break)
        'Time since last step: %{customdata[0]}min' +  # Formatted time string
        '<br>' +
        'Time in this section: %{customdata[2]}min' +
        '<extra></extra>'           # Remove secondary box.
        )
    # Hover labels for travel time bits
    # (because incomplete time info here):
    hovertemplate_travel = (
        '%{customdata[1]}'          # Name of this checkpoint
        '<br>' +                    # (line break)
        'Time since last step: %{customdata[0]}' +  # Formatted time string
        '<extra></extra>'           # Remove secondary box.
        )

    def plot_chunk_emoji(df, offset, hovertemplate):
        fig.add_trace(go.Scatter(
            x=df['cumulative_time'] + offset,
            y=[axis_ticklabel] * len(df),
            text=df['emoji'],
            mode='text+lines',
            showlegend=False,
            line=dict(color='grey', width=3),
            textfont=dict(size=24),
            customdata=np.stack((
                df['value'],
                df['label_timeline'],
                df['cumulative_time'],
            ), axis=-1),
            hovertemplate=hovertemplate,
        ))

    def draw_arrow(x_list, arrow_colour):
        fig.add_trace(go.Scatter(
            x=x_list,
            y=[axis_ticklabel] * len(x_list),
            **arrow_kwargs,
            line=dict(color=arrow_colour, width=10),
        ))

    def draw_unit(x, marker_unit_kwargs, unit_time, unit_label,):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[axis_ticklabel],
            mode='markers',
            showlegend=False,
            marker=marker_unit_kwargs,
            customdata=np.stack((
                [unit_time],
                [unit_label],
            ), axis=-1),
            hovertemplate=hovertemplate_travel,
        ))

    def draw_start_location(x):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[axis_ticklabel],
            mode='text',
            text=['🏠'],
            showlegend=False,
            textfont=dict(size=24),
            customdata=np.stack((
                ['0min'],
                ['Leave starting location'],
            ), axis=-1),
            hovertemplate=hovertemplate_travel,
        ))

    arrow_kwargs = dict(
        mode='lines+markers',
        marker=dict(size=20, symbol='arrow-up', angleref='previous',
                    standoff=10),
        showlegend=False,
        hoverinfo='skip',
    )
    marker_ivt_unit_kwargs = dict(size=15, symbol='circle', color='white',
                                  line={'color': 'black', 'width': 1},)
    marker_mt_unit_kwargs = dict(size=18, symbol='star', color='white',
                                 line={'color': 'black', 'width': 1},)

    # ----- Begin plot -----
    fig = go.Figure()

    # Draw background rectangle for each chunk and label them.
    for chunk in df_chunk_coords.index:
        s = df_chunk_coords.loc[chunk]
        if 'treat' in chunk:
            pass
        else:
            fig.add_vrect(
                x0=s['min'], x1=s['max'],
                line_width=0, fillcolor='silver', opacity=0.2
                )
        fig.add_annotation(
            y=1.0,
            x=0.5 * (s['min'] + s['max']),
            text=s['label'],
            showarrow=False,
            yref='paper',
            yanchor='bottom'
        )

    # Draw the scatter data.
    for i, scenario in enumerate(df_chunks.index):
        s = df_chunks.loc[scenario]
        # Pick out info for each chunk:
        axis_ticklabel = s['label']
        col_prehosp = f'times_prehospital_{s["prehosp"]}'
        col_first = f'times_{s["first_unit"]}'
        col_second = f'times_{s["second_unit"]}'

        df_prehosp = get_times(df_times, col_prehosp)
        df_first_unit = get_times(df_times, col_first)
        if isinstance(s['second_unit'], str):
            use_second_unit = True
            df_second_unit = get_times(df_times, col_second)
        else:
            use_second_unit = False

        # Pre-hospital chunk:
        plot_chunk_emoji(
            df_prehosp,
            df_chunk_coords.loc['prehosp', 'offset'],
            hovertemplate
            )
        plot_chunk_emoji(
            df_first_unit,
            df_chunk_coords.loc['first_unit', 'offset'],
            hovertemplate
            )
        if use_second_unit:
            plot_chunk_emoji(
                df_second_unit,
                df_chunk_coords.loc['second_unit', 'offset'],
                hovertemplate
                )
        # Draw travel times:
        x_travel_first = [
            df_chunk_coords.loc['travel_to_first', 'offset'],
            df_chunk_coords.loc['travel_to_first', ['offset', 'width']].sum()
            ]
        # Is the first unit the MT unit?
        if use_second_unit:
            # The first unit is the IVT-only unit.
            marker_unit_kwargs = marker_ivt_unit_kwargs
            arrow_colour = 'grey'
            unit_label = 'Arrive at IVT-only unit'
        else:
            marker_unit_kwargs = marker_mt_unit_kwargs
            arrow_colour = '#ff4b4b'
            unit_label = 'Arrive at MT unit'
        # Connecting arrow:
        draw_arrow(x_travel_first, arrow_colour)

        # Start location:
        draw_start_location(x_travel_first[0])
        # First unit:
        draw_unit(x_travel_first[1], marker_unit_kwargs, 'Depends',
                  unit_label)

        if use_second_unit:
            x_travel_transfer = [
                df_chunk_coords.loc['travel_transfer', 'offset'],
                df_chunk_coords.loc['travel_transfer',
                                    ['offset', 'width']].sum()
                ]
            # Same for transfer unit.
            # Connecting arrow:
            draw_arrow(x_travel_transfer, arrow_colour)
            # First unit:
            draw_unit(x_travel_transfer[0], marker_ivt_unit_kwargs,
                      '0min', 'Leave IVT-only unit')
            # MT unit:
            draw_unit(x_travel_transfer[1], marker_mt_unit_kwargs,
                      'Depends', 'Arrive at MT-only unit')

        # Treatment times:
        m = df_treats['scenario'] == scenario
        m_ivt = m & (df_treats['treatment'] == 'ivt')
        m_mt = m & (df_treats['treatment'] == 'mt')
        time_ivt = df_treats.loc[m_ivt, 'hr_min'].values[0]
        time_mt = df_treats.loc[m_mt, 'hr_min'].values[0]
        s = f'💉 IVT: {time_ivt}<br>🔧 MT: {time_mt}'
        x = np.mean([df_chunk_coords.loc['treat_times', 'min'],
                     df_chunk_coords.loc['treat_times', 'max']])
        fig.add_annotation(
            y=axis_ticklabel,
            x=x,
            text=s,
            showarrow=False,
        )

    # Axis ranges:
    fig.update_layout(xaxis_range=[
        df_chunk_coords['min'].min() - 2,
        df_chunk_coords['max'].max() + 2
        ])
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[]))
    fig.update_yaxes(autorange="reversed")  # Flip veritcally

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='Treatment pathways')
    fig.update_layout(width=2000, height=400, margin_b=0, margin_t=75)
    fig.update_layout(dragmode=False)
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
            'pan',
            'select',
            'zoomIn',
            'zoomOut',
            'autoScale',
            'lasso2d'
            ],
        # Options when the image is saved:
        'toImageButtonOptions': {'height': None, 'width': None},
        }
    st.plotly_chart(fig, config=plotly_config, width='content')
