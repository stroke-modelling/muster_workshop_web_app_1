"""
Geography.
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np
import os

from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stroke_maps.load_data

from classes.geography_processing import Geoprocessing
from utilities.utils import print_progress_loc, update_plotly_font_sizes


# ----- Functions -----
def import_stroke_unit_services(
        use_msu=True,
        keep_only_ivt_mt=False,
        keep_only_england=True
        ):
    """
    """
    # Set up stroke unit services (IVT, MT, MSU).
    df_unit_services = stroke_maps.load_data.stroke_unit_region_lookup()

    # Rename columns to match what the rest of the model here wants.
    df_unit_services.index.name = 'Postcode'
    df_unit_services = df_unit_services.rename(columns={
        'use_ivt': 'Use_IVT',
        'use_mt': 'Use_MT',
        'use_msu': 'Use_MSU',
    })

    if keep_only_ivt_mt:
        # Remove stroke units that don't offer IVT or MT:
        mask = (
            (df_unit_services['Use_IVT'] == 1) |
            (df_unit_services['Use_MT'] == 1)
        )
        df_unit_services = df_unit_services.loc[mask].copy()
    else:
        pass

    if keep_only_england:
        # Limit to England:
        mask = df_unit_services['country'] == 'England'
        df_unit_services = df_unit_services.loc[mask].copy()
        # Remove Wales:
        df_unit_services = df_unit_services.loc[
            df_unit_services['region_type'] != 'LHB'].copy()
    else:
        pass

    # Limit the units list to only units in the travel time matrix:
    df_travel = pd.read_csv(
        './data/inter_hospital_time_calibrated.csv',
        index_col='from_postcode'
        )
    units_allowed = df_travel.index.values
    mask_allowed = df_unit_services.index.isin(units_allowed)
    df_unit_services = df_unit_services[mask_allowed].copy()

    # Limit which columns to show:
    cols_to_keep = ['ssnap_name', 'Use_IVT', 'Use_MT', 'isdn']
    df_unit_services = df_unit_services[cols_to_keep]

    if use_msu:
        df_unit_services['Use_MSU'] = df_unit_services['Use_MT'].copy()

    # Sort by ISDN name for nicer display:
    df_unit_services = df_unit_services.sort_values(['isdn', 'ssnap_name'])

    return df_unit_services


def load_lsoa_region_lookups():
    # Load region info for each LSOA:
    # Relative import from package files:
    df_lsoa_regions = stroke_maps.load_data.lsoa_region_lookup()
    df_lsoa_regions = df_lsoa_regions.reset_index()

    # Load further region data linking SICBL to other regions:
    df_regions = stroke_maps.load_data.region_lookup()
    df_regions = df_regions.reset_index()
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_regions,
        left_on='region_code', right_on='region_code', how='left'
        )

    # Load ambulance service data:
    df_lsoa_ambo = stroke_maps.load_data.ambulance_lsoa_lookup()
    df_lsoa_ambo = df_lsoa_ambo.reset_index()
    # Merge in:
    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_ambo[['LSOA11NM', 'ambo22']],
        left_on='lsoa', right_on='LSOA11NM', how='left'
        ).drop('LSOA11NM', axis='columns')
    return df_lsoa_regions


def select_unit_services():
    """
    """
    # Load stroke unit data from file:
    df = import_stroke_unit_services(
        use_msu=False,
        keep_only_ivt_mt=False,
        keep_only_england=True
        )

    # Display this as an editable dataframe:
    df_unit_services = st.data_editor(
        df,
        disabled=['postcode', 'ssnap_name', 'isdn'],
        # height=180  # limit height to show fewer rows
        # Make columns display as checkboxes instead of 0/1 ints:
        column_config={
            'Use_IVT': st.column_config.CheckboxColumn(),
            'Use_MT': st.column_config.CheckboxColumn(),
        },
        )
    return df_unit_services


@st.cache_data
def find_nearest_units_each_lsoa(df_unit_services, _log=True, _log_loc=None):
    """

    Result
    ------
    df_geo - pd.Dataframe. Columns 'LSOA', 'nearest_ivt_unit',
             'nearest_ivt_time', 'nearest_mt_unit', 'nearest_mt_time',
             'transfer_unit', 'transfer_required', 'transfer_time',
             'nearest_msu_unit', 'nearest_msu_time', 'Admissions',
             'nearest_ivt_then_mt_time'
    """
    try:
        geo = st.session_state['geo']
    except KeyError:
        # Process and save geographic data
        # (only needed when hospital data changes)
        geo = Geoprocessing(
            limit_to_england=True
            )
    # Update units:
    geo.df_unit_services = df_unit_services
    geo.update_unit_services()
    # Rerun geography:
    geo.run()
    # Reset index because Model expects a column named 'lsoa':
    df_geo = geo.get_combined_data().copy(deep=True).reset_index()
    # Round travel times to nearest minute.
    # +1e-5 to make all 0.5 times round up to next minute.
    cols_times = ['nearest_ivt_time', 'nearest_mt_time', 'transfer_time',
                  'nearest_msu_time']
    df_geo[cols_times] = np.round(df_geo[cols_times] + 1e-5, 0)
    # Separate column for separate travel time including transfer:
    df_geo['nearest_ivt_then_mt_time'] = (
        df_geo['nearest_ivt_time'] + df_geo['transfer_time'])

    # Cache the geo class so that on the next run all of the big
    # data files are not loaded in another time.
    st.session_state['geo'] = geo

    if _log:
        p = 'Assigned LSOA to nearest units.'
        print_progress_loc(p, _log_loc)
    return df_geo


@st.cache_data
def find_unique_travel_times(
        df_times,
        cols_ivt=['nearest_ivt_time', 'nearest_mt_time'],
        cols_mt=['nearest_ivt_then_mt_time', 'nearest_mt_time'],
        cols_pairs={
            'transfer': ('nearest_ivt_time', 'nearest_ivt_then_mt_time'),
            'no_transfer': ('nearest_mt_time', 'nearest_mt_time')
        },
        _log=True,
        _log_loc=None
        ):
    """

    """
    # IVT can either be at the nearest unit or at the MT unit if
    # redirected. MT is always at the MT unit, either travelling
    # there directly or going via the IVT-only unit.
    times_to_ivt = sorted(list(set(df_times[cols_ivt].values.flatten())))
    times_to_mt = sorted(list(set(df_times[cols_mt].values.flatten())))

    # Find all pairs of times.
    # Combinations are: IVT at nearest unit, then MT after transfer;
    #                   IVT and MT at nearest MT unit.
    all_pairs = {}
    for label, pair in cols_pairs.items():
        pairs_here = df_times[list(pair)].drop_duplicates()
        # Don't use rename dictionary because can have duplicate
        # column names in pair.
        pairs_here.columns = ['travel_for_ivt', 'travel_for_mt']
        all_pairs[label] = pairs_here

    if _log:
        p = 'Gathered unique travel times.'
        print_progress_loc(p, _log_loc)
    return times_to_ivt, times_to_mt, all_pairs


@st.cache_data
def find_region_admissions_by_unique_travel_times(
        df_lsoa_units_times, keep_only_england=True, unique_travel=True,
        _log=True, _log_loc=None
        ):
    """
    df_lsoa_units_times includes admissions data.

    Multiple layers in this dictionary.
    + dict_region_unique_times
      + all_patients
      + nearest_unit_no_mt
        + usual_care_ivt
        + usual_care_mt
        + redirection_ivt
        + redirection_mt
    """
    # Load in LSOA-region lookup:
    df_lsoa_regions = load_lsoa_region_lookups()
    # Columns: 'lsoa', 'lsoa_code', 'region', 'region_code',
    # 'region_type', 'short_code', 'country', 'icb', 'icb_code',
    # 'isdn', 'ambo22'.
    if keep_only_england:
        mask_eng = df_lsoa_regions['region_type'] == 'SICBL'
        df_lsoa_regions = df_lsoa_regions.loc[mask_eng].copy()

    # Merge in admissions and timings data:
    cols_to_merge = ['LSOA', 'Admissions', 'transfer_required',
                     'nearest_ivt_unit']
    if unique_travel:
        cols_to_merge += ['nearest_ivt_time', 'nearest_mt_time',
                          'nearest_ivt_then_mt_time']
    else:
        scens = ['usual_care', 'redirection_approved',
                 'redirection_rejected']
        treats = ['ivt', 'mt']
        cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
        cols_to_merge += cols_treat_scen

    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_units_times[cols_to_merge],
        left_on='lsoa', right_on='LSOA', how='right'
        )
    # Calculate this separately for each region type.
    region_types = ['national', 'icb', 'isdn', 'ambo22', 'nearest_ivt_unit']  # 'region
    dict_region_unique_times = {}
    masks = {'all_patients': slice(None),
             'nearest_unit_no_mt': df_lsoa_regions['transfer_required']}

    if unique_travel:
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        time_cols_dict = {
            'usual_care_ivt': ['nearest_ivt_time'],
            'usual_care_mt': ['nearest_ivt_then_mt_time'],
            'redirection_ivt': ['nearest_mt_time'],
            'redirection_mt': ['nearest_mt_time']
            }
        for mask_label, mask in masks.items():
            dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            for region_type in region_types:
                dict_region_unique_times[mask_label][region_type] = {}
                for time_label, time_cols in time_cols_dict.items():
                    if region_type == 'national':
                        cols = ['Admissions'] + time_cols
                        df = df_here[cols].groupby(time_cols).sum()
                        df = df.rename(columns={'Admissions': 'National'})
                    else:
                        cols = [region_type, 'Admissions'] + time_cols
                        df = df_here[cols].groupby(
                            [region_type, *time_cols]).sum()
                        # df has columns for region, time, and admissions.
                        # Change to index of time, one column per region,
                        # values of admissions:
                        df = (df.unstack(time_cols).transpose()
                              .reset_index().set_index(time_cols)
                              .drop('level_0', axis='columns')
                              )
                    dict_region_unique_times[
                        mask_label][region_type][time_label] = df

        if _log:
            p = 'Found total admissions with each unique travel time per region.'
            print_progress_loc(p, _log_loc)
    else:
        # Unique treatment time combinations.
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        for mask_label, mask in masks.items():
            dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            for region_type in region_types:
                if region_type == 'national':
                    cols = ['Admissions'] + cols_treat_scen
                    df = df_here[cols].groupby(cols_treat_scen).sum()
                    df = df.rename(columns={'Admissions': 'National'})
                else:
                    cols = [region_type, 'Admissions'] + cols_treat_scen
                    df = df_here[cols].groupby(
                        [region_type, *cols_treat_scen]).sum()
                    # df has columns for region, time, and admissions.
                    # Change to index of time, one column per region,
                    # values of admissions:
                    df = (df.unstack(cols_treat_scen).transpose()
                          .reset_index().set_index(cols_treat_scen)
                          .drop('level_0', axis='columns')
                          )
                dict_region_unique_times[
                    mask_label][region_type] = df

        if _log:
            p = 'Found total admissions with each set of unique treatment times per region.'
            print_progress_loc(p, _log_loc)
    return dict_region_unique_times


def calculate_region_outcomes(
        df_regions,
        df_outcomes,
        _log=True,
        _log_loc=None
        ):
    """
    df_regions contains admission numbers for unique treatment times.
    """
    df_outcomes = df_outcomes.copy()
    # Convert mRS distributions to non-cumulative:
    cols_mrs = [f'mrs_dists_{i}' for i in range(7)]
    cols_mrs_noncum = [c.replace('dists_', 'dists_noncum_')
                       for c in cols_mrs]
    df_outcomes[cols_mrs_noncum] = (
        np.diff(df_outcomes[cols_mrs], axis=1, prepend=0.0))

    df_in = pd.concat((df_regions, df_outcomes), axis='columns')

    cols_out = list(df_outcomes.columns)
    cols_std = [f'{c}_std' for c in cols_out]
    df_out = pd.DataFrame(columns=cols_out+cols_std)
    for region in df_regions.columns:
        mask = df_in[region].notna()
        # Take admissions-weighted average of outcomes.
        vals = df_in.loc[mask, cols_out]
        weights = df_in.loc[mask, region]
        # Create stats from these data:
        weighted_stats = DescrStatsW(vals, weights=weights, ddof=0)
        # Means (one value per outcome, mRS band):
        means = weighted_stats.mean
        # Standard deviations (one value per outcome, mRS band):
        stds = weighted_stats.std
        # Round these values:
        means = np.round(means, 3)
        stds = np.round(stds, 3)
        # Store result:
        s = pd.Series(list(means) + list(stds), index=cols_out+cols_std)
        df_out.loc[region] = s

    if _log:
        p = 'Found average outcomes for each region.'
        print_progress_loc(p, _log_loc)
    return df_out


def select_highlighted_regions(df_unit_services):
    # Select a region based on what's actually in the data,
    # not by guessing in advance which IVT units are included for example.
    region_options_dict = load_region_lists(df_unit_services)
    bar_options = ['National']
    for key, region_list in region_options_dict.items():
        bar_options += [f'{key}: {v}' for v in region_list]

    highlighted_options = st.multiselect('Regions to highlight',
                                         bar_options, default='National')

    def pick_out_region_name(bar_option):
        if bar_option.startswith('ISDN: '):
            str_selected_region = bar_option.split('ISDN: ')[-1]
            col_region = 'isdn'
        elif bar_option.startswith('ICB: '):
            str_selected_region = bar_option.split('ICB: ')[-1]
            col_region = 'icb'
        elif bar_option.startswith('Ambulance service: '):
            str_selected_region = bar_option.split('Ambulance service: ')[-1]
            col_region = 'ambo22'
        elif bar_option.startswith('Nearest unit: '):
            str_selected_region_name = bar_option.split('Nearest unit: ')[-1]
            # Convert unit name to postcode:
            mask = df_unit_services['ssnap_name'] == str_selected_region_name
            str_selected_region = df_unit_services.loc[mask].index[0]
            col_region = 'nearest_ivt_unit'
        else:
            str_selected_region = 'National'
            col_region = 'national'
        return str_selected_region, col_region

    highlighted_regions = []
    region_types = []
    for h in highlighted_options:
        s, c = pick_out_region_name(h)
        highlighted_regions.append(s)
        region_types.append(c)
    df = pd.DataFrame(np.vstack((highlighted_regions, region_types)).T,
                      columns=['highlighted_region', 'region_type'])

    return df


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
    mask = df_unit_services_full['Use_IVT'] == 1
    nearest_ivt_unit_names_list = sorted(
        df_unit_services_full.loc[mask, 'ssnap_name'])

    # Key for region type, value for list of options.
    region_options_dict = {
        'ISDN': isdn_list,
        'ICB': icb_list,
        'Nearest unit': nearest_ivt_unit_names_list,
        'Ambulance service': ambo_list
    }

    return region_options_dict


def calculate_nested_average_outcomes(
        dict_outcomes,
        dict_region_admissions_unique,
        region_types,
        df_highlight=None,
        _log=True, _log_loc=None,
        ):
    """
    dict_region_admissions_unique_treatment_times

    If highlighted teams are given, calculate only the data for
    those teams and store the mixed region types in a single dataframe.
    If all regions are being calculated, then calculate the
    data for all regions and store a separate df for each region type. 
    """
    d = {}
    for subgroup, dict_subgroup_outcomes in dict_outcomes.items():
        d[subgroup] = {}
        for scenario, df_subgroup_outcomes in dict_subgroup_outcomes.items():
            d[subgroup][scenario] = {}
            for lsoa_subset, region_dicts in (
                    dict_region_admissions_unique.items()
                    ):
                if df_highlight is not None:
                    # Gather dataframes for different regions types
                    # in here:
                    list_df_h = []
                else:
                    d[subgroup][scenario][lsoa_subset] = {}
                for region_type in region_types:
                    admissions_df = region_dicts[region_type]
                    if df_highlight is not None:
                        # Limit to only the highlighted teams:
                        mask = (df_highlight['region_type'] ==
                                region_type)
                        teams_here = df_highlight.loc[
                            mask, 'highlighted_region']
                        sc = ((region_type == 'nearest_ivt_unit') &
                              (lsoa_subset == 'nearest_unit_no_mt'))
                        if sc:
                            # Special case: expect that CSCs will be missing
                            # for the lsoa subset of only areas whose nearest
                            # unit does not provide MT. So drop CSCs here.
                            teams_removed = list(
                                set(teams_here) -
                                set(list(admissions_df.columns))
                                )
                            teams_here = [t for t in teams_here if t in
                                          admissions_df.columns]
                        else:
                            pass
                        admissions_df = admissions_df[teams_here]
                    else:
                        sc = False
                    # Gather weighted outcomes for these teams:
                    df = calculate_region_outcomes(
                        admissions_df,
                        df_subgroup_outcomes,
                        _log=False
                        )
                    if sc:
                        # Placeholder data for CSCs.
                        for t in teams_removed:
                            df.loc[t] = pd.NA
                    else:
                        pass
                    if df_highlight is not None:
                        list_df_h.append(df)
                    else:
                        # Store full df for each region type in here:
                        d[subgroup][scenario][lsoa_subset][region_type] = df

                if df_highlight is not None:
                    # Combine all region types into one dataframe:
                    df_h = pd.concat(list_df_h, axis='rows')
                    d[subgroup][scenario][lsoa_subset] = df_h
    if _log:
        if df_highlight is not None:
            p = 'Found average outcomes for each highlighted region.'
        else:
            p = 'Found average outcomes for selected region type.'
        print_progress_loc(p, _log_loc)
    return d


def display_region_summary(series_u, series_r, k='mrs_0-2'):
    """
    Show metrics of key outcomes.

    u is usual care, r is redirection available.
    """
    label_dict = {
        'mrs_0-2': {
            'label': 'Proportion with mRS<2',
            'format': '.1%',
            'delta_color': 'normal',
        },
        'mrs_shift': {
            'label': 'Change in mRS score',
            'format': '.3f',
            'delta_color': 'inverse',
        },
        'utility_shift': {
            'label': 'Change in utility',
            'format': '.3f',
            'delta_color': 'normal',
        },
    }
    s = 'from usual care'
    d = series_r[k] - series_u[k]
    st.metric(
        label_dict[k]['label'],
        value=f"{series_r[k]:^{label_dict[k]['format']}}",
        delta=f"{d:^{label_dict[k]['format']}} {s}",
        delta_color=label_dict[k]['delta_color']
        )
    st.write(series_r[k], series_u[k])
    st.write(series_r[f'{k}_std'], series_u[f'{k}_std'])


def plot_mrs_bars(
        mrs_lists_dict, title_text='', return_fig=False, key=None
        ):
    # fig = go.Figure()
    subplot_titles = [
        'Discharge disability<br>probability distribution',
        'Cumulative probability<br>of discharge disability'
    ]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=subplot_titles, shared_xaxes=True)
    fig.update_layout(xaxis_showticklabels=True)

    for label, mrs_dict in mrs_lists_dict.items():

        fig.add_trace(go.Bar(
            x=[*range(7)],
            y=mrs_dict['noncum'],
            error_y=dict(
                type='data',
                array=mrs_dict['std'],
                visible=True),
            name=mrs_dict['label'],
            legendgroup=1,
            marker_color=mrs_dict['colour'],
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[*range(7)],
            y=mrs_dict['cum'],
            name=mrs_dict['label'],
            legendgroup=2,
            marker_color=mrs_dict['colour'],
            mode='lines',
            line=dict(dash=mrs_dict['linestyle'])
            ), row=2, col=1)

    fig.update_layout(barmode='group')
    # Bump the second half of the legend downwards:
    # (bump amount is eyeballed based on fig height)
    fig.update_layout(legend_tracegroupgap=240)

    fig.update_layout(title=title_text)
    for row in [1, 2]:  # 'all' doesn't work for some reason
        fig.update_xaxes(
            title_text='Discharge disability (mRS)',
            # Ensure that all mRS ticks are shown:
            tickmode='linear',
            tick0=0,
            dtick=1,
            row=row, col=1
            )
    fig.update_yaxes(title_text='Probability', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative probability', row=2, col=1)

    # Figure setup.
    # Give enough of a top margin that the main title doesn't
    # clash with the top subplot title.
    fig.update_layout(
        # width=1200,
        height=700,
        margin_t=150,
        )
    fig = update_plotly_font_sizes(fig)

    if return_fig:
        return fig
    else:
        # Options for the mode bar.
        # (which doesn't appear on touch devices.)
        plotly_config = {
            # Mode bar always visible:
            # 'displayModeBar': True,
            # Plotly logo in the mode bar:
            'displaylogo': False,
            # Remove the following from the mode bar:
            'modeBarButtonsToRemove': [
                # 'zoom',
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
            config=plotly_config,
            key=key
            )


def select_full_data_type():
    region_types = ['lsoa', 'national', 'icb', 'isdn', 'ambo22',
                    'nearest_ivt_unit']
    dict_labels = {
        'lsoa': 'Full results by LSOA',
        'national': 'National average',
        'icb': 'Integrated Care Board',
        'isdn': 'Integrated Stroke Delivery Network',
        'ambo22': 'Ambulance service',
        'nearest_ivt_unit': 'Nearest unit with IVT'
    }

    def f(label):
        """Display layer with nice name instead of key."""
        return dict_labels[label]
    # Pick a layer to use for calculating population results:
    full_data_type = st.selectbox(
        'Choose a region type for the full results.',
        options=region_types,
        format_func=f,
        index=3,
        )
    return full_data_type


def load_or_calculate_region_outlines(
        outline_name,
        df_lsoa,
        col_lhs='nearest_ivt_unit_name',
        col_rhs='nearest_mt_unit_name',
        ):
    """
    Don't replace these outlines with stroke-maps!
    These versions match the simplified LSOA shapes.
    """
    # Load in another gdf:

    if outline_name == 'ISDN':
        load_gdf_catchment = True
        outline_file = './data/outline_isdns.geojson'
        outline_names_col = 'isdn'
    elif outline_name == 'ICB':
        load_gdf_catchment = True
        outline_file = './data/outline_icbs.geojson'
        outline_names_col = 'icb'  # to display
    elif outline_name == 'Ambulance service':
        load_gdf_catchment = True
        outline_file = './data/outline_ambo22s.geojson'
        outline_names_col = 'ambo22'  # to display
    elif outline_name == 'Nearest service':
        load_gdf_catchment = False
        outline_names_col = 'Nearest service'

        # Make catchment area polygons:
        gdf_catchment_lhs = dissolve_polygons_by_value(
            df_lsoa.copy().reset_index()[['lsoa', col_lhs]],
            col=col_lhs,
            load_msoa=True
            )
        gdf_catchment_lhs = gdf_catchment_lhs.rename(
            columns={col_lhs: 'Nearest service'})

        gdf_catchment_rhs = dissolve_polygons_by_value(
            df_lsoa.copy().reset_index()[['lsoa', col_rhs]],
            col=col_rhs,
            load_msoa=True
            )
        gdf_catchment_rhs = gdf_catchment_rhs.rename(
            columns={col_rhs: 'Nearest service'})

    if load_gdf_catchment:
        gdf_catchment_lhs = geopandas.read_file(outline_file)
        # Convert to British National Grid:
        gdf_catchment_lhs = gdf_catchment_lhs.to_crs('EPSG:27700')
        # st.write(gdf_catchment['geometry'])
        # # Make geometry valid:
        # gdf_catchment['geometry'] = [
        #     make_valid(g) if g is not None else g
        #     for g in gdf_catchment['geometry'].values
        #     ]
        gdf_catchment_rhs = gdf_catchment_lhs.copy()

    # Make colour transparent:
    gdf_catchment_lhs['colour'] = 'rgba(0, 0, 0, 0)'
    gdf_catchment_rhs['colour'] = 'rgba(0, 0, 0, 0)'
    # Make a dummy column for the legend entry:
    gdf_catchment_lhs['outline_type'] = outline_name
    gdf_catchment_rhs['outline_type'] = outline_name

    gdf_catchment_pop = gdf_catchment_lhs.copy()
    return (
        outline_names_col,
        gdf_catchment_lhs,
        gdf_catchment_rhs,
        gdf_catchment_pop
    )


def plot_basic_travel_options():
    fig = go.Figure()

    t = 2.0
    label_y_off = 0.5
    t_max = t*np.cos(45*np.pi/180.0)
    coords_dict = {
        'patient': [0, 0],
        'csc': [t, 0],
        'psc': [t*0.5, t_max]
    }
    arrow_kwargs = dict(
        mode='lines+markers',
        marker=dict(size=20, symbol='arrow-up', angleref='previous', standoff=16),
        showlegend=False,
        hoverinfo='skip',
    )

    fig.add_trace(go.Scatter(
        x=[coords_dict['patient'][0], coords_dict['psc'][0], coords_dict['csc'][0],],
        y=[coords_dict['patient'][1], coords_dict['psc'][1], coords_dict['csc'][1],],
        **arrow_kwargs,
        line=dict(color='grey', width=10),
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['patient'][0], coords_dict['csc'][0],],
        y=[coords_dict['patient'][1], coords_dict['csc'][1],],
        **arrow_kwargs,
        line=dict(color='#ff4b4b', width=10),
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['patient'][0]],
        y=[coords_dict['patient'][1]],
        text=['🏠'],
        mode='text+markers',
        textfont=dict(size=32),
        marker=dict(size=0, color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['psc'][0]],
        y=[coords_dict['psc'][1]],
        mode='markers',
        marker=dict(size=20, symbol='circle', color='white', line={'color': 'black', 'width': 1},),
        name='IVT unit',
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['csc'][0]],
        y=[coords_dict['csc'][1]],
        mode='markers',
        marker=dict(size=26, symbol='star', color='white', line={'color': 'black', 'width': 1},),
        name='MT unit',
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.add_annotation(
        y=coords_dict['patient'][1] - label_y_off,
        x=coords_dict['patient'][0],
        text='Patient<br>location',
        showarrow=False,
        font=dict(size=14),
        yanchor='top',
        )
    fig.add_annotation(
        y=coords_dict['psc'][1] + label_y_off,
        x=coords_dict['psc'][0],
        text='IVT unit',
        showarrow=False,
        font=dict(size=14),
        yanchor='bottom',
        )
    fig.add_annotation(
        y=coords_dict['csc'][1] - label_y_off,
        x=coords_dict['csc'][0],
        text='MT unit',
        showarrow=False,
        font=dict(size=14),
        yanchor='top',
        )

    # Set axes properties
    fig.update_xaxes(range=[-label_y_off, t+label_y_off],
                     zeroline=False, showgrid=False, showticklabels=False)
    fig.update_yaxes(range=[-3.0*label_y_off, t_max+3.0*label_y_off],
                     zeroline=False, showgrid=False, showticklabels=False)
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    # Set figure size
    fig.update_layout(width=200, height=200, margin_t=0, margin_b=0,
                      margin_l=0, margin_r=0,)

    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title_text='')
    fig.update_layout(dragmode=False)  # change from default zoombox
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
    st.plotly_chart(
        fig,
        width='content',
        config=plotly_config
        )
