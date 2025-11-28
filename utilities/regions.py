"""
Geography.
"""
# ----- Imports -----
import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.stats.weightstats import DescrStatsW  # for mRS dist stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stroke_maps.load_data

from classes.geography_processing import Geoprocessing
from utilities.utils import print_progress_loc, update_plotly_font_sizes, \
    set_inputs_changed, set_rerun_full_results, set_rerun_region_summaries, \
    make_formatted_time_str, set_rerun_lsoa_units_times


# ----- Functions -----
@st.cache_data
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
        df_unit_services.insert(
            3, 'Use_MSU', df_unit_services['Use_MT'].copy())

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


def select_unit_services(use_msu=False):
    """
    """
    # If this has already been loaded in, keep that version instead
    # so changes are retained:
    try:
        df_unit_services = st.session_state['input_df_unit_services']
    except KeyError:
        # Load stroke unit data from file:
        df_unit_services = import_stroke_unit_services(
            use_msu=use_msu,
            keep_only_ivt_mt=False,
            keep_only_england=True
            )
        df_unit_services.index.name = 'Postcode'

    # Manually apply the edits from data_editor.
    try:
        units_data_editor = st.session_state['units_data_editor']
        # Update each of the changes listed in this dict:
        for ind in list(units_data_editor['edited_rows'].keys()):
            for col in list(units_data_editor['edited_rows'][ind].keys()):
                # Which row in the dataframe is this?
                ind_name = df_unit_services.iloc[[ind]].index
                # Is the value True or False?
                val = units_data_editor['edited_rows'][ind][col]
                val = 1 if val is True else 0
                # Update this value in the dataframe:
                df_unit_services.loc[ind_name, col] = val
        # Delete the changelog:
        del st.session_state['units_data_editor']
        # ^ otherwise the same edits would be applied again
        # as soon as the data_editor widget is rendered.
    except KeyError:
        # The edit dict doesn't exist yet.
        # This is the first run of the script.
        pass

    # Force MT units to provide IVT:
    df_unit_services.loc[df_unit_services['Use_MT'] == 1, 'Use_IVT'] = 1

    # Store a copy of the dataframe with our edits:
    st.session_state['input_df_unit_services'] = df_unit_services.copy()

    # Add some junk data so that data_editor thinks it's a new df:
    # (prevents storing multiple changes in the edited_rows)
    df_unit_services.loc[df_unit_services.index[0], 'junk'] = np.random.rand(1)

    # Display this as an editable dataframe:
    st.data_editor(
        df_unit_services,
        column_order=['Postcode', 'ssnap_name', 'Use_IVT', 'Use_MT', 'isdn'],
        disabled=['Postcode', 'ssnap_name', 'isdn'],
        # height=180  # limit height to show fewer rows
        # Make columns display as checkboxes instead of 0/1 ints:
        column_config={
            'Use_IVT': st.column_config.CheckboxColumn(),
            'Use_MT': st.column_config.CheckboxColumn(),
            'Use_MSU': st.column_config.CheckboxColumn(),
        },
        key='units_data_editor',
        on_change=set_rerun_lsoa_units_times,
        )
    # Do not keep a copy of the returned edited dataframe.
    # We'll update it ourselves when the script reruns.
    # The script reruns immediately after the dataframe is edited
    # or a button is pressed.

    # Delete junk:
    df_unit_services = df_unit_services.drop('junk', axis='columns')

    return df_unit_services


def select_unit_services_muster(
        use_msu=True,
        container_dataeditor=None,
        container_buttons=None
        ):
    """
    """
    # If this has already been loaded in, keep that version instead
    # so changes are retained:
    try:
        df_unit_services = st.session_state['df_unit_services']
    except KeyError:
        # Load stroke unit data from file:
        df_unit_services = import_stroke_unit_services(
            use_msu=use_msu,
            keep_only_ivt_mt=False,
            keep_only_england=True
            )

    # Select either:
    # + MSU at all IVT-only units
    # + MSU at all MT units
    # + MSU at all IVT and/or MT units
    with container_buttons:
        # n_cols = 6
        # cols = st.columns(n_cols)
        # i = 0
        # with cols[i % n_cols]:
        add_all_ivt = st.button('Place MSU at all IVT-only units',
                                on_click=set_inputs_changed)
        # i += 1
        # with cols[i % n_cols]:
        add_all_mt = st.button('Place MSU at all MT units',
                               on_click=set_inputs_changed)
        # i += 1
        # with cols[i % n_cols]:
        add_all = st.button('Place MSU at all units',
                            on_click=set_inputs_changed)
        # i += 1
        # with cols[i % n_cols]:
        remove_all_ivt = st.button('Remove MSU from all IVT-only units',
                                   on_click=set_inputs_changed)
        # i += 1
        # with cols[i % n_cols]:
        remove_all_mt = st.button('Remove MSU from all MT units',
                                  on_click=set_inputs_changed)
        # i += 1
        # with cols[i % n_cols]:
        remove_all = st.button('Remove MSU from all units',
                               on_click=set_inputs_changed)

    # Which units need to be changed in each case:
    units_ivt_bool = (
        (df_unit_services['Use_IVT'] == 1) &
        (df_unit_services['Use_MT'] == 0)
    )
    units_mt_bool = (
        (df_unit_services['Use_MT'] == 1)
    )
    # Apply change of the last button pressed.
    # The button is only True if it was pressed on the last run
    # of the script.
    if add_all_ivt:
        df_unit_services.loc[units_ivt_bool, 'Use_MSU'] = 1
    if add_all_mt:
        df_unit_services.loc[units_mt_bool, 'Use_MSU'] = 1
    if add_all:
        df_unit_services['Use_MSU'] = 1
    if remove_all_ivt:
        df_unit_services.loc[units_ivt_bool, 'Use_MSU'] = 0
    if remove_all_mt:
        df_unit_services.loc[units_mt_bool, 'Use_MSU'] = 0
    if remove_all:
        df_unit_services['Use_MSU'] = 0

    # Manually apply the edits from data_editor.
    try:
        units_data_editor = st.session_state['units_data_editor']
        # Update each of the changes listed in this dict:
        for ind in list(units_data_editor['edited_rows'].keys()):
            for col in list(units_data_editor['edited_rows'][ind].keys()):
                # Which row in the dataframe is this?
                ind_name = df_unit_services.iloc[[ind]].index
                # Is the value True or False?
                val = units_data_editor['edited_rows'][ind][col]
                val = 1 if val is True else 0
                # Update this value in the dataframe:
                df_unit_services.loc[ind_name, col] = val
        # Delete the changelog:
        del st.session_state['units_data_editor']
        # ^ otherwise the same edits would be applied again
        # as soon as the data_editor widget is rendered.
    except KeyError:
        # The edit dict doesn't exist yet.
        # This is the first run of the script.
        pass

    # Force MT units to provide IVT:
    df_unit_services.loc[df_unit_services['Use_MT'] == 1, 'Use_IVT'] = 1

    # Store a copy of the dataframe with our edits:
    st.session_state['df_unit_services'] = df_unit_services.copy()

    # Add some junk data so that data_editor thinks it's a new df:
    # (prevents storing multiple changes in the edited_rows)
    df_unit_services.loc[df_unit_services.index[0], 'junk'] = np.random.rand(1)

    # Display data_editor to collect changes from the user:
    with container_dataeditor:
        st.data_editor(
            df_unit_services,
            column_order=['postcode', 'ssnap_name',
                          'Use_IVT', 'Use_MT', 'Use_MSU', 'isdn'],
            disabled=['postcode', 'ssnap_name', 'isdn'],
            # height=180  # limit height to show fewer rows
            # Make columns display as checkboxes instead of 0/1 ints:
            column_config={
                'Use_IVT': st.column_config.CheckboxColumn(),
                'Use_MT': st.column_config.CheckboxColumn(),
                'Use_MSU': st.column_config.CheckboxColumn(),
            },
            key='units_data_editor',
            on_change=set_rerun_lsoa_units_times
            )
    # Do not keep a copy of the returned edited dataframe.
    # We'll update it ourselves when the script reruns.
    # The script reruns immediately after the dataframe is edited
    # or a button is pressed.

    # Delete junk:
    df_unit_services = df_unit_services.drop('junk', axis='columns')
    return df_unit_services


def find_nearest_units_each_lsoa(
        df_unit_services,
        use_msu=False,
        _log=True,
        _log_loc=None
        ):
    """

    Result
    ------
    df_geo - pd.Dataframe. Columns 'LSOA', 'nearest_ivt_unit',
             'nearest_ivt_time', 'nearest_mt_unit', 'nearest_mt_time',
             'transfer_unit', 'transfer_required', 'transfer_time',
             'nearest_msu_unit', 'nearest_msu_time', 'Admissions',
             'nearest_ivt_then_mt_time'
    """
    # try:
    #     geo = st.session_state['geo']
    # except KeyError:
    # Process and save geographic data
    # (only needed when hospital data changes)
    geo = Geoprocessing(
        limit_to_england=True,
        use_msu=use_msu
        )
    # Update units:
    geo.df_unit_services = df_unit_services
    geo.update_unit_services()
    # Rerun geography:
    geo.run()
    # Reset index because Model expects a column named 'lsoa':
    df_geo = geo.get_combined_data().copy(deep=True).reset_index()
    # Separate column for separate travel time including transfer:
    df_geo['nearest_ivt_then_mt_time'] = (
        df_geo['nearest_ivt_time'] + df_geo['transfer_time'])

    # # Cache the geo class so that on the next run all of the big
    # # data files are not loaded in another time.
    # st.session_state['geo'] = geo

    if _log:
        p = 'Assigned LSOA to nearest units.'
        print_progress_loc(p, _log_loc)
    return df_geo


def calculate_extra_muster_travel_times(
        df_lsoa_units_times,
        df_pathway_steps
        ):
    # Make more travel times.
    # MSU:
    s = df_pathway_steps.loc['scale_msu_travel_times', 'value']
    df_lsoa_units_times['msu_response_time'] = (
        s * df_lsoa_units_times['nearest_msu_time'])
    df_lsoa_units_times['msu_response_then_mt_time'] = (
        df_lsoa_units_times['msu_response_time'] + (
            s * df_lsoa_units_times['nearest_mt_time']
        )
    )
    # Usual care:
    a = df_pathway_steps.loc['process_time_ambulance_response', 'value']
    df_lsoa_units_times['ambo_response_then_nearest_ivt_time'] = (
        a + df_lsoa_units_times['nearest_ivt_time'])
    df_lsoa_units_times['ambo_response_then_nearest_mt_time'] = (
        a + df_lsoa_units_times['nearest_mt_time'])
    df_lsoa_units_times['ambo_response_then_nearest_ivt_then_mt_time'] = (
        a + df_lsoa_units_times['nearest_ivt_then_mt_time'])
    return df_lsoa_units_times


def find_unique_travel_times(
        df_times,
        cols_ivt=['nearest_ivt_time', 'nearest_mt_time'],
        cols_mt=['nearest_ivt_then_mt_time', 'nearest_mt_time'],
        cols_pairs={
            'transfer': ('nearest_ivt_time', 'nearest_ivt_then_mt_time'),
            'no_transfer': ('nearest_mt_time', 'nearest_mt_time')
        },
        cols_pairs_labels=['travel_for_ivt', 'travel_for_mt'],
        _log=True,
        _log_loc=None
        ):
    """
    Initial ambulance travel time is the time from the
    MSU base (MUSTER) or the fixed ambulance response time
    (usual care). In OPTIMIST, the fixed time is already
    included in the "treatment times without travel".
    """
    # In usual care,
    # IVT can either be at the nearest unit or at the MT unit if
    # redirected. MT is always at the MT unit, either travelling
    # there directly or going via the IVT-only unit.
    # With an MSU, IVT is either given in the MSU or not at all.
    times_to_ivt = sorted(list(set(df_times[cols_ivt].values.flatten())))
    times_to_mt = sorted(list(set(df_times[cols_mt].values.flatten())))

    # Find all pairs of times.
    # For usual care, combinations are:
    #     IVT at nearest unit, then MT after transfer;
    #     IVT and MT at nearest MT unit.
    # For MSU, also have: IVT in the MSU, then MT at nearest MT unit.
    all_pairs = {}
    for label, pair in cols_pairs.items():
        pairs_here = df_times[list(pair)].drop_duplicates()
        # Don't use rename dictionary because can have duplicate
        # column names in pair.
        pairs_here.columns = cols_pairs_labels
        all_pairs[label] = pairs_here

    if _log:
        p = 'Gathered unique travel times.'
        print_progress_loc(p, _log_loc)
    return times_to_ivt, times_to_mt, all_pairs


def find_region_admissions_by_unique_travel_times(
        df_lsoa_units_times,
        region_types,
        df_highlighted_regions=None,
        keep_only_england=True,
        unique_travel=True,
        project='optimist',
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

    region_types = ['national', 'icb', 'isdn', 'ambo22', 'nearest_ivt_unit']
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
        if project == 'optimist':
            cols_to_merge += ['nearest_ivt_time', 'nearest_mt_time',
                              'nearest_ivt_then_mt_time']
            time_cols_dict = {
                'usual_care_ivt': ['nearest_ivt_time'],
                'usual_care_mt': ['nearest_ivt_then_mt_time'],
                'redirection_ivt': ['nearest_mt_time'],
                'redirection_mt': ['nearest_mt_time']
                }
        else:
            cols_to_merge += [
                'ambo_response_then_nearest_ivt_time',
                'ambo_response_then_nearest_ivt_then_mt_time',
                'msu_response_time',
                'msu_response_then_mt_time',
                ]
            time_cols_dict = {
                'usual_care_ivt': ['ambo_response_then_nearest_ivt_time'],
                'usual_care_mt': [
                    'ambo_response_then_nearest_ivt_then_mt_time'],
                'msu_ivt_ivt': ['msu_response_time'],
                'msu_ivt_mt': ['msu_response_then_mt_time'],
                'msu_no_ivt_ivt': ['msu_response_time'],
                'msu_no_ivt_mt': ['msu_response_then_mt_time'],
                }
    else:
        if project == 'optimist':
            scens = ['usual_care', 'redirection_approved',
                     'redirection_rejected']
        else:
            scens = ['usual_care', 'msu_ivt', 'msu_no_ivt']
        treats = ['ivt', 'mt']
        cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
        if 'msu_no_ivt' in scens:
            cols_treat_scen.remove('msu_no_ivt_ivt')
        # cols_treat_scen = [c for c in cols_treat_scen
        #                    if c in df_lsoa_units_times.columns]
        cols_to_merge += cols_treat_scen

    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_units_times[cols_to_merge],
        left_on='lsoa', right_on='LSOA', how='right'
        )
    # Calculate this separately for each region type.
    dict_region_unique_times = {}
    masks = {'all_patients': slice(None),
             'nearest_unit_no_mt': df_lsoa_regions['transfer_required']}

    if unique_travel:
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        for mask_label, mask in masks.items():
            dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            for time_label, time_cols in time_cols_dict.items():
                dfs_to_concat = []
                for region_type in region_types:
                    # dict_region_unique_times[mask_label][region_type] = {}
                    if region_type == 'national':
                        cols = ['Admissions'] + time_cols
                        df = df_here[cols].groupby(time_cols).sum()
                        df = df.rename(columns={'Admissions': 'National'})
                    else:
                        if isinstance(df_highlighted_regions, pd.DataFrame):
                            # Limit to only the highlighted regions:
                            mask = (df_highlighted_regions['region_type']
                                    == region_type)
                            regions_here = df_highlighted_regions.loc[
                                mask, 'highlighted_region']
                            mask = df_here[region_type].isin(regions_here)
                        else:
                            # Include all regions:
                            mask = slice(None)
                        cols = [region_type, 'Admissions'] + time_cols
                        df = df_here.loc[mask, cols].groupby(
                            [region_type, *time_cols]).sum()
                        # df has columns for region, time, and admissions.
                        # Change to index of time, one column per region,
                        # values of admissions:
                        df = (df.unstack(time_cols).transpose()
                              .reset_index().set_index(time_cols)
                              .drop('level_0', axis='columns')
                              )
                    # dict_region_unique_times[
                    #     mask_label][region_type][time_label] = df
                    dfs_to_concat.append(df)
                df = pd.concat(dfs_to_concat, axis='columns')
                dict_region_unique_times[mask_label][time_label] = df

        if _log:
            p = '''Found total admissions with each unique travel
            time per region.'''
            print_progress_loc(p, _log_loc)
    else:
        # Unique treatment time combinations.
        # For usual care, all IVT is at "nearest ivt unit"
        # and all MT is at "nearest MT unit" after "time to nearest
        # ivt unit plus transfer time" (for no transfer, time is
        # zero). Under redirection, all IVT and all MT is at
        # "nearest MT unit".
        for mask_label, mask in masks.items():
            # dict_region_unique_times[mask_label] = {}
            df_here = df_lsoa_regions.loc[mask]
            dfs_to_concat = []
            for region_type in region_types:
                if region_type == 'national':
                    cols = ['Admissions'] + cols_treat_scen
                    df = df_here[cols].groupby(cols_treat_scen).sum()
                    df = df.rename(columns={'Admissions': 'National'})
                else:
                    if isinstance(df_highlighted_regions, pd.DataFrame):
                        # Limit to only the highlighted regions:
                        mask = (df_highlighted_regions['region_type']
                                == region_type)
                        regions_here = df_highlighted_regions.loc[
                            mask, 'highlighted_region']
                        mask = df_here[region_type].isin(regions_here)
                    else:
                        # Include all regions:
                        mask = slice(None)
                    cols = [region_type, 'Admissions'] + cols_treat_scen
                    df = df_here.loc[mask, cols].groupby(
                        [region_type, *cols_treat_scen]).sum()
                    # df has columns for region, time, and admissions.
                    # Change to index of time, one column per region,
                    # values of admissions:
                    df = (df.unstack(cols_treat_scen).transpose()
                          .reset_index().set_index(cols_treat_scen)
                          .drop('level_0', axis='columns')
                          )
                dfs_to_concat.append(df)
            df = pd.concat(dfs_to_concat, axis='columns')
            dict_region_unique_times[mask_label] = df
            # dict_region_unique_times[mask_label][region_type] = df

        if _log:
            p = '''Found total admissions with each set of unique
            treatment times per region.'''
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

    highlighted_options = st.multiselect(
        'Regions to highlight', bar_options, default='National',
        on_change=set_rerun_region_summaries)

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
            for lsoa_subset, admissions_df in (
                    dict_region_admissions_unique.items()
                    ):
                # Gather weighted outcomes for these teams:
                df = calculate_region_outcomes(
                    admissions_df,
                    df_subgroup_outcomes,
                    _log=False
                )
                # Store full df for each region type in here:
                d[subgroup][scenario][lsoa_subset] = df
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
    # st.write(series_r[k], series_u[k])
    # st.write(series_r[f'{k}_std'], series_u[f'{k}_std'])


def plot_mrs_bars_plus_cumulative(
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
    fig.update_layout(
        # width=1200,
        height=700,
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
            config=plotly_config,
            key=key
            )


def plot_mrs_bars(
        mrs_lists_dict, key=None
        ):
    fig = go.Figure()

    for label, mrs_dict in mrs_lists_dict.items():
        if 'std' in mrs_dict.keys():
            error_y = dict(
                type='data',
                array=100.0*mrs_dict['std'],
                visible=True
                )
        else:
            error_y = None
        fig.add_trace(go.Bar(
            x=[*range(7)],
            y=100.0*mrs_dict['noncum'],
            error_y=error_y,
            name=mrs_dict['label'],
            marker_color=mrs_dict['colour'],
            ))

    fig.update_layout(barmode='group')

    fig.update_layout(xaxis_showticklabels=True)
    fig.update_xaxes(
        title_text='Discharge disability (mRS)',
        # Ensure that all mRS ticks are shown:
        tickmode='linear',
        tick0=0,
        dtick=1,
        )
    fig.update_yaxes(title_text='Probability (%)')
    fig.update_layout(legend=dict(
        yanchor='top',
        y=-0.4,
        yref='paper',
        xanchor='center',
        x=0.5,
        orientation='h'
    ))
    # Figure setup.
    fig.update_layout(height=250, margin_t=0)
    fig = update_plotly_font_sizes(fig)
    fig.update_layout(title='')
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
        on_change=set_rerun_full_results
        )
    return full_data_type


def plot_basic_travel_options():
    fig = go.Figure()

    t = 2.0
    label_y_off = 0.5
    t_max = t*np.cos(45*np.pi/180.0)
    coords_dict = {
        'patient': [0, 0],
        'csc': [t*1.2, 0],
        'psc': [t*0.5, t_max]
    }
    arrow_kwargs = dict(
        mode='lines+markers',
        marker=dict(size=20, symbol='arrow-up', angleref='previous',
                    standoff=16),
        showlegend=False,
        hoverinfo='skip',
    )

    fig.add_trace(go.Scatter(
        x=[coords_dict['patient'][0], coords_dict['psc'][0],
           coords_dict['csc'][0],],
        y=[coords_dict['patient'][1], coords_dict['psc'][1],
           coords_dict['csc'][1],],
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
        marker=dict(size=20, symbol='circle', color='white',
                    line={'color': 'black', 'width': 1},),
        name='IVT unit',
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['csc'][0]],
        y=[coords_dict['csc'][1]],
        mode='markers',
        marker=dict(size=26, symbol='star', color='white',
                    line={'color': 'black', 'width': 1},),
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
        config=plotly_config,
        width='content',
        )


def plot_basic_travel_options_msu():
    fig = go.Figure()

    t = 2.0
    label_y_off = 0.5
    t_max = t*np.cos(45*np.pi/180.0)
    coords_dict = {
        'patient': [0, 0],
        'csc': [t*1.2, 0],
        'psc': [t*0.5, t_max],
        'ambo': [-t*1.2, 0]
    }
    arrow_kwargs = dict(
        mode='lines+markers',
        marker=dict(size=20, symbol='arrow-up', angleref='previous',
                    standoff=16),
        showlegend=False,
        hoverinfo='skip',
    )

    fig.add_trace(go.Scatter(
        x=[coords_dict['ambo'][0], coords_dict['patient'][0],
           coords_dict['psc'][0], coords_dict['csc'][0],],
        y=[coords_dict['ambo'][1], coords_dict['patient'][1],
           coords_dict['psc'][1], coords_dict['csc'][1],],
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
        x=[coords_dict['ambo'][0]],
        y=[coords_dict['ambo'][1]],
        text=['🚑'],
        mode='text+markers',
        textfont=dict(size=32),
        marker=dict(size=0, color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        hoverinfo='skip',
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
        marker=dict(size=20, symbol='circle', color='white',
                    line={'color': 'black', 'width': 1},),
        name='IVT unit',
        hoverinfo='skip',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['csc'][0]],
        y=[coords_dict['csc'][1]],
        mode='markers',
        marker=dict(size=26, symbol='star', color='white',
                    line={'color': 'black', 'width': 1},),
        name='MT unit',
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.add_annotation(
        y=coords_dict['ambo'][1] - label_y_off,
        x=coords_dict['ambo'][0],
        text='Ambulance<br>location',
        showarrow=False,
        font=dict(size=14),
        yanchor='top',
        )
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

    # MSU version:
    msu_offset = 4
    coords_dict = {
        'msu': [-t*1.2, -msu_offset],
        'patient': [0, -msu_offset],
        'csc': [t*1.2, -msu_offset],
    }
    # Separate arrow traces to break the line at the middle marker:
    fig.add_trace(go.Scatter(
        x=[coords_dict['patient'][0], coords_dict['csc'][0],],
        y=[coords_dict['patient'][1], coords_dict['csc'][1],],
        **arrow_kwargs,
        line=dict(color='#ff4b4b', width=10),
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['msu'][0], coords_dict['patient'][0]],
        y=[coords_dict['msu'][1], coords_dict['patient'][1]],
        **arrow_kwargs,
        line=dict(color='#ff4b4b', width=10),
    ))
    fig.add_trace(go.Scatter(
        x=[coords_dict['msu'][0]],
        y=[coords_dict['msu'][1]],
        text=['🚑'],
        mode='text+markers',
        textfont=dict(size=32),
        marker=dict(size=0, color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        hoverinfo='skip',
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
        x=[coords_dict['csc'][0]],
        y=[coords_dict['csc'][1]],
        mode='markers',
        marker=dict(size=26, symbol='star', color='white',
                    line={'color': 'black', 'width': 1},),
        name='MT unit',
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.add_annotation(
        y=coords_dict['msu'][1] - label_y_off,
        x=coords_dict['msu'][0],
        text='MSU base',
        showarrow=False,
        font=dict(size=14),
        yanchor='top',
        )
    fig.add_annotation(
        y=coords_dict['patient'][1] + label_y_off,
        x=coords_dict['patient'][0],
        text='Patient<br>location',
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
    fig.update_xaxes(range=[-t-label_y_off, t*1.2+label_y_off],
                     zeroline=False, showgrid=False, showticklabels=False)
    fig.update_yaxes(range=[-msu_offset-3.0*label_y_off, t_max+3.0*label_y_off],
                     zeroline=False, showgrid=False, showticklabels=False)
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    # Set figure size
    fig.update_layout(width=400, height=400, margin_t=0, margin_b=0,
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
        config=plotly_config,
        width='content',
        )


def gather_this_region_travel_times(
        dict_highlighted_region_travel_times,
        lsoa_subset,
        region,
        time_cols=['usual_care_ivt', 'usual_care_mt', 'redirection_mt'],
        ):
    time_bins = np.arange(0, 185, 5)
    admissions_times = []
    for t in time_cols:
        times = dict_highlighted_region_travel_times[
            lsoa_subset][t][region].dropna()
        a_times, _ = np.histogram(
            times.index, bins=time_bins, weights=times.values)
        a_times /= a_times.sum()
        admissions_times.append(a_times)
    return time_bins, admissions_times


def plot_travel_times(
        times,
        admissions_lists,
        subplot_titles=[
            'To nearest unit',
            'To nearest then MT unit',
            'To MT unit directly'
            ]
        ):
    fig = make_subplots(rows=3, cols=1, subplot_titles=subplot_titles)
    for i, admissions in enumerate(admissions_lists):
        fig.add_trace(go.Bar(
            x=times,
            y=100.0*admissions,
            name=subplot_titles[i],
            marker_color='#0072b2',
            showlegend=False
        ), row=i+1, col=1)
    fig.update_yaxes(title_text='% patients', row=2, col=1)
    fig.update_xaxes(title_text='Travel time (minutes)', row=3, col=1)
    ticks = np.arange(0, 185, 30)
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=ticks))
    fig.update_layout(xaxis2=dict(tickmode='array', tickvals=ticks))
    fig.update_layout(xaxis3=dict(tickmode='array', tickvals=ticks))
    fig.update_layout(width=210, height=400, margin_t=20)
    fig.update_layout(dragmode=False)  # change from default zoombox
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


# def calculate_average_treatment_times_highlighted_regions(
#         dict_region_admissions_unique,
#         region_types,
#         df_highlight,
#         _log=True, _log_loc=None,
#         ):
#     """
#     dict_region_admissions_unique_treatment_times

#     If highlighted teams are given, calculate only the data for
#     those teams and store the mixed region types in a single dataframe.
#     """
#     d = {}
#     for lsoa_subset, region_dicts in (
#             dict_region_admissions_unique.items()
#             ):
#         list_df_h = []
#         for region_type in region_types:
#             admissions_df = region_dicts[region_type]
#             # Limit to only the highlighted teams:
#             mask = (df_highlight['region_type'] == region_type)
#             teams_here = df_highlight.loc[mask, 'highlighted_region']
#             sc = ((region_type == 'nearest_ivt_unit') &
#                   (lsoa_subset == 'nearest_unit_no_mt'))
#             if sc:
#                 # Special case: expect that CSCs will be missing
#                 # for the lsoa subset of only areas whose nearest
#                 # unit does not provide MT. So drop CSCs here.
#                 teams_removed = list(
#                     set(teams_here) -
#                     set(list(admissions_df.columns))
#                     )
#                 teams_here = [t for t in teams_here if t in
#                               admissions_df.columns]
#             else:
#                 pass
#             admissions_df = admissions_df[teams_here]

#             # Gather weighted treatment times for these teams:
#             df = calculate_average_treatment_times(admissions_df, _log=False)
#             if sc:
#                 # Placeholder data for CSCs.
#                 for t in teams_removed:
#                     df.loc[t] = pd.NA
#             else:
#                 pass
#             list_df_h.append(df)

#         # Combine all region types into one dataframe:
#         df_h = pd.concat(list_df_h, axis='rows')
#         d[lsoa_subset] = df_h
#     if _log:
#         p = 'Found average treatment times for each highlighted region.'
#         print_progress_loc(p, _log_loc)
#     return d


def calculate_average_treatment_times_highlighted_regions(
        dict_region_admissions_unique,
        _log=True, _log_loc=None,
        ):
    """
    dict_region_admissions_unique_treatment_times

    If highlighted teams are given, calculate only the data for
    those teams and store the mixed region types in a single dataframe.
    """
    d = {}
    for lsoa_subset, admissions_df in dict_region_admissions_unique.items():
        # Gather weighted treatment times for these teams:
        df = calculate_average_treatment_times(admissions_df, _log=False)
        d[lsoa_subset] = df
    if _log:
        p = 'Found average treatment times for each highlighted region.'
        print_progress_loc(p, _log_loc)
    return d


def calculate_average_treatment_times(
        df_in,
        _log=True,
        _log_loc=None
        ):
    """
    df_regions contains admission numbers for unique treatment times.
    """
    time_cols = list(df_in.index.names)
    regions = list(df_in.columns)
    df_in = df_in.reset_index()

    cols_out = time_cols
    cols_std = [f'{c}_std' for c in cols_out]
    df_out = pd.DataFrame(columns=cols_out+cols_std)
    for region in regions:
        mask = df_in[region].notna()
        # Take admissions-weighted average of outcomes.
        vals = df_in.loc[mask, time_cols]
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
        s = pd.Series(list(means) + list(stds), index=df_out.columns)
        df_out.loc[region] = s

    if _log:
        p = 'Found average outcomes for each region.'
        print_progress_loc(p, _log_loc)
    return df_out


def make_average_treatment_time_df(s_treats):
    scens_all_labels = {
        'usual_care': 'Usual care',
        'redirection_approved': 'Redirection approved',
        'redirection_rejected': 'Redirection rejected',
        }
    treats_labels = {
        'ivt': 'IVT',
        'mt': 'MT'
        }
    scens_to_use = [s for s in scens_all_labels.keys()
                    if any([v.startswith(s) for v in s_treats.keys()])]
    scens_labels = dict([(k, scens_all_labels[k]) for k in scens_to_use])
    arr_treats = []
    for scen in scens_labels.keys():
        scen_treats = []
        for treat in treats_labels.keys():
            t_mean = s_treats[f'{scen}_{treat}']
            t_mean = make_formatted_time_str(t_mean)
            t_std = s_treats[f'{scen}_{treat}_std']
            if t_std >= 60.0:
                t_std = make_formatted_time_str(t_std)
            else:
                t_std = f'{t_std:02.0f}min'
            t_str = (f"{t_mean}" + r' $\pm$ ' + f"{t_std}")
            scen_treats.append(t_str)
        arr_treats.append(scen_treats)
    df_treats = pd.DataFrame(
        arr_treats,
        columns=treats_labels.values(),
        index=scens_labels.values()
        )
    return df_treats


def calculate_no_treatment_mrs(pops, dict_no_treatment_outcomes):
    cols_mrs = [f'mrs_dists_{i}' for i in range(7)]
    cols_mrs_noncum = [c.replace('dists_', 'dists_noncum_') for c in cols_mrs]

    prop_nlvo = (
        pops[pops.index.str.startswith('nlvo')].sum())
    df_no_treat = (
        (prop_nlvo *
            dict_no_treatment_outcomes['nlvo_no_treatment']) +
        ((1.0 - prop_nlvo) *
            dict_no_treatment_outcomes['lvo_no_treatment'])
    )
    df_no_treat[cols_mrs_noncum] = np.diff(df_no_treat[cols_mrs], prepend=0.0)
    # Round values:
    df_no_treat[cols_mrs_noncum] = np.round(df_no_treat[cols_mrs_noncum], 3)
    df_no_treat = df_no_treat.squeeze()
    return df_no_treat


def find_unit_admissions_by_region(
        df_lsoa_units_times,
        prop_of_all_stroke,
        region_types=[],
        df_highlight=None,
        keep_only_england=True,
        _log=True,
        _log_loc=None
        ):
    """
    region_types = ['national', 'icb', 'isdn', 'ambo22', 'nearest_ivt_unit']
    """
    # Load in LSOA-region lookup:
    df_lsoa_regions = load_lsoa_region_lookups()
    # Columns: 'lsoa', 'lsoa_code', 'region', 'region_code',
    # 'region_type', 'short_code', 'country', 'icb', 'icb_code',
    # 'isdn', 'ambo22'.
    if keep_only_england:
        mask_eng = df_lsoa_regions['region_type'] == 'SICBL'
        df_lsoa_regions = df_lsoa_regions.loc[mask_eng].copy()
    cols_to_merge = ['nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit',
                     'transfer_required', 'Admissions', 'LSOA']
    unit_cols = cols_to_merge[:3]
    df_lsoa_regions = pd.merge(
        df_lsoa_regions, df_lsoa_units_times[cols_to_merge],
        left_on='lsoa', right_on='LSOA', how='right'
        )
    # Calculate this separately for each region type.
    # Just total count of admissions:
    # df_admissions - one column per region,
    # rows for 'admissions_all', 'prop_nearest_unit_no_mt'.
    # Total admissions in this region:
    df_admissions = pd.DataFrame()
    # Calculate for these subgroups of LSOA:
    masks = {'all_patients': slice(None),
             'nearest_unit_no_mt': df_lsoa_regions['transfer_required']}
    # Admissions by stroke unit:
    dfs_to_concat = []
    for region_type in region_types:
        # Limit to only the highlighted teams:
        if isinstance(df_highlight, pd.DataFrame):
            mask = (df_highlight['region_type'] == region_type)
            regions_here = df_highlight.loc[mask, 'highlighted_region']
        else:
            regions_here = list(set(df_lsoa_regions[region_type]))

        for region in regions_here:
            if region_type == 'national':
                mask_reg = slice(None)
            else:
                mask_reg = df_lsoa_regions[region_type] == region
            for mask_label, mask_lsoa in masks.items():
                # Pick out just these LSOA:
                if (region_type == 'national') & (mask_label == 'all_patients'):
                    mask_combo = slice(None)
                elif (region_type == 'national'):
                    mask_combo = mask_lsoa
                elif (mask_label == 'all_patients'):
                    mask_combo = mask_reg
                else:
                    mask_combo = mask_reg & mask_lsoa
                df_admissions.loc[region, f'admissions_{mask_label}'] = (
                     df_lsoa_regions.loc[mask_combo, 'Admissions'].sum() *
                     prop_of_all_stroke
                     )
            # TO DO - also calculate this for the LSOA subset whose nearest unit does not provide MT.
            # TO DO -------------------------------------------------------------------------        TO DO
            # How many people go to each combination of stroke units?
            # First unit, nearest MT unit, transfer unit combos.
            df_unit_admissions = (
                df_lsoa_regions.loc[mask_reg, unit_cols + ['Admissions']].
                groupby(unit_cols).sum() *
                prop_of_all_stroke
            )
            df_unit_admissions = df_unit_admissions.rename(
                columns={'Admissions': region})
            dfs_to_concat.append(df_unit_admissions)
    df_unit_admissions = pd.concat(dfs_to_concat, axis='columns').reset_index()

    df_admissions['prop_nearest_unit_no_mt'] = (
        df_admissions['admissions_nearest_unit_no_mt'] /
        df_admissions['admissions_all_patients']
        )

    if _log:
        p = 'Calculated number of admissions to each stroke unit by region.'
        print_progress_loc(p, _log_loc)
    return df_admissions, df_unit_admissions


def calculate_network_usual_care(df_network, dict_pops_u):
    prop_mt_usual_care = (
        dict_pops_u
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        )
    # Usual care:
    df_net_u = df_network.copy()
    # Convert unit columns to generic strings:
    cols_units = [c for c in df_net_u.columns if 'unit' in c]
    df_net_u[cols_units] = df_net_u[cols_units].astype('string')
    # Combine columns that share nearest and transfer units:
    df_net_u = df_net_u.drop('nearest_mt_unit', axis='columns')
    df_net_u = df_net_u.groupby(['nearest_ivt_unit', 'transfer_unit']
                                ).sum().reset_index()
    df_net_u.insert(0, 'nearest_unit', df_net_u['nearest_ivt_unit'])
    df_net_u = df_net_u.rename(columns={'nearest_ivt_unit': 'first_unit'})
    df_net_u['admissions_catchment_to_first_unit'] = df_net_u['admissions'].copy()
    # Only include transfers for patients who receive MT
    # and who aren't already at an MT unit.
    df_net_u['admissions_first_unit_to_transfer'] = (
        df_net_u['admissions'].copy() * prop_mt_usual_care)
    df_net_u['thrombectomy'] = (
        df_net_u['admissions_first_unit_to_transfer'].copy())
    mask_no_transfer = (df_net_u['first_unit'] == df_net_u['transfer_unit'])
    df_net_u.loc[mask_no_transfer, 'admissions_first_unit_to_transfer'] = 0.0
    df_net_u['nearest_unit'] = 'nearest_' + df_net_u['nearest_unit'].astype(str)
    return df_net_u


def calculate_network_redir(df_network, dict_pops_r):
    prop_no_redir = (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        ['full_population'].sum()
        )
    prop_mt_redir_approved = (
        dict_pops_r[dict_pops_r['scenario'] == 'redir_accepted']
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        ) / (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        ['full_population'].sum()
    )
    prop_mt_redir_not_approved = (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        ) / (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        ['full_population'].sum()
    )
    # Redirection scenario:
    df_net_r = df_network.copy()
    # Convert unit columns to generic strings:
    cols_units = [c for c in df_net_r.columns if 'unit' in c]
    df_net_r[cols_units] = df_net_r[cols_units].astype('string')
    # No redir, similar to usual care above:
    df_net_no_redir = df_net_r.copy()
    df_net_no_redir['admissions'] *= prop_no_redir
    # Combine columns that share nearest and transfer units:
    df_net_no_redir = df_net_no_redir.drop('nearest_mt_unit', axis='columns')
    df_net_no_redir = df_net_no_redir.groupby(
        ['nearest_ivt_unit', 'transfer_unit']).sum().reset_index()
    df_net_no_redir.insert(0, 'nearest_unit', df_net_no_redir['nearest_ivt_unit'])
    df_net_no_redir = df_net_no_redir.rename(
        columns={'nearest_ivt_unit': 'first_unit'})
    df_net_no_redir['admissions_catchment_to_first_unit'] = (
        df_net_no_redir['admissions'].copy())
    # Only include transfers for patients who receive MT
    # and who aren't already at an MT unit.
    df_net_no_redir['admissions_first_unit_to_transfer'] = (
        df_net_no_redir['admissions'] * prop_mt_redir_not_approved)
    df_net_no_redir['thrombectomy'] = (
        df_net_no_redir['admissions_first_unit_to_transfer'])
    mask_no_transfer = (df_net_no_redir['first_unit'] ==
                        df_net_no_redir['transfer_unit'])
    df_net_no_redir.loc[mask_no_transfer,
                        'admissions_first_unit_to_transfer'] = 0.0
    df_net_no_redir['redirected'] = 0
    # Redirected:
    df_net_redir = df_net_r.copy()
    df_net_redir['admissions'] *= (1.0 - prop_no_redir)
    df_net_redir['admissions_catchment_to_first_unit'] = (
        df_net_redir['admissions'].copy())
    df_net_redir.insert(0, 'nearest_unit', df_net_redir['nearest_ivt_unit'])
    df_net_redir = df_net_redir.rename(columns={'nearest_mt_unit': 'first_unit'})
    df_net_redir = df_net_redir.drop('transfer_unit', axis='columns')
    df_net_redir = df_net_redir.drop('nearest_ivt_unit', axis='columns')
    df_net_redir['thrombectomy'] = (
        df_net_redir['admissions_catchment_to_first_unit'] *
        prop_mt_redir_approved
        )
    df_net_redir['transfer_unit'] = df_net_redir['first_unit']
    df_net_redir['admissions_first_unit_to_transfer'] = 0.0
    df_net_redir['redirected'] = 1
    # Combine redir and no redir:
    df_net_r = pd.concat((df_net_redir, df_net_no_redir), axis='rows')
    # Combine data for patients whose nearest unit is MT:
    df_net_r = df_net_r.groupby(
        ['nearest_unit', 'first_unit', 'transfer_unit']).sum().reset_index()
    df_net_r['nearest_unit'] = 'nearest_' + df_net_r['nearest_unit'].astype(str)
    return df_net_r


def calculate_region_treat_stats(dict_pops_u, dict_pops_r, s_admissions):
    d = {}
    # Numbers redirected:
    d['prop_mt_usual_care'] = (
        dict_pops_u
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        )
    d['prop_mt_redir_approved'] = (
        dict_pops_r[dict_pops_r['scenario'] == 'redir_accepted']
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        )
    d['prop_mt_redir_not_approved'] = (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        )
    d['prop_ivt_only_usual_care'] = (
        dict_pops_u
        .loc[['nlvo_ivt', 'lvo_ivt'], 'full_population'].sum()
        )
    d['prop_ivt_only_redir_approved'] = (
        dict_pops_r[dict_pops_r['scenario'] == 'redir_accepted']
        .loc[['nlvo_ivt', 'lvo_ivt'], 'full_population'].sum()
        )
    d['prop_ivt_only_redir_not_approved'] = (
        dict_pops_r[dict_pops_r['scenario'] != 'redir_accepted']
        .loc[['nlvo_ivt', 'lvo_ivt'], 'full_population'].sum()
        )
    # Number of thrombectomies:
    d['n_mt'] = (
        d['prop_mt_usual_care'] * s_admissions['admissions_all_patients'])
    # Usual care:
    d['n_mt_nearest_unit_has_mt_usual_care'] = (
        d['prop_mt_usual_care'] *
        (s_admissions['admissions_all_patients'] -
         s_admissions['admissions_nearest_unit_no_mt'])
        )
    d['n_mt_nearest_unit_no_mt_and_not_redirected_usual_care'] = (
        d['prop_mt_usual_care'] *
        s_admissions['admissions_nearest_unit_no_mt'])
    d['n_mt_nearest_unit_no_mt_and_redirected_usual_care'] = 0.0
    # Redir scenario:
    d['n_mt_nearest_unit_has_mt_redir'] = (
        d['n_mt_nearest_unit_has_mt_usual_care'])
    d['n_mt_nearest_unit_no_mt_and_not_redirected_redir'] = (
        d['prop_mt_redir_not_approved'] *
        s_admissions['admissions_nearest_unit_no_mt'])
    d['n_mt_nearest_unit_no_mt_and_redirected_redir'] = (
        d['prop_mt_redir_approved'] *
        s_admissions['admissions_nearest_unit_no_mt'])

    # Number of thrombolysis only (not thrombectomy):
    d['n_ivt_only'] = (
        d['prop_ivt_only_usual_care'] *
        s_admissions['admissions_all_patients']
        )
    # Usual care:
    d['n_ivt_only_nearest_unit_has_mt_usual_care'] = (
        d['prop_ivt_only_usual_care'] *
        (s_admissions['admissions_all_patients'] -
         s_admissions['admissions_nearest_unit_no_mt'])
        )
    d['n_ivt_only_nearest_unit_no_mt_and_redirected_usual_care'] = 0.0
    d['n_ivt_only_nearest_unit_no_mt_and_not_redirected_usual_care'] = (
        d['prop_ivt_only_usual_care'] *
        s_admissions['admissions_nearest_unit_no_mt']
        )
    # Redir scenario:
    d['n_ivt_only_nearest_unit_has_mt_redir'] = (
        d['n_ivt_only_nearest_unit_has_mt_usual_care'])
    d['n_ivt_only_nearest_unit_no_mt_and_not_redirected_redir'] = (
        d['prop_ivt_only_redir_not_approved'] *
        s_admissions['admissions_nearest_unit_no_mt']
        )
    d['n_ivt_only_nearest_unit_no_mt_and_redirected_redir'] = (
        d['prop_ivt_only_redir_approved'] *
        s_admissions['admissions_nearest_unit_no_mt']
        )
    return d
