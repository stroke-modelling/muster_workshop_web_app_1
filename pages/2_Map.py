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
from importlib_resources import files
import pandas as pd

from stroke_maps.geo import check_scenario_level
# Custom functions:
import utilities.utils as utils
import utilities.maps as maps
# Containers:
import utilities.container_inputs as inputs
import utilities.container_results as results


@st.cache_data
def main_calculations(input_dict, df_unit_services):
    # Run the outcomes with the selected pathway:
    df_lsoa = results.make_outcomes(input_dict, df_unit_services)

    # TO DO - the results df contains a mix of scenarios
    # (drip and ship, mothership, msu) in the column names.
    # Pull them out and put them into 'scenario' header.
    # Also need to do something with separate nlvo, lvo, treatment types
    # because current setup just wants some averaged added utility outcome
    # rather than split by stroke type.

    # --- MSOAs for geography ---
    df_msoa = inputs.convert_lsoa_to_msoa_results(df_lsoa)

    # Check whether the input DataFrames have a 'scenario' column level.
    # This is required for talking to stroke-maps package.
    # If not, add one now with a placeholder scenario name.
    df_msoa = check_scenario_level(df_msoa)

    # Merge outcome and geography:
    gdf_boundaries_msoa = maps._load_geometry_msoa(df_msoa)

    # --- LSOAs for grouping results ---
    # Merge in other region info.

    # Load region info for each LSOA:
    # Relative import from package files:
    path_to_file = files('stroke_maps.data').joinpath('regions_lsoa_ew.csv')
    df_lsoa_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    df_lsoa = pd.merge(df_lsoa, df_lsoa_regions, left_on='lsoa', right_on='lsoa', how='left')

    # Load further region data linking SICBL to other regions:
    path_to_file = files('stroke_maps.data').joinpath('regions_ew.csv')
    df_regions = pd.read_csv(path_to_file)  # , index_col=[0, 1])
    # Drop columns already in df_lsoa:
    df_regions = df_regions.drop(['region', 'region_type'], axis='columns')
    df_lsoa = pd.merge(df_lsoa, df_regions, left_on='region_code', right_on='region_code', how='left')

    # TO DO - please please please sort out this function that's collecting everything.
    # st calls should be in the main body.
    results_tabs = st.tabs(['Results by IVT unit catchment', 'Results by ISDN', 'Results by ICB', 'Full results by LSOA'])

    # Replace some zeros with NaN:
    mask = df_lsoa['transfer_required']
    df_lsoa.loc[~mask, 'transfer_time'] = pd.NA

    with results_tabs[3]:
        st.markdown('### Results by LSOA')
        st.write(df_lsoa)

    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_lsoa = df_lsoa.drop([
        'lsoa', 'lsoa_code', 'nearest_mt_unit', 'transfer_unit',
        'nearest_msu_unit', 'short_code', 'country'
        ], axis='columns')

    # Glob results by ICB:
    df_icb = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_icb = df_icb.drop([
        'nearest_ivt_unit', 'region', 'region_type', 'region_code', 'icb_code', 'isdn'
        ], axis='columns')
    # Average:
    df_icb = df_icb.groupby('icb').mean()

    # Glob results by ISDN:
    df_isdn = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_isdn = df_isdn.drop([
        'nearest_ivt_unit', 'region', 'region_type', 'region_code', 'icb', 'icb_code'
        ], axis='columns')
    # Average:
    df_isdn = df_isdn.groupby('isdn').mean()

    # Glob results by nearest IVT unit:
    df_nearest_ivt = df_lsoa.copy()
    # Remove string columns:
    # (temporary - I don't know how else to groupby a df with some object columns)
    df_nearest_ivt = df_nearest_ivt.drop([
        'region', 'region_type', 'region_code', 'icb', 'icb_code', 'isdn'
        ], axis='columns')
    # Average:
    df_nearest_ivt = df_nearest_ivt.groupby('nearest_ivt_unit').mean()
    # Merge back in the unit names:
    df_nearest_ivt = pd.merge(
        df_unit_services['stroke_team'],
        df_nearest_ivt, how='right', left_on='Postcode', right_index=True)

    # Set some columns to bool for nicer display:
    cols_bool = ['transfer_required', 'England']
    for col in cols_bool:
        for df in [df_icb, df_isdn, df_nearest_ivt, df_lsoa]:
            df[col] = df[col].astype(bool)

    with results_tabs[0]:
        st.markdown('### Results by nearest IVT unit')
        st.markdown('Results are the mean values of all LSOA in each IVT unit catchment area.')
        st.write(df_nearest_ivt)

    with results_tabs[1]:
        st.markdown('### Results by ISDN')
        st.markdown('Results are the mean values of all LSOA in each ISDN.')
        st.write(df_isdn)

    with results_tabs[2]:
        st.markdown('### Results by ICB')
        st.markdown('Results are the mean values of all LSOA in each ICB.')
        st.write(df_icb)

    return gdf_boundaries_msoa


# ###########################
# ##### START OF SCRIPT #####
# ###########################
# page_setup()
st.set_page_config(
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )

# Make containers:
# +-----------------------+
# |    container_intro    |
# +-----------------------+
# |     container_map     |
# +-----------------------+
# | container_map_inputs  |
# +-----------------------+
# | container_results_tables  |
# +-----------------------+
# |  container_select_outcome   |
# +-----------------------+
container_intro = st.container()
with st.sidebar:
    container_unit_services = st.container()
container_map = st.empty()
container_map_inputs = st.container()
container_results_tables = st.container()
container_select_outcome = st.container()

# ###########################
# ########## SETUP ##########
# ###########################

with container_intro:
    st.markdown('# Benefit in outcomes from Mobile Stroke Units')

# Set up stroke unit services (IVT, MT, MSU).
from stroke_maps.catchment import Catchment
catchment = Catchment()
df_unit_services = catchment.get_unit_services()
# Remove Wales:
df_unit_services = df_unit_services.loc[df_unit_services['region_type'] != 'LHB'].copy()
df_unit_services_full = df_unit_services.copy()
# Limit which columns to show:
cols_to_keep = [
    'stroke_team',
    'use_ivt',
    'use_mt',
    'use_msu',
    # 'transfer_unit_postcode',  # to add back in later if stroke-maps replaces geography_processing class
    # 'region',
    # 'icb',
    'isdn'
]
df_unit_services = df_unit_services[cols_to_keep]
# Change 1/0 columns to bool for formatting:
cols_use = ['use_ivt', 'use_mt', 'use_msu']
df_unit_services[cols_use] = df_unit_services[cols_use].astype(bool)
# Sort by ISDN name for nicer display:
df_unit_services = df_unit_services.sort_values('isdn')


# Draw the input selection boxes in this function:
with st.sidebar:
    with st.form('Model setup'):
        st.header('Pathway inputs')
        input_dict = inputs.select_parameters_map()

        def make_example_str():
            time_not_msu = 20.0
            time_msu = 20.0 * scale_msu_travel_times

            example_str = ''.join([
                f'For example, with a scale factor of {scale_msu_travel_times}, '
                f'a journey that takes {time_not_msu:.0f} minutes ',
                f'in a normal ambulance would take {time_msu:.0f} minutes ',
                'in a Mobile Stroke Unit vehicle.'
                ])
            st.markdown(example_str)

        # Set a scale factor for how quickly the MSU can travel.
        scale_msu_travel_times = st.number_input(
            'Scale factor for MSU travel speed',
            min_value=1.0,
            max_value=5.0,
            # on_change=make_example_str
        )
        input_dict['scale_msu_travel_times'] = scale_msu_travel_times
        make_example_str()

        st.header('Stroke unit services')
        st.markdown('Update which services the stroke units provide:')
        # Display and store any changes from the user:
        df_unit_services = st.data_editor(
            df_unit_services,
            disabled=['postcode', 'stroke_team', 'isdn'],
            height=180  # limit height to show fewer rows
            )
        submitted = st.form_submit_button('Submit')

        if submitted:
            carry_on_please = True
        else:
            carry_on_please = False


# Restore dtypes:
df_unit_services[cols_use] = df_unit_services[cols_use].astype(int)

# Update the full data (for maps) with the changes:
cols_to_merge = cols_use  # + ['transfer_unit_postcode']
df_unit_services_full = df_unit_services_full.drop(cols_to_merge, axis='columns')
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

with container_map_inputs:
    cols = st.columns(2)  # make more columns than needed to space closer
with container_select_outcome:
    st.markdown('### Alternative outcome measure for map')
    st.markdown('Try these if you dare.')
scenario_dict = inputs.select_scenario([container_select_outcome] + cols)


# If the requested data is nLVO + MT, stop now.
stop_bool = (
    (scenario_dict['stroke_type'] == 'nlvo') &
    ('mt' in scenario_dict['treatment_type'])
)
if stop_bool:
    st.warning('No data for nLVO with MT.')
    st.stop()

scenario_types = ['drip_ship', 'diff_msu_minus_drip_ship']
# Draw a blank map in a container and then replace the contents with
# this intended map once it's finished being drawn
with container_map:
    maps.plotly_blank_maps(scenario_types, n_blank=2)

colour_dict = inputs.set_up_colours(scenario_dict | {'scenario_type': 'not diff'})
colour_diff_dict = inputs.set_up_colours(scenario_dict | {'scenario_type': 'diff'}, v_name='d')

with container_results_tables:
    gdf_boundaries_msoa = main_calculations(input_dict, df_unit_services)

# Find geometry column for plot function:
col_geo = utils.find_multiindex_column_names(
    gdf_boundaries_msoa, property=['geometry'])


columns_colours = [
    '_'.join([
        scenario_dict['stroke_type'],
        scenario_type,
        scenario_dict['treatment_type'],
        scenario_dict['outcome_type']
    ])
    for scenario_type in scenario_types
    ]
colour_dict['column'] = columns_colours[0]
colour_diff_dict['column'] = columns_colours[1]

maps.plotly_many_maps(
    gdf_boundaries_msoa,
    column_geometry=col_geo,
    colour_dicts=[colour_dict, colour_diff_dict],
    subplot_titles=scenario_types,
    legend_title=f'v: {scenario_dict["outcome_type_str"]};<br>d: Benefit of MSU over drip-and-ship',
    container_map=container_map,
    df_units=df_unit_services_full
)
