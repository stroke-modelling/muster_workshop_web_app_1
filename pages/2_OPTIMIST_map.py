"""
Outcomes with usual care and redirection with maps for OPTIMIST.

This version calculates as few elements as possible. Instead of
calculating full data for thousands of LSOA, we instead process the
unique treatment times that may be shared across many LSOA.
The full LSOA results are only built up at the end when required.
"""
#MARK: Imports
# ###################
# ##### IMPORTS #####
# ###################
import streamlit as st
import pandas as pd

# ----- Custom functions -----
import utilities.regions as reg
import utilities.maps as maps
import utilities.plot_maps as plot_maps
import utilities.pathway as pathway
import utilities.outcomes as outcomes
import utilities.population as pop
import utilities.colour_setup as colour_setup


#MARK: Functions
# #####################
# ##### FUNCTIONS #####
# #####################


# ################
# ##### MAIN #####
# ################

#MARK: Page layout
# #######################
# ##### PAGE LAYOUT #####
# #######################
st.set_page_config(
    page_title='OPTIMIST',
    page_icon=':ambulance:',
    layout='wide'
    )


def set_up_page_layout():
    """
    Set up container placement and return as dict.
    """
    c = {}
    st.title('Benefit in outcomes from redirection')
    c['units_setup'] = st.container()
    c['units_map'] = st.container()
    with st.expander('Edit unit services'):
        c['units_df'] = st.container()

    c['pathway'] = st.container()
    c['pathway_inputs'] = st.container(horizontal=True)
    c['pathway_summary'] = st.container()
    c['onion_setup'] = st.container()
    c['pop_plots'] = st.container(horizontal=True)

    # ----- Results -----
    c['region_summaries'] = st.container()
    c['highlighted_regions'] = st.container(horizontal=True)
    c['maps'] = st.container()
    with st.sidebar:
        c['map_setup'] = st.container()
    # with st.expander('Full data tables'):
    c['full_results'] = st.container()
    return c


containers = set_up_page_layout()

#MARK: Page text
# #####################
# ##### PAGE TEXT #####
# #####################
with containers['units_setup']:
    st.header('Stroke units')
with containers['pathway']:
    st.header('Pathway')
with containers['onion_setup']:
    st.header('Population')
with containers['region_summaries']:
    st.header('Region summaries')
with containers['maps']:
    st.header('England maps')
with containers['full_results']:
    st.header('Full results tables')

#MARK: Setup
# #################
# ##### SETUP #####
# #################

# ----- Outcome choice -----
with containers['map_setup']:
    map_outcome = outcomes.select_outcome_type()
    cmap_name = colour_setup.select_colour_maps()
    dicts_colours = colour_setup.select_colour_limits(map_outcome, cmap_name)


# ----- Unit services -----
# Show the map of England with the stroke units and so introduce
# the idea that some geographical areas are nearest a CSC (MT)
# and others would need a transfer to the MT unit.
# Allow unit services to be updated in a data editor table.
# Show the areas whose nearest unit is not a CSC.
# Group the data over each geographical region. Regions
# include ICBs, ISDNs, ambulance services, nearest unit.
# Find two copies of the results - one with all LSOA in the region
# and one with only LSOA whose nearest unit is not a CSC.
# --- CALCULATIONS:
# + Calculate the travel times for all LSOA. Flag which LSOA
#   are nearest a CSC.
# + Gather the unique travel times.
# + Convert list of all LSOA in a region to a list of number of
#   admissions per unique treatment time.

with containers['units_df']:
    df_unit_services = reg.select_unit_services()
with containers['units_setup']:  # for log_loc
    df_lsoa_units_times = reg.find_nearest_units_each_lsoa(
        df_unit_services, _log_loc=containers['units_setup'])
# Load LSOA geometry:
df_raster, transform_dict = maps.load_lsoa_raster_lookup()
map_traces = plot_maps.make_constant_map_traces(
    df_raster, transform_dict, dicts_colours['pop'])
map_traces = (
    plot_maps.make_shared_map_traces(
        df_unit_services, df_lsoa_units_times, df_raster, transform_dict
    ) | map_traces
)
with containers['units_map']:
    outline_labels_dict = {
        'none': 'None',
        'icb': 'Integrated Care Board',
        'isdn': 'Integrated Stroke Delivery Network',
        'ambo22': 'Ambulance service',
    }

    def f(label):
        """Display layer with nice name instead of key."""
        return outline_labels_dict[label]
    outline_name = st.radio(
        'Region type to draw on maps',
        outline_labels_dict.keys(),
        format_func=f,
        horizontal=True
        )
with containers['units_map']:
    plot_maps.draw_units_map(map_traces, outline_name)
with containers['units_setup']:  # for log_loc
    unique_travel_for_ivt, unique_travel_for_mt, dict_unique_travel_pairs = (
        reg.find_unique_travel_times(df_lsoa_units_times,
                                     _log_loc=containers['units_setup'])
        )
    dict_region_admissions_unique_times = (
        reg.find_region_admissions_by_unique_travel_times(
            df_lsoa_units_times, _log_loc=containers['units_setup'])
        )
# Note: logs print in wrong location for cached functions,
# so have extra "with" blocks in the lines above.


# ----- Pathway timings -----
# Show the summary of pathway timings for each case: usual care;
# redirection approved; redirection rejected. Show a timeline
# image with the timings for the separate steps made clear.
# Allow the timings to be changed with a series of widgets.
# --- CALCULATIONS:
# + Add up treatment times without travel for IVT and MT
#   in each scenario.
# + Find unique treatment times and pairs of treatment times.

with containers['pathway_inputs']:
    df_pathway_steps = pathway.select_pathway_timings('optimist')
series_treatment_times_without_travel = (
    pathway.calculate_treatment_times_without_travel(
        df_pathway_steps, ['usual_care', 'prehospdiag'],
        _log_loc=containers['pathway']
        )
    )
with containers['pathway_summary']:
    pathway.draw_timeline(df_pathway_steps)
    pathway.show_treatment_time_summary(
        series_treatment_times_without_travel)
unique_treatment_ivt, unique_treatment_mt = pathway.calculate_treatment_times(
    series_treatment_times_without_travel,
    unique_travel_for_ivt,
    unique_travel_for_mt,
    _log_loc=containers['pathway']
    )
unique_treatment_pairs = pathway.find_unique_treatment_time_pairs(
    dict_unique_travel_pairs, series_treatment_times_without_travel,
    _log=True, _log_loc=containers['pathway'],
)
# LSOA-level treatment times:
df_lsoa_units_times = pathway.calculate_treatment_times_each_lsoa_scenarios(
    df_lsoa_units_times,
    series_treatment_times_without_travel,
    _log_loc=containers['pathway']
    )
# Find the unique sets of treatment times:
scens = ['usual_care', 'redirection_approved', 'redirection_rejected']
treats = ['ivt', 'mt']
cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
df_treat_times_sets_unique = (
    df_lsoa_units_times[cols_treat_scen].drop_duplicates())
# Update index to normal range:
df_treat_times_sets_unique['index'] = range(
    len(df_treat_times_sets_unique))
df_treat_times_sets_unique = (
    df_treat_times_sets_unique.set_index('index'))
# Find how many admissions per region have each set of
# unique treatment times:
with containers['pathway']:  # for log_loc
    dict_region_admissions_unique_treatment_times = (
        reg.find_region_admissions_by_unique_travel_times(
            df_lsoa_units_times, unique_travel=False,
            _log_loc=containers['pathway'])
        )


# ----- Base outcomes -----
# Calculate base outcomes for the given travel times and scenarios.
# Find outcomes for all of the unique treatment times given.
# --- CALCULATIONS:
# + Calculate outcomes for unique treatment times for the base
#   groups: nLVO + IVT, LVO + IVT, LVO + MT.
# + For unique pairs of times to treatment, find when LVO + IVT
#   is better than LVO + MT.
dict_no_treatment_outcomes = outcomes.load_no_treatment_outcomes(
    _log_loc=containers['pathway'])
dict_base_outcomes = outcomes.calculate_unique_outcomes(
    unique_treatment_ivt, unique_treatment_mt,
    _log_loc=containers['pathway'])
# Combine dicts:
dict_base_outcomes = dict_base_outcomes | dict_no_treatment_outcomes

df_base_lvo_ivt_mt_better = outcomes.flag_lvo_ivt_better_than_mt(
    dict_base_outcomes['lvo_ivt'],
    dict_base_outcomes['lvo_mt'],
    unique_treatment_pairs,
    _log_loc=containers['pathway']
    )
dict_base_outcomes['lvo_ivt_mt'] = outcomes.combine_lvo_ivt_mt_outcomes(
    dict_base_outcomes['lvo_ivt'],
    dict_base_outcomes['lvo_mt'],
    df_base_lvo_ivt_mt_better,
    _log_loc=containers['pathway']
    )

# ----- Patient population (onion layer) -----
# Decide the patient population parameters. There are different
# subgroups of patients in each layer of the SPEEDY onion graph.
# The layer determines the proportion of patients with each stroke
# type and the proportions of patients who will be redirected.
# Inputs:
# + nLVO / LVO proportions in this subgroup,
# + full population proportions considered for redirection,
# + sensitivity / specificity of redirection diagnostic.
# --- CALCULATIONS:
# + Unique time results for nLVO + LVO combo for usual care
#   and for "redirection considered" groups.

with containers['onion_setup']:
    dict_onion = pop.select_onion_population()
dict_onion = pop.calculate_population_subgroups(dict_onion)

with containers['onion_setup']:
    df_subgroups = pop.select_subgroups_for_results()

df_pop_usual_care, df_pop_redir = (
    pop.calculate_population_subgroup_grid(dict_onion, df_subgroups))

with containers['pop_plots']:
    for s in df_subgroups.index:
        pop.plot_population_props(
            df_pop_usual_care[['scenario'] + [s]],
            df_pop_redir[['scenario'] + [s]],
            s,
            df_subgroups.loc[s]
            )

dict_outcomes = {}
for s in df_subgroups.index:
    dict_outcomes[s] = pop.calculate_unique_outcomes_onion(
        dict_base_outcomes,
        df_pop_usual_care,
        df_pop_redir,
        df_subgroups.loc[s],
        df_treat_times_sets_unique,
        s,
        _log_loc=containers['onion_setup']
    )


#MARK: Results
# ###################
# ##### RESULTS #####
# ###################
# ----- Region summaries -----
with containers['region_summaries']:
    df_highlighted_regions = reg.select_highlighted_regions(df_unit_services)
# Only find the region results for highlighted region types:
highlighted_region_types = sorted(list(set(
    df_highlighted_regions['region_type'])))
# Average the results over each geographical region.
# Find two copies of the results - one with all LSOA in the region
# and one with only LSOA whose nearest unit is not a CSC.
# --- CALCULATIONS:
# + Calculate admissions-weighted average outcomes.

# Nest levels: subgroup, scenario, lsoa subset.
dict_highlighted_region_outcomes = reg.calculate_nested_average_outcomes(
    dict_outcomes,
    dict_region_admissions_unique_treatment_times,
    highlighted_region_types,
    df_highlighted_regions,
    _log_loc=containers['region_summaries']
    )

# Display chosen results:
with containers['region_summaries']:
    use_lsoa_subset = st.toggle(
        'Use only patients whose nearest unit does not provide MT',
        value=True,
        )
lsoa_subset = 'nearest_unit_no_mt' if use_lsoa_subset else 'all_patients'

cols_mrs = [f'mrs_dists_{i}' for i in range(7)]
cols_mrs_noncum = [c.replace('dists_', 'dists_noncum_') for c in cols_mrs]
cols_mrs_std = [f'{c}_std' for c in cols_mrs]

for r, region in enumerate(df_highlighted_regions['highlighted_region']):
    region_type = df_highlighted_regions.loc[
        df_highlighted_regions['highlighted_region'] == region,
        'region_type'].values[0]
    # Pick out label for the box:
    if region_type == 'nearest_ivt_unit':
        region_label = df_unit_services.loc[region, 'ssnap_name']
    else:
        region_label = region
    with containers['highlighted_regions']:
        ch = st.container(border=True)
    with ch:
        st.header(region_label)
        for subgroup in df_subgroups.index:
            df_u = dict_highlighted_region_outcomes[subgroup][
                'usual_care'][lsoa_subset].loc[region]
            df_r = dict_highlighted_region_outcomes[subgroup][
                'redir_allowed'][lsoa_subset].loc[region]
            cs = st.container(border=True)
            with cs:
                if df_u.isna().all() & df_r.isna().all():
                    st.markdown('No data available.')
                else:
                    st.subheader(df_subgroups.loc[subgroup, 'label'])
                    reg.display_region_summary(df_u, df_r)
                    mrs_lists_dict = {
                        'usual_care': {
                            'noncum': df_u[cols_mrs_noncum],
                            'cum': df_u[cols_mrs],
                            'std': df_u[cols_mrs_std],
                            'colour': '#0072b2',
                            'linestyle': 'solid',
                            'label': 'Usual care',
                        },
                        'redir_allowed': {
                            'noncum': df_r[cols_mrs_noncum],
                            'cum': df_r[cols_mrs],
                            'std': df_r[cols_mrs_std],
                            'colour': '#56b4e9',
                            'linestyle': 'dash',
                            'label': 'Redirection available'
                        },
                    }
                    reg.plot_mrs_bars(mrs_lists_dict)

# ----- Maps -----
# For the selected data type to show on the maps, gather the full
# LSOA-level data.
with containers['maps']:
    subgroup_map, subgroup_map_label = maps.select_map_data(df_subgroups)

map_arrs_dict = maps.gather_map_arrays(
    dict_outcomes[subgroup_map]['usual_care'],
    dict_outcomes[subgroup_map]['redir_allowed'],
    df_lsoa_units_times,
    df_raster,
    transform_dict,
    col_map=map_outcome,
    _log_loc=containers['maps']
    )
for col, arr in map_arrs_dict.items():
    map_traces[col] = plot_maps.make_trace_heatmap(
        arr, transform_dict, dicts_colours[col], name=col)


with containers['maps']:
    plot_maps.plot_outcome_maps(
        map_traces,
        ['usual_care', 'redir_minus_usual_care', 'pop'],
        dicts_colours,
        )


# ----- Full LSOA results -----
# Generate on request, not by default with each re-run.
with containers['full_results']:
    generate_full_data = st.checkbox('Show options to generate full data')
if generate_full_data:
    with containers['full_results']:
        full_results_type = reg.select_full_data_type()
    if full_results_type == 'lsoa':
        # Calculate LSOA-level results.
        dict_full_outcomes = pop.gather_lsoa_level_outcomes(
            dict_outcomes,
            df_lsoa_units_times,
            _log_loc=containers['full_results']
            )
    else:
        # Calculate the full outcomes for just this selected region type
        # but for all the nested subsets (subgroup, scenario, LSOA subset):
        dict_full_outcomes = reg.calculate_nested_average_outcomes(
            dict_outcomes,
            dict_region_admissions_unique_treatment_times,
            [full_results_type],
            _log_loc=containers['full_results']
            )
    with containers['full_results']:
        if full_results_type == 'lsoa':
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown('Unit postcode lookup')
                st.dataframe(df_unit_services['ssnap_name'].sort_index())
            with cols[1]:
                st.markdown('Travel and treatment times:')
                st.dataframe(df_lsoa_units_times.set_index('LSOA'))
            for subgroup, df_full in dict_full_outcomes.items():
                st.subheader(df_subgroups.loc[subgroup, 'label'])
                st.dataframe(df_full)
        else:
            use_lsoa_subset_full = st.toggle(
                'Use only patients whose nearest unit does not provide MT',
                value=True,
                key='full_lsoa_subset'
                )
            lsoa_subset_full = ('nearest_unit_no_mt' if use_lsoa_subset_full
                                else 'all_patients')
            for subgroup, dict_full in dict_full_outcomes.items():
                dfs = []
                for scen in ['usual_care', 'redir_allowed']:
                    df = dict_full[scen][lsoa_subset_full][full_results_type]
                    df = df.rename(columns=dict(
                        [(c, f'{c}_{scen}') for c in df.columns]))
                    dfs.append(df)
                df_full = pd.concat(dfs, axis='columns')

                if full_results_type == 'nearest_ivt_unit':
                    # Change postcodes to unit names:
                    df_full.index = df_full.index.map(
                        df_unit_services['ssnap_name'])
                    df_full = df_full.sort_index()

                st.subheader(df_subgroups.loc[subgroup, 'label'])
                st.dataframe(df_full)
