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
    page_title='MUSTER',
    page_icon=':ambulance:',
    layout='wide'
    )


def set_up_page_layout():
    """
    Set up container placement and return as dict.
    """
    c = {}
    st.title('Benefit in outcomes from Mobile Stroke Units')

    # ----- Setup -----
    c['setup'] = st.container(border=True)
    tab_titles = [
        'Stroke units', 'Treatment pathway', 'Population onion', 'Subgroups'
    ]
    with c['setup']:
        st.header('Setup')
        (c['units_setup'], c['pathway'], c['onion'], c['onion_subgroups']) = (
            st.tabs(tab_titles))

    with c['units_setup']:
        cols = st.columns([1, 2])
        with cols[0]:
            c['units_text'] = st.container()
        with cols[1]:
            c['units_map'] = st.container()
        with st.expander('Edit unit services'):
            c['units_df'] = st.container()

    with c['pathway']:
        c['pathway_text'] = st.container()
        c['pathway_fig'] = st.container()
        c['pathway_drop'] = st.expander('Edit pathway timings')
        with c['pathway_drop']:
            cols = st.columns(2)
            with cols[0]:
                st.markdown('Pre-hospital:')
                c['pathway_inputs_prehosp'] = st.container(
                    border=True, horizontal=True)
                st.markdown('MSU:')
                c['pathway_inputs_msu'] = st.container(
                    border=True, horizontal=True)
            with cols[1]:
                st.markdown('After arrival at stroke unit:')
                c['pathway_inputs_units'] = st.container(
                    border=True, horizontal=True)

    with c['onion']:
        c['onion_text'] = st.container()
        c['onion_setup'] = st.expander('Edit population proportions')

    with c['onion_subgroups']:
        c['pop_plots'] = st.container()

    # ----- Results -----
    c['run_results'] = st.container()
    c['results'] = st.container(border=True)
    tabs_results = ['Region summaries', 'England maps', 'Full results tables']
    with c['results']:
        st.header('Results')
        (c['region_summaries'], c['maps'], c['full_results']) = (
            st.tabs(tabs_results))

    with c['region_summaries']:
        c['region_select'] = st.container()
        c['highlighted_regions'] = st.container(horizontal=True)

    with c['maps']:
        c['map_fig'] = st.container()
        with st.expander('Accessibility & advanced map setup'):
            c['map_setup'] = st.container()
    with c['full_results']:
        c['full_results_setup'] = st.container()

    # ----- Log -----
    c['log'] = st.expander('Log of calculations', width=500)
    log_keys_labels = {
        'log_units': 'Stroke units',
        'log_pathway': 'Treatment pathway',
        'log_onion': 'Population onion',
        'log_subgroups': 'Subgroups',
        'log_regions': 'Region summaries',
        'log_maps': 'England maps',
        'log_full_results': 'Full results tables',
    }
    with c['log']:
        for key, label in log_keys_labels.items():
            c[key] = st.container()
            with c[key]:
                st.markdown(f':green[{label}]')

    return c


containers = set_up_page_layout()

#MARK: Page text
# #####################
# ##### PAGE TEXT #####
# #####################
with containers['units_text']:
    reg.plot_basic_travel_options_msu()
    st.markdown('''
In usual care:
+ patients whose nearest unit has MT always travel directly to an MT unit (:primary[red path]).
+ other patients travel first to the IVT unit and then if necessary are transferred to the MT unit (:grey[grey path]).

With a Mobile Stroke Unit, the ambulance travels to the patient (:primary[red path]).
The patient has a scan in the van, receives IVT if necessary, and is transported to the MT unit (:primary[red path]).

The MSU can mean faster access to IVT when the travel time is reduced.
It also means that patients who require MT can go directly to the MT unit instead of extra travel time and delays associated with the transfer.
''')
with containers['pathway_text']:
    st.markdown('''
The time to treatment depends on the travel times and whether the MSU was used.

Assumptions:  
1. The MSU is based at a stroke unit.
1. The MSU always transfers patients to an MT unit.
1. The MSU gives thrombolysis on scene and so spends longer on-scene than when IVT is not given.
2. All stroke units share the same time from arrival to delivery of IVT.
3. The time from arrival to delivery of MT can be different for patients admitted directly to the MT unit and for patients who received a transfer.
4. All other pathway timings are the same in every scenario.
''')
with containers['onion_text']:
    st.markdown('''
The population can have different make-ups of patients,
e.g. types of stroke and proportions treated.  
These proportions can be changed in the following table:
:red-background[(NOTE, November 2025, all values are placeholders)]
''')
with containers['pop_plots']:
    st.markdown('''
We calculate six base outcomes. These can be combined in different proportions to find the outcomes for a selected subgroup of patients.
''')

#MARK: Setup
# #################
# ##### SETUP #####
# #################

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
    df_unit_services = reg.select_unit_services(use_msu=True)
with containers['log_units']:  # for log_loc
    df_lsoa_units_times = reg.find_nearest_units_each_lsoa(
        df_unit_services, _log_loc=containers['log_units'])
# Load LSOA geometry:
df_raster, transform_dict = maps.load_lsoa_raster_lookup()
map_traces = plot_maps.make_constant_map_traces()
map_traces = (
    plot_maps.make_shared_map_traces(
        df_unit_services, df_lsoa_units_times, df_raster, transform_dict
    ) | map_traces
)
with containers['units_text']:
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
    plot_maps.draw_units_msu_map(map_traces, outline_name)
with containers['log_units']:  # for log_loc
    unique_travel_for_ivt, unique_travel_for_mt, dict_unique_travel_pairs = (
        reg.find_unique_travel_times(df_lsoa_units_times,
                                     _log_loc=containers['log_units'])
        )
    dict_region_admissions_unique_times = (
        reg.find_region_admissions_by_unique_travel_times(
            df_lsoa_units_times, _log_loc=containers['log_units'])
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

df_pathway_steps = pathway.select_pathway_timings(
    'muster', [containers['pathway_inputs_prehosp'],
               containers['pathway_inputs_units'],
               containers['pathway_inputs_msu']]
    )
series_treatment_times_without_travel = (
    pathway.calculate_treatment_times_without_travel(
        df_pathway_steps, ['usual_care', 'msu'],
        _log_loc=containers['log_pathway']
        )
    )
st.write(series_treatment_times_without_travel)
with containers['pathway_fig']:
    pathway.draw_timeline(df_pathway_steps,
                          series_treatment_times_without_travel,
                          use_msu=True)

unique_treatment_ivt, unique_treatment_mt = pathway.calculate_treatment_times(
    series_treatment_times_without_travel,
    unique_travel_for_ivt,
    unique_travel_for_mt,
    _log_loc=containers['log_pathway']
    )
unique_treatment_pairs = pathway.find_unique_treatment_time_pairs(
    dict_unique_travel_pairs, series_treatment_times_without_travel,
    _log=True, _log_loc=containers['log_pathway'],
)
# LSOA-level treatment times:
df_lsoa_units_times = pathway.calculate_treatment_times_each_lsoa_scenarios(
    df_lsoa_units_times,
    series_treatment_times_without_travel,
    _log_loc=containers['log_pathway']
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
with containers['log_pathway']:  # for log_loc
    dict_region_admissions_unique_treatment_times = (
        reg.find_region_admissions_by_unique_travel_times(
            df_lsoa_units_times, unique_travel=False,
            _log_loc=containers['log_pathway'])
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
    _log_loc=containers['log_pathway'])
dict_base_outcomes = outcomes.calculate_unique_outcomes(
    unique_treatment_ivt, unique_treatment_mt,
    _log_loc=containers['log_pathway'])
# Combine dicts:
dict_base_outcomes = dict_base_outcomes | dict_no_treatment_outcomes

df_base_lvo_ivt_mt_better = outcomes.flag_lvo_ivt_better_than_mt(
    dict_base_outcomes['lvo_ivt'],
    dict_base_outcomes['lvo_mt'],
    unique_treatment_pairs,
    _log_loc=containers['log_pathway']
    )
dict_base_outcomes['lvo_ivt_mt'] = outcomes.combine_lvo_ivt_mt_outcomes(
    dict_base_outcomes['lvo_ivt'],
    dict_base_outcomes['lvo_mt'],
    df_base_lvo_ivt_mt_better,
    _log_loc=containers['log_pathway']
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
    df_onion_pops = pop.set_up_onion_parameters()
with containers['onion_text']:
    dict_onion = pop.select_onion_population(df_onion_pops)
dict_onion = pop.calculate_population_subgroups(
    dict_onion, _log_loc=containers['log_onion'])


# ----- Subgroups (this onion layer) -----
with containers['pop_plots']:
    df_subgroups = pop.select_subgroups_for_results()

df_pop_usual_care, df_pop_redir, df_pop_redir_accepted_only = (
    pop.calculate_population_subgroup_grid(
        dict_onion, df_subgroups, _log_loc=containers['log_subgroups']
        ))

with containers['pop_plots']:
    n_cols = 2
    cols = st.columns(n_cols)
    for i, s in enumerate(df_subgroups.index):
        with cols[i % n_cols]:
            c = st.container(border=True)
        with c:
            pop.plot_population_props(
                df_pop_usual_care[['scenario'] + [s]],
                df_pop_redir[['scenario'] + [s]],
                s,
                df_subgroups.loc[s]
                )


# ----- Outcomes -----
dict_outcomes = {}
for s in df_subgroups.index:
    dict_outcomes[s] = pop.calculate_unique_outcomes_onion(
        dict_base_outcomes,
        df_pop_usual_care,
        df_pop_redir,
        df_pop_redir_accepted_only,
        df_subgroups.loc[s],
        df_treat_times_sets_unique,
        s,
        _log_loc=containers['log_subgroups']
    )

with containers['run_results']:
    run_results = st.button('Calculate results - currently this button does nothing', type='primary')
if run_results:
    pass
else:
    # TO DO - make it so that the results don't recalculate entirely
    # while you're still setting up. Also perhaps cache the old results
    # and show them while setting up?
    pass


#MARK: Results
# ###################
# ##### RESULTS #####
# ###################
# ----- Region summaries -----
with containers['region_select']:
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
    _log_loc=containers['log_regions']
    )
dict_highlighted_region_travel_times = (
    reg.gather_travel_times_highlighted_regions(
        dict_region_admissions_unique_times,
        highlighted_region_types,
        df_highlighted_regions,
        _log_loc=containers['log_regions']
    )
)
df_highlighted_region_admissions = reg.gather_admissions_highlighted_regions(
    dict_region_admissions_unique_times,
    highlighted_region_types,
    df_highlighted_regions,
    _log_loc=containers['log_regions']
)

# Display chosen results:
with containers['region_select']:
    use_lsoa_subset = st.toggle(
        'Use only patients whose nearest unit does not provide MT.',
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
        ch = st.container(border=True, width=500)
    with ch:
        st.subheader(region_label)
        cols = st.columns(2)
        with cols[0]:
            # Travel times
            time_bins, admissions_times = reg.gather_this_region_travel_times(
                dict_highlighted_region_travel_times, lsoa_subset, region)
            reg.plot_travel_times(time_bins, admissions_times)
        with cols[1]:
            # Admissions
            s_admissions = df_highlighted_region_admissions[region]
            st.metric('Annual stroke admissions',
                      f"{s_admissions['admissions_all']:.1f}")
            n = 'Proportion of patients whose  \nnearest unit offers MT'
            p = 100.0*(1.0 - s_admissions['prop_nearest_unit_no_mt'])
            st.metric(n, f"{p:.1f}%")
        # Outcomes:
        for s, subgroup in enumerate(df_subgroups.index):
            df_u = dict_highlighted_region_outcomes[subgroup][
                'usual_care'][lsoa_subset].loc[region]
            df_r = dict_highlighted_region_outcomes[subgroup][
                'redir_allowed'][lsoa_subset].loc[region]
            cs = st.expander(df_subgroups.loc[subgroup, 'label'],
                             expanded=(True if s == 0 else False))
            with cs:
                if df_u.isna().all() & df_r.isna().all():
                    st.markdown('No data available.')
                else:
                    cc = st.container(horizontal=True)
                    with cc:
                        for key in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
                            with st.container():
                                reg.display_region_summary(df_u, df_r, key)

                    # TO DO - generate "no treatment" data for
                    # this combination of patients, draw bars.

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
                    reg.plot_mrs_bars(mrs_lists_dict,
                                      key='_'.join([region, subgroup]))

# ----- Maps -----
# For the selected data type to show on the maps, gather the full
# LSOA-level data. Then reshape into the raster array.
with containers['map_setup']:
    map_outcome = outcomes.select_outcome_type()
# Gather data for maps:
with containers['map_fig']:
    subgroup_map, subgroup_map_label = maps.select_map_data(df_subgroups)
    use_full_redir = st.toggle(
        '''In middle map, include "reject redirection" and
        "usual care" patients.''',
        value=False,
        key='full_redir_subset'
        )
    redir_subset = ('redir_allowed' if use_full_redir
                    else 'redir_accepted_only')
map_arrs_dict, vlim_dict = maps.gather_map_arrays(
    dict_outcomes[subgroup_map]['usual_care'],
    dict_outcomes[subgroup_map][redir_subset],
    df_lsoa_units_times,
    df_raster,
    transform_dict,
    col_map=map_outcome,
    _log_loc=containers['log_maps']
    )
map_arrs_dict['pop'], vlim_dict_pop = maps.gather_pop_map(
    df_raster, transform_dict)
vlim_dict = vlim_dict | vlim_dict_pop

# Set up colour limits:
with containers['map_setup']:
    dicts_colours = colour_setup.select_colour_limits(map_outcome, vlim_dict)

# Make colour maps and traces:
with containers['map_setup']:
    default_cmap_name, all_cmaps = colour_setup.select_colour_maps()
for p, dp in dicts_colours.items():
    dicts_colours[p]['cmap'] = colour_setup.make_colour_list(
        default_cmap_name, vmin=dp['vmin'], vmax=dp['vmax'])
for col, arr in map_arrs_dict.items():
    map_traces[col] = plot_maps.make_trace_heatmap(
        arr, transform_dict, dicts_colours[col], name=col)

with containers['map_fig']:
    plot_maps.plot_outcome_maps(
        map_traces,
        ['usual_care', 'redir_minus_usual_care', 'pop'],
        dicts_colours,
        all_cmaps,
        outline_name,
        )


# ----- Full LSOA results -----
# Generate on request, not by default with each re-run.
with containers['full_results_setup']:
    generate_full_data = st.checkbox('Show options to generate full data')
if generate_full_data:
    with containers['full_results_setup']:
        full_results_type = reg.select_full_data_type()
    if full_results_type == 'lsoa':
        # Calculate LSOA-level results.
        dict_full_outcomes = pop.gather_lsoa_level_outcomes(
            dict_outcomes,
            df_lsoa_units_times,
            _log_loc=containers['log_full_results']
            )
    else:
        # Calculate the full outcomes for just this selected region type
        # but for all the nested subsets (subgroup, scenario, LSOA subset):
        dict_full_outcomes = reg.calculate_nested_average_outcomes(
            dict_outcomes,
            dict_region_admissions_unique_treatment_times,
            [full_results_type],
            _log_loc=containers['log_full_results']
            )
    with containers['full_results_setup']:
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
                'Use only patients whose nearest unit does not provide MT.',
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
