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

from utilities.utils import set_rerun_map, set_rerun_full_results

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

try:
    page_last_run = st.session_state['page_last_run']
    if page_last_run != 'MUSTER':
        # Clear the OPTIMIST results.
        keys_to_del = list(st.session_state.keys())
        for key in keys_to_del:
            del st.session_state[key]
except KeyError:
    # No page has been run yet.
    pass
st.session_state['page_last_run'] = 'MUSTER'
# Set these so that all results run on first go of script:
if 'inputs_changed' not in st.session_state.keys():
    st.session_state['inputs_changed'] = True
    st.session_state['rerun_region_summaries'] = True
    st.session_state['rerun_maps'] = True
    st.session_state['rerun_full_results'] = True


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
            c['units_buttons'] = st.container(horizontal=True)
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
        c['highlighted_region_summaries'] = st.container(horizontal=True)
        c_dict = {
            'region_treat_stats': 'Treatment numbers',
            'region_times': 'Travel and treatment times',
            # 'region_unit_admissions': 'Admissions by unit',
            # 'region_unit_maps': 'Patient transport maps',
        }
        for container_name, container_label in c_dict.items():
            c[f'{container_name}_top'] = st.expander(
                container_label, expanded=False)
            with c[f'{container_name}_top']:
                h = True if container_name != 'region_unit_maps' else False
                c[container_name] = st.container(horizontal=h)

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

df_unit_services = reg.select_unit_services_muster(
    use_msu=True,
    container_buttons=containers['units_buttons'],
    container_dataeditor=containers['units_df'],
    )
with containers['units_df']:
    if all(df_unit_services['Use_MSU'] == 0):
        st.error('There must be at least one MSU.', icon='❗')
        st.stop()
with containers['log_units']:  # for log_loc
    df_lsoa_units_times = reg.find_nearest_units_each_lsoa(
        df_unit_services, use_msu=True, _log_loc=containers['log_units'])
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


# Need the pathway inputs now because
# the ambulance response time is needed
# to calculate unique travel times.
df_pathway_steps = pathway.select_pathway_timings(
    'muster', [containers['pathway_inputs_prehosp'],
               containers['pathway_inputs_units'],
               containers['pathway_inputs_msu']]
    )
df_lsoa_units_times = reg.calculate_extra_muster_travel_times(
    df_lsoa_units_times, df_pathway_steps
)

with containers['log_units']:  # for log_loc
    unique_travel_for_ivt, unique_travel_for_mt, dict_unique_travel_pairs = (
        reg.find_unique_travel_times(
            df_lsoa_units_times,
            cols_ivt=[
                'msu_response_time',                    # MSU
                'ambo_response_then_nearest_ivt_time',  # usual care
                ],
            cols_mt=[
                'msu_response_then_mt_time',                    # MSU
                'ambo_response_then_nearest_ivt_then_mt_time',  # usual care
                ],
            cols_pairs={
                'usual_care': (
                    'ambo_response_then_nearest_ivt_time',
                    'ambo_response_then_nearest_ivt_then_mt_time'
                    ),
                'msu': (
                    'msu_response_time',
                    'msu_response_then_mt_time'
                    ),
            },
            cols_pairs_labels=['travel_for_ivt', 'travel_for_mt'],
            _log_loc=containers['log_units'])
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

series_treatment_times_without_travel = (
    pathway.calculate_treatment_times_without_travel(
        df_pathway_steps, ['usual_care', 'msu'],
        _log_loc=containers['log_pathway']
        )
    )
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
df_lsoa_units_times = (
    pathway.calculate_treatment_times_each_lsoa_scenarios_muster(
        df_lsoa_units_times,
        series_treatment_times_without_travel,
        _log_loc=containers['log_pathway']
        )
)
# Find the unique sets of treatment times:
scens = ['usual_care', 'msu_ivt', 'msu_no_ivt']
treats = ['ivt', 'mt']
cols_treat_scen = [f'{s}_{t}' for s in scens for t in treats]
cols_treat_scen = [c for c in cols_treat_scen
                   if c in df_lsoa_units_times.columns]
df_treat_times_sets_unique = (
    df_lsoa_units_times[cols_treat_scen].drop_duplicates())
# Update index to normal range:
df_treat_times_sets_unique['index'] = range(
    len(df_treat_times_sets_unique))
df_treat_times_sets_unique = (
    df_treat_times_sets_unique.set_index('index'))

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
    df_onion_pops = pop.set_up_onion_parameters('muster', use_debug=False)
dict_onion = df_onion_pops.loc[
    df_onion_pops['population'] == 'muster'].squeeze()
    # df_onion_pops['population'] == 'debug'].squeeze()

# ----- Subgroups (this onion layer) -----
with containers['pop_plots']:
    df_subgroups = pop.select_subgroups_for_results()

dict_pops = (
    pop.calculate_population_subgroup_grid_muster(
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
                dict_pops['usual_care'][['scenario'] + [s]],
                dict_pops['msu'][['scenario'] + [s]],
                s,
                df_subgroups.loc[s],
                titles=['<b>Usual care</b>', '<b>MSU available</b>']
                )

# ----- Outcomes -----
# Only recalculate results if anything above here has changed.
# Don't rerun outcomes when selecting options in the Results section.
with containers['run_results']:
    rerun_results = st.button('Recalculate results', type='primary')

# Calculate results if this is the first go through the app
# or the button has been pressed.
if ('dict_outcomes' not in st.session_state.keys()) or rerun_results:
    # Re-run results.
    st.session_state['dict_outcomes'] = {}
    for s in df_subgroups.index:
        st.session_state['dict_outcomes'][s] = (
            pop.calculate_unique_outcomes_onion(
                dict_base_outcomes,
                {'usual_care': dict_pops['usual_care'],
                 'msu': dict_pops['msu']},
                df_subgroups.loc[s],
                df_treat_times_sets_unique,
                s,
                _log_loc=containers['log_subgroups']
            )
        )
    st.session_state['df_lsoa_units_times'] = df_lsoa_units_times
    st.session_state['df_subgroups'] = df_subgroups
    st.session_state['dict_pops'] = dict_pops
    st.session_state['inputs_changed'] = False
    st.session_state['rerun_region_summaries'] = True
    st.session_state['rerun_maps'] = True
    st.session_state['rerun_full_results'] = True
elif st.session_state['inputs_changed']:
    with containers['run_results']:
        st.warning('Results are for previous set of inputs.', icon='⚠️')
else:
    # Don't re-run results.
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

if st.session_state['rerun_region_summaries']:
    if len(highlighted_region_types) == 0:
        # Placeholder empty dfs:
        st.session_state['dict_highlighted_region_travel_times'] = {}
        st.session_state['dict_highlighted_region_unique_treatment_times'] = {}
        st.session_state['df_highlighted_region_admissions'] = pd.DataFrame()
        st.session_state['df_region_unit_admissions'] = pd.DataFrame()
        st.session_state['dict_highlighted_region_outcomes'] = {}
        st.session_state['dict_highlighted_region_average_treatment_times'] = {}
    else:
        st.session_state['dict_highlighted_region_travel_times'] = (
            reg.find_region_admissions_by_unique_travel_times(
                df_lsoa_units_times, 
                highlighted_region_types,
                df_highlighted_regions,
                project='muster',
                _log_loc=containers['log_regions'])
            )
        # Find how many admissions per region have each set of
        # unique treatment times:
        st.session_state['dict_highlighted_region_unique_treatment_times'] = (
            reg.find_region_admissions_by_unique_travel_times(
                df_lsoa_units_times,
                highlighted_region_types,
                df_highlighted_regions,
                unique_travel=False,
                project='muster',
                _log_loc=containers['log_regions'])
            )

        (
            st.session_state['df_highlighted_region_admissions'],
            st.session_state['df_region_unit_admissions']
            ) = (
            reg.find_unit_admissions_by_region(
                df_lsoa_units_times,
                highlighted_region_types,
                df_highlighted_regions,
                _log_loc=containers['log_regions'],
                )
        )

        # Nest levels: subgroup, scenario, lsoa subset.
        st.session_state['dict_highlighted_region_outcomes'] = (
            reg.calculate_nested_average_outcomes(
                st.session_state['dict_outcomes'],
                st.session_state['dict_highlighted_region_unique_treatment_times'],
                df_highlighted_regions,
                _log_loc=containers['log_regions']
                )
        )
        st.session_state['dict_highlighted_region_average_treatment_times'] = (
            reg.calculate_average_treatment_times_highlighted_regions(
                st.session_state['dict_highlighted_region_unique_treatment_times'],
                _log_loc=containers['log_regions']
                )
        )
        st.session_state['rerun_region_summaries'] = False
else:
    pass

# Display chosen results:
with containers['region_select']:
    use_lsoa_subset = st.toggle(
        'Use only patients whose nearest unit does not provide MT.',
        value=True,
        )
lsoa_subset = 'nearest_unit_no_mt' if use_lsoa_subset else 'all_patients'

# Set up containers for the outcome subgroups:
with containers['region_summaries']:
    for s, subgroup in enumerate(st.session_state['df_subgroups'].index):
        containers[f'{subgroup}_top'] = st.expander(
            st.session_state['df_subgroups'].loc[subgroup, 'label'],
            expanded=(True if s == 0 else False)
            )
        with containers[f'{subgroup}_top']:
            containers[subgroup] = st.container(horizontal=True)

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
    with containers['highlighted_region_summaries']:
        ch = st.container(border=True, width=500)
    with ch:
        st.subheader(region_label)
        st.write('summary summary summary bits')

    with containers['region_treat_stats']:
        c = st.container(width=400, border=True)
        with c:
            st.subheader(region_label)
            # Admissions
            s_admissions = (
                st.session_state['df_highlighted_region_admissions']
                .loc[region]
                )
            st.metric('Annual stroke admissions',
                      f"{s_admissions['admissions_all_patients']:.1f}")
            n = 'Proportion of patients whose  \nnearest unit offers MT'
            p = 100.0*(1.0 - s_admissions['prop_nearest_unit_no_mt'])
            st.metric(n, f"{p:.1f}%")

    with containers['region_times']:
        c = st.container(width=400, border=True)
        with c:
            st.subheader(region_label)
            # Travel times
            time_cols = ['usual_care_ivt', 'usual_care_mt',
                         'msu_ivt_ivt']
            time_bins, admissions_times = reg.gather_this_region_travel_times(
                st.session_state['dict_highlighted_region_travel_times'],
                lsoa_subset, region, time_cols)
            subplot_titles = ['To nearest unit', 'To nearest then MT unit',
                              'From MSU base']
            reg.plot_travel_times(time_bins, admissions_times,
                                  subplot_titles)

            # Average treatment times:
            s_treats = st.session_state[
                'dict_highlighted_region_average_treatment_times'][
                    lsoa_subset].loc[region]
            df_treats = reg.make_average_treatment_time_df(s_treats)
            st.markdown(r'Mean treatment times ($\pm$ 1 standard deviation):')
            st.table(df_treats)

    # Outcomes:
    for s, subgroup in enumerate(st.session_state['df_subgroups'].index):
        # Calculate "no treatment" data.
        # Should have the same total proportions of nLVO
        # and LVO in both the usual care and redir groups,
        # with only the details of who goes where differing,
        # so only calculate one set of no-treatment mRS dists.
        pops = (
            st.session_state['dict_pops']['usual_care'][subgroup])
        df_no_treat = reg.calculate_no_treatment_mrs(
            pops, dict_no_treatment_outcomes)

        with containers[subgroup]:
            c = st.container(width=500, border=True)
        with c:
            st.subheader(region_label)
            df_u = st.session_state['dict_highlighted_region_outcomes'][
                subgroup]['usual_care'][lsoa_subset].loc[region]
            df_r = st.session_state['dict_highlighted_region_outcomes'][
                subgroup]['msu'][lsoa_subset].loc[region]

            if df_u.isna().all() & df_r.isna().all():
                st.markdown('No data available.')
            else:
                cc = st.container(horizontal=True)
                with cc:
                    for key in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
                        with st.container():
                            reg.display_region_summary(df_u, df_r, key)

                mrs_lists_dict = {
                    'usual_care': {
                        'noncum': df_u[cols_mrs_noncum],
                        'cum': df_u[cols_mrs],
                        'std': df_u[cols_mrs_std],
                        'colour': '#0072b2',
                        'linestyle': 'solid',
                        'label': 'Usual care',
                    },
                    'msu': {
                        'noncum': df_r[cols_mrs_noncum],
                        'cum': df_r[cols_mrs],
                        'std': df_r[cols_mrs_std],
                        'colour': '#56b4e9',
                        'linestyle': 'dash',
                        'label': 'MSU'
                    },
                    'no_treatment': {
                        'noncum': df_no_treat[cols_mrs_noncum],
                        'cum': df_no_treat[cols_mrs],
                        'colour': 'grey',
                        'label': 'No treatment'
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
    subgroup_map, subgroup_map_label = maps.select_map_data(
        st.session_state['df_subgroups']
    )
    # use_full_redir = st.toggle(
    #     '''In middle map, include "reject redirection" and
    #     "usual care" patients.''',
    #     value=False,
    #     key='full_redir_subset'
    #     )
    # redir_subset = ('redir_allowed' if use_full_redir
    #                 else 'redir_accepted_only')


if st.session_state['rerun_maps']:
    st.session_state['map_arrs_dict'], st.session_state['vlim_dict'] = (
        maps.gather_map_arrays(
            st.session_state['dict_outcomes'][subgroup_map]['usual_care'],
            st.session_state['dict_outcomes'][subgroup_map]['msu'],
            df_lsoa_units_times,
            df_raster,
            transform_dict,
            col_map=map_outcome,
            map_labels=['usual_care', 'msu'],
            scenarios=['usual_care', 'msu_ivt', 'msu_no_ivt'],
            _log_loc=containers['log_maps']
            )
    )
    st.session_state['map_arrs_dict']['pop'], vlim_dict_pop = (
        maps.gather_pop_map(df_raster, transform_dict))
    st.session_state['vlim_dict'] = (
        st.session_state['vlim_dict'] | vlim_dict_pop)

# Set up colour limits:
with containers['map_setup']:
    dicts_colours = colour_setup.select_colour_limits(
        map_outcome, st.session_state['vlim_dict'], 'msu', 'MSU')

# Make colour maps and traces:
with containers['map_setup']:
    default_cmap_name, all_cmaps = colour_setup.select_colour_maps()
for p, dp in dicts_colours.items():
    dicts_colours[p]['cmap'] = colour_setup.make_colour_list(
        default_cmap_name, vmin=dp['vmin'], vmax=dp['vmax'])

with containers['map_setup']:
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
        horizontal=True,
        on_change=set_rerun_map,
        key='maps_outcomes_outline'
        )

if st.session_state['rerun_maps']:
    # Make traces for maps:
    for col, arr in st.session_state['map_arrs_dict'].items():
        map_traces[col] = plot_maps.make_trace_heatmap(
            arr, transform_dict, dicts_colours[col], name=col)
    st.session_state['maps_fig'] = plot_maps.plot_outcome_maps(
        map_traces,
        ['usual_care', 'msu_minus_usual_care', 'pop'],
        dicts_colours,
        all_cmaps,
        outline_name,
        show_msu_bases=True,
        title=subgroup_map_label,
        )
    st.session_state['rerun_maps'] = False

with containers['map_fig']:
    plotly_config = plot_maps.get_map_config()
    st.plotly_chart(
        st.session_state['maps_fig'],
        # width='content',
        config=plotly_config
        )


# ----- Full LSOA results -----
# Generate on request, not by default with each re-run.
with containers['full_results_setup']:
    generate_full_data = st.checkbox(
        'Show options to generate full data',
        on_change=set_rerun_full_results
        )
if generate_full_data:
    with containers['full_results_setup']:
        full_results_type = reg.select_full_data_type()
    if st.session_state['rerun_full_results']:
        # Only rerun these if the following have changed:
        # "setup" section inputs; full results type.
        if full_results_type == 'lsoa':
            # Calculate LSOA-level results.
            redir_scens = ['usual_care', 'msu_ivt', 'msu_no_ivt']
            treats = ['ivt', 'mt']
            cols_times = [f'{s}_{t}' for s in redir_scens for t in treats]
            cols_times.remove('msu_no_ivt_ivt')
            st.session_state['dict_full_outcomes'] = (
                pop.gather_lsoa_level_outcomes(
                    st.session_state['dict_outcomes'],
                    st.session_state['df_lsoa_units_times'],
                    cols_times,
                    _log_loc=containers['log_full_results']
                    )
            )
        else:
            # Find how many admissions per region have each set of
            # unique treatment times:
            dict_this_region_unique_treatment_times = (
                reg.find_region_admissions_by_unique_travel_times(
                    st.session_state['df_lsoa_units_times'],
                    [full_results_type],
                    unique_travel=False,
                    project='muster',
                    _log_loc=containers['log_regions'])
                )
            # Calculate the full outcomes for just this selected region type
            # but for all the nested subsets (subgroup, scenario, LSOA subset):
            st.session_state['dict_full_outcomes'] = (
                reg.calculate_nested_average_outcomes(
                    st.session_state['dict_outcomes'],
                    dict_this_region_unique_treatment_times,
                    _log_loc=containers['log_full_results']
                    )
            )
        st.session_state['rerun_full_results'] = False
    else:
        pass
    with containers['full_results_setup']:
        if full_results_type == 'lsoa':
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown('Unit postcode lookup')
                st.dataframe(df_unit_services['ssnap_name'].sort_index())
            with cols[1]:
                st.markdown('Travel and treatment times:')
                st.dataframe(df_lsoa_units_times.set_index('LSOA'))
            for subgroup, df_full in (
                    st.session_state['dict_full_outcomes'].items()):
                st.subheader(
                    st.session_state['df_subgroups'].loc[subgroup, 'label'])
                st.dataframe(df_full)
        else:
            use_lsoa_subset_full = st.toggle(
                'Use only patients whose nearest unit does not provide MT.',
                value=True,
                key='full_lsoa_subset'
                )
            lsoa_subset_full = ('nearest_unit_no_mt' if use_lsoa_subset_full
                                else 'all_patients')
            for subgroup, dict_full in (
                    st.session_state['dict_full_outcomes'].items()):
                dfs = []
                for scen in ['usual_care', 'msu']:
                    df = dict_full[scen][lsoa_subset_full]
                    df = df.rename(columns=dict(
                        [(c, f'{c}_{scen}') for c in df.columns]))
                    dfs.append(df)
                df_full = pd.concat(dfs, axis='columns')

                if full_results_type == 'nearest_ivt_unit':
                    # Change postcodes to unit names:
                    df_full.index = df_full.index.map(
                        df_unit_services['ssnap_name'])
                    df_full = df_full.sort_index()

                st.subheader(
                    st.session_state['df_subgroups'].loc[subgroup, 'label'])
                st.dataframe(df_full)
else:
    # Remove stored full data.
    try:
        del st.session_state['dict_full_outcomes']
    except KeyError:
        pass
    set_rerun_full_results()
