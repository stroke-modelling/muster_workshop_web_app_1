"""
Outcomes with usual care and redirection with maps for OPTIMIST.

This version calculates as few elements as possible. Instead of
calculating full data for thousands of LSOA, we instead process the
unique treatment times that may be shared across many LSOA.
The full LSOA results are only built up at the end when required.

# Note: logs print in wrong location for cached functions,
# so have extra "with" blocks in places.

Possibly worth doing - change big df postcodes to int, LSOA to FID (int).
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
from utilities.utils import set_rerun_map, set_rerun_full_results, \
    set_rerun_lsoa_units_times
import utilities.admissions as admissions
import utilities.times as times
import utilities.mrs_dists as mrs


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

try:
    page_last_run = st.session_state['page_last_run']
    if page_last_run != 'OPTIMIST':
        # Clear the MUSTER results.
        keys_to_del = list(st.session_state.keys())
        for key in keys_to_del:
            del st.session_state[key]
except KeyError:
    # No page has been run yet.
    pass
st.session_state['page_last_run'] = 'OPTIMIST'
# Set these so that all results run on first go of script:
change_keys = ['inputs_changed', 'rerun_region_summaries', 'rerun_maps',
               'rerun_full_results', 'rerun_lsoa_units_times']
for k in change_keys:
    if k not in st.session_state.keys():
        st.session_state[k] = True


def set_up_page_layout():
    """
    Set up container placement and return as dict.
    """
    c = {}
    st.title('Benefit in outcomes from redirection')

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
        cols = st.columns([1, 1])
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
            with cols[1]:
                st.markdown('After arrival:')
                c['pathway_inputs_units'] = st.container(
                    border=True, horizontal=True)

    with c['onion']:
        cols = st.columns([1, 1])
        with cols[0]:
            c['onion_fig'] = st.container()
        with cols[1]:
            c['onion_text'] = st.container()
        c['onion_text2'] = st.container()
        c['onion_setup'] = st.expander('Edit population proportions')

    with c['onion_subgroups']:
        c['pop_plots'] = st.container()

    # ----- Results -----
    c['run_results'] = st.container()
    c['region_select'] = st.container()
    c['results'] = st.container()
    c['map_setup'] = st.expander('Outcome map setup')
    c['full_results'] = st.expander('Full results tables')
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
    cols = st.columns([1, 2])
    with cols[1]:
        st.markdown('''
We assume that all of the MT units also provide IVT.

In usual care:
+ patients whose nearest unit has MT always travel directly to an MT
unit (:primary[red path]).
+ other patients travel first to the IVT unit and then if necessary are
transferred to the MT unit (:grey[grey path]).

With redirection, patients who would normally travel to the IVT unit
first (:grey[grey path]) instead travel directly to the MT unit
(:primary[red path]).
''')
    with cols[0]:
        reg.plot_basic_travel_options()
    st.markdown('''
For most patients, redirection means faster access to MT because of the
reduced travel time and delays for hospital transfer.
It also means slower access to IVT because of the increased travel time
to the MT unit compared with the IVT unit.
''')
with containers['pathway_text']:
    st.markdown('''
The time to treatment depends on the travel times and whether
redirection was considered.

Assumptions:
1. When redirection is considered, the ambulance spends more time
on-scene to do the pre-hospital diagnostic.
2. All stroke units share the same time from arrival to delivery of
IVT.
3. The time from arrival to delivery of MT can be different for
patients admitted directly to the MT unit and for patients who
received a transfer.
4. All other pathway timings are the same in every scenario.
''')
with containers['onion_text']:
    st.markdown('''
The population can be grouped in a series of subsets:
+ :grey-background[Full study population]:  
  + All patients suspected to be stroke by ambulance staff  
  __and__  
  + All patients confirmed to be stroke at hospital, conveyed by
ambulance but not initially suspected to be stroke by ambulance staff.
+ :orange-background[Ambulance suspected stroke population]: patients
suspected to be stroke by ambulance staff.
+ :yellow-background[Target population]: Ambulance suspected stroke
patients who have the pathway initiation crieria i.e. drip-ship area,
specified clinical features.
+ :green-background[Primary analysis population]: Patients with
ischaemic stroke.
+ :blue-background[Thrombectomy]: Patients with LVO ischaemic stroke
who receive thrombectomy.
''')
with containers['onion_text2']:
    st.markdown('''
The different layers of the onion have different make-ups of patients,
e.g. types of stroke and proportions treated.  
These proportions can be changed in the following table:
:red-background[(NOTE, November 2025, all values are placeholders)]
''')
with containers['pop_plots']:
    st.markdown('''
We calculate six base outcomes. These can be combined in different
proportions to find the outcomes for the ischaemic stroke patients in a
selected subgroup of patients.
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

# Setup for map:
df_raster, transform_dict = maps.load_lsoa_raster_lookup()
map_traces_constant = plot_maps.make_constant_map_traces()

with containers['units_df']:
    df_unit_services = reg.select_unit_services()
mt_units = df_unit_services.loc[df_unit_services['Use_MT'] == 1].index.values

# If units are updated, the following block runs.
# Calculate LSOA-unit allocation now so that the unit catchment
# can be shown on the units map.
if st.session_state['rerun_lsoa_units_times']:
    df_lsoa_units_times = reg.find_nearest_units_each_lsoa(
        df_unit_services,
        _log_loc=containers['log_units']
        )
    st.session_state['input_df_lsoa_units_times'] = df_lsoa_units_times
    # For map:
    (st.session_state['map_traces_shared'],
     st.session_state['input_df_unit_services']) = (
        plot_maps.make_shared_map_traces(
            df_unit_services,
            df_lsoa_units_times, df_raster, transform_dict
        )
    )
    set_rerun_lsoa_units_times(False)
else:
    pass

# Will update map_traces dict later with the outcome maps.
map_traces = st.session_state['map_traces_shared'] | map_traces_constant

with containers['units_text']:
    outline_labels_dict = {
        'none': 'None',
        'ambo22': 'Ambulance service',
        'icb': 'Integrated Care Board',
        'isdn': 'Integrated Stroke Delivery Network',
        'nearest_ivt_unit': 'Nearest IVT unit',
        'nearest_mt_unit': 'Nearest MT unit',
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
    'optimist', [containers['pathway_inputs_prehosp'],
                 containers['pathway_inputs_units']]
    )
series_treatment_times_without_travel = (
    pathway.calculate_treatment_times_without_travel(
        df_pathway_steps['value'],
        ['usual_care', 'prehospdiag'],
        _log_loc=containers['log_pathway']
        )
    )
with containers['pathway_fig']:
    pathway.draw_timeline(df_pathway_steps,
                          series_treatment_times_without_travel)


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

with containers['onion_fig']:
    pop.plot_onion()
with containers['onion_setup']:
    df_onion_pops = pop.set_up_onion_parameters()
with containers['onion_text']:
    dict_onion = pop.select_onion_population(df_onion_pops)
dict_onion = pop.calculate_population_subgroups(
    dict_onion, _log_loc=containers['log_onion'])
# Keep a copy of the label for this population.
# Use it throughout the results to make it clearer which patients
# are included.
str_this_population = dict_onion['label']
prop_this_population = dict_onion['prop_of_all_stroke']


# ----- Subgroups (this onion layer) -----
with containers['pop_plots']:
    df_subgroups = pop.select_subgroups_for_results()

dict_pops = pop.calculate_population_subgroup_grid(
    dict_onion, df_subgroups, _log_loc=containers['log_subgroups']
    )


with containers['pop_plots']:
    n_cols = 2
    cols = st.columns(n_cols)
    for i, s in enumerate(df_subgroups.index):
        with cols[i % n_cols]:
            c = st.container(border=True)
        with c:
            pop.plot_population_props(
                dict_pops['usual_care'][['scenario'] + [s]],
                dict_pops['redir_allowed'][['scenario'] + [s]],
                s,
                df_subgroups.loc[s]
                )

# ----- Outcomes -----
# Only recalculate results if anything above here has changed.
# Don't rerun outcomes when selecting options in the Results section.
with containers['run_results']:
    rerun_results = st.button('Recalculate results', type='primary')

# Bits we need regardless of re-run:
df_lsoa_units_times = st.session_state['input_df_lsoa_units_times']
# Not expecting to change df_unit_services from now on,
# only look up its data:
df_unit_services = st.session_state['input_df_unit_services']
dict_no_treatment_outcomes = outcomes.load_no_treatment_outcomes(
    _log_loc=containers['log_pathway'])

# Calculate results if this is the first go through the app
# or the button has been pressed.
if ('dict_outcomes' not in st.session_state.keys()) or rerun_results:

    unique_travel_for_ivt, unique_travel_for_mt, dict_unique_travel_pairs = (
        reg.find_unique_travel_times(
            df_lsoa_units_times, _log_loc=containers['log_units'])
        )
    unique_treatment_ivt, unique_treatment_mt = (
        pathway.calculate_treatment_times(
            series_treatment_times_without_travel,
            unique_travel_for_ivt,
            unique_travel_for_mt,
            _log_loc=containers['log_pathway']
            )
    )
    unique_treatment_pairs = pathway.find_unique_treatment_time_pairs(
        dict_unique_travel_pairs, series_treatment_times_without_travel,
        _log=True, _log_loc=containers['log_pathway'],
    )

    # LSOA-level treatment times:
    df_lsoa_units_times = (
        pathway.calculate_treatment_times_each_lsoa_scenarios(
            df_lsoa_units_times,
            series_treatment_times_without_travel,
            _log_loc=containers['log_pathway']
            )
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

    # ----- Base outcomes -----
    # Calculate base outcomes for the given travel times and scenarios.
    # Find outcomes for all of the unique treatment times given.
    # --- CALCULATIONS:
    # + Calculate outcomes for unique treatment times for the base
    #   groups: nLVO + IVT, LVO + IVT, LVO + MT.
    # + For unique pairs of times to treatment, find when LVO + IVT
    #   is better than LVO + MT.
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

    # Re-run results.
    st.session_state['dict_outcomes'] = {}
    for s in df_subgroups.index:
        st.session_state['dict_outcomes'][s] = (
            pop.calculate_unique_outcomes_onion(
                dict_base_outcomes,
                dict_pops,
                df_subgroups.loc[s],
                df_treat_times_sets_unique,
                s,
                _log_loc=containers['log_subgroups']
            )
        )
    st.session_state['df_lsoa_units_times'] = df_lsoa_units_times
    st.session_state['df_subgroups'] = df_subgroups
    st.session_state['dict_pops'] = dict_pops
    st.session_state['dict_onion'] = dict_onion
    st.session_state['inputs_changed'] = False
    st.session_state['rerun_region_summaries'] = True
    st.session_state['rerun_maps'] = True
    st.session_state['rerun_full_results'] = True
else:
    if st.session_state['inputs_changed']:
        with containers['run_results']:
            st.warning('Results are for previous set of inputs.', icon='⚠️')
    # Pull out this data from before:
    df_lsoa_units_times = st.session_state['df_lsoa_units_times']


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
with containers['results']:
    c_names = list(df_highlighted_regions['highlighted_region'].values)
    conts_tabs = st.tabs(c_names)
    for (label, cont) in zip(c_names, conts_tabs):
        containers[f'results_{label}'] = cont

# --- CALCULATIONS:
# + Calculate admissions-weighted average outcomes.
(options_labels, scenario_labels, scenario_help,
outcome_labels, outcome_formats, bar_colours) = mrs.set_up_mrs_labels()

if st.session_state['rerun_region_summaries']:
    # Clear existing map figs/setup:
    keys_to_clear = [k for k in st.session_state.keys()
                     if k.startswith('map') & ('region' in k)]
    keys_to_save = []
    for key in keys_to_clear:
        # Keep the key if it contains a highlighted region.
        for r in df_highlighted_regions['highlighted_region']:
            if r in key:
                keys_to_save.append(key)
    keys_to_clear = list(set(keys_to_clear) - set(keys_to_save))
    for key in keys_to_clear:
        del st.session_state[key]

    if len(highlighted_region_types) == 0:
        # Placeholder empty dfs:
        st.session_state['dict_highlighted_region_travel_times'] = {}
        st.session_state['dict_highlighted_region_unique_treatment_times'] = {}
        st.session_state['df_highlighted_region_admissions'] = pd.DataFrame()
        st.session_state['df_region_unit_admissions'] = pd.DataFrame()
        st.session_state['dict_highlighted_region_outcomes'] = {}
        st.session_state[
            'dict_highlighted_region_average_treatment_times'] = {}
    else:
        st.session_state['dict_highlighted_region_travel_times'] = (
            reg.find_region_admissions_by_unique_travel_times(
                df_lsoa_units_times,
                highlighted_region_types,
                df_highlighted_regions,
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
                _log_loc=containers['log_regions'])
            )
        (
            st.session_state['df_highlighted_region_admissions'],
            st.session_state['df_region_unit_admissions']
            ) = (
            reg.find_unit_admissions_by_region(
                df_lsoa_units_times,
                dict_onion['prop_of_all_stroke'],
                highlighted_region_types,
                df_highlighted_regions,
                _log_loc=containers['log_regions'],
                )
        )

        # Nest levels: subgroup, scenario, lsoa subset.
        st.session_state['dict_highlighted_region_outcomes'] = (
            reg.calculate_nested_average_outcomes(
                st.session_state['dict_outcomes'],
                st.session_state[
                    'dict_highlighted_region_unique_treatment_times'],
                use_highlighted_teams=True,
                _log_loc=containers['log_regions']
                )
        )
        st.session_state['dict_highlighted_region_average_treatment_times'] = (
            reg.calculate_average_treatment_times_highlighted_regions(
                st.session_state[
                    'dict_highlighted_region_unique_treatment_times'],
                _log_loc=containers['log_regions']
                )
        )


        # ----- Calculations for results for each region -----
        st.session_state['dict_admissions_onion'] = {}
        for r, region in enumerate(df_highlighted_regions['highlighted_region']):
            # --- Admissions onion ---
            # Define these to shorten function inputs:
            d = st.session_state['df_region_unit_admissions']
            p = dict_onion['prop_of_all_stroke']
            df_admissions_onion = admissions.calculate_region_admissions_onion(
                d[f'{region}_nearest_unit_no_mt'] / p,
                d[f'{region}_all_patients'] / p,
                df_onion_pops,
                )
            # Store results:
            st.session_state['dict_admissions_onion'][region] = df_admissions_onion

        st.session_state['dict_networks'] = {}
        for r, region in enumerate(df_highlighted_regions['highlighted_region']):
            # --- Admissions changes ---
            # For now, calculate for all patients:
            # lsoa_subset = 'nearest_unit_no_mt' if use_lsoa_subset else 'all_patients'
            lsoa_subset = 'all_patients'
            # Setup for admissions:
            c = f'{region}_{lsoa_subset}'
            cols = ['nearest_ivt_unit', 'nearest_mt_unit', 'transfer_unit', c]
            df_network = (st.session_state['df_region_unit_admissions'][cols].copy()
                        .rename(columns={c: 'admissions'}))
            # Only keep units that have admissions from selected region:
            df_network = df_network.dropna(subset=['admissions'])
            # Find tracked admissions between units
            # split by usual care and redirection:
            df_net_u = admissions.calculate_network_usual_care(
                df_network, st.session_state['dict_pops']['usual_care'], mt_units)
            df_net_r = admissions.calculate_network_redir(
                df_network, st.session_state['dict_pops']['redir_allowed'], mt_units)
            # Pick out which units exist in these networks:
            dict_network_units = admissions.gather_network_units(df_net_u, df_net_r)


            # Tabulate direct admissions and transfers in usual care
            # and in redir scenario for each unit.
            cols = ['first_unit', 'admissions_catchment_to_first_unit',
                    'admissions_first_unit_to_transfer']
            df_unit_admissions = df_net_u[cols].copy()
            # Combine redirection scenario rows. Patients for one first unit
            # can cover many rows depending on their catchment unit,
            # so find total admissions to each unit regardless of whether
            # they were redirected there.
            df_unit_admissions_redir = (
                df_net_r[cols].copy().groupby('first_unit').sum().reset_index())

            # if 1 == 0:
            #     # Optionally limit to only one MT unit.
            #     with containers_h['redir_flow_mt_select']:
            #         mt_unit_here = admissions.select_mt_unit_here(
            #             df_unit_services, dict_network_units)
            #     if mt_unit_here == 'all':
            #         mt_unit_here_label = 'MT units'
            #     else:
            #         mt_unit_here_label = df_unit_services.loc[mt_unit_here, 'ssnap_name']
            #         df_net_u = df_net_u[df_net_u['transfer_unit'] == mt_unit_here]
            #         df_net_r = df_net_r[df_net_r['transfer_unit'] == mt_unit_here]
            # mt_unit_here_label = 'MT units'

            # Calculate admissions for the flowchart.
            # Squash the network dfs by unit into generic unit type.
            df_net_u_gen = admissions.convert_network_to_generic(df_net_u)
            df_net_r_gen = admissions.convert_network_to_generic(df_net_r)
            # Rejig into table to show:
            df_region_admissions_generic = admissions.gather_region_admissions_generic(
                df_net_u_gen, df_net_r_gen)
            # Rejig for flowcharts:
            dict_region_admissions_generic = (
                admissions.calculate_region_admissions_generic(
                    df_region_admissions_generic))

            # Store results:
            st.session_state['dict_networks'][region] = {
                'df_net_u': df_net_u,
                'df_net_r': df_net_r,
                'units': dict_network_units,
                'df_generic': df_region_admissions_generic,
                'dict_generic': dict_region_admissions_generic,
            }

        st.session_state['dict_map_catchment'] = {}
        for r, region in enumerate(df_highlighted_regions['highlighted_region']):
            # Set up region display:
            region_type = df_highlighted_regions.loc[
                df_highlighted_regions['highlighted_region'] == region,
                'region_type'].values[0]

            # --- Catchment map ---
            gdf_units = plot_maps.generate_node_coordinates(
                df_unit_services, st.session_state['dict_networks'][region]['units']['all'])
            gdf_region, region_display_name = (
                plot_maps.load_region_outline_here(region_type, region))
            # Set up colours for catchment units:
            catch_trace, transform_dict_units, gdf_nearest_units = (
                plot_maps.make_unit_catchment_raster(
                    df_lsoa_units_times,
                    # gdf_units.loc[[n.replace('nearest_', '')
                    #                 for n in nearest_units]],
                    gdf_units.loc[st.session_state['dict_networks'][region]['units']['all']],
                    df_raster,
                    transform_dict,
                    nearest_unit_column='nearest_ivt_unit',
                    redo_transform=True,
                    )
            )
            # Limit the extent of the map to reasonable bounds:
            bounds, x_buffer, y_buffer = plot_maps.set_network_map_bounds(
                gdf_units, gdf_region, transform_dict_units)
            gdf_nearest_units = (
                plot_maps.make_coords_nearest_unit_catchment(
                    gdf_units, df_net_u, bounds,
                    st.session_state['dict_networks'][region]['units']['nearest'],
                    x_buffer, y_buffer
                    )
                )
            # New gdf for plotting stroke unit markers:
            gdf_units_here = pd.merge(
                gdf_units.reset_index(),
                df_net_u[['first_unit', 'admissions']],
                left_on='Postcode', right_on='first_unit', how='left'
                ).set_index('Postcode')
            # Store results:
            st.session_state['dict_map_catchment'][region] = {
                'gdf_units': gdf_units_here,
                'gdf_nearest_units': gdf_nearest_units,
                'bounds': bounds,
                'catch_trace': catch_trace,
                'gdf_region': gdf_region,
            }

        st.session_state['dict_times'] = {}
        for r, region in enumerate(df_highlighted_regions['highlighted_region']):
            # --- Time changes ---
            # Quantiles.
            # Set up time df for quantile calculations:
            # This dict has separate entries for "all_patients" and
            # "nearest_unit_no_mt":
            s = 'dict_highlighted_region_unique_treatment_times'
            df_times = st.session_state[s]['nearest_unit_no_mt'][[region]]
            df_times = df_times.reset_index().copy()
            # Calculate difference due to redir:
            r = 'redirection_approved'
            u = 'usual_care'
            c = 'redir_change'
            for i in ['ivt', 'mt']:
                df_times[f'{c}_{i}'] = df_times[f'{r}_{i}'] - df_times[f'{u}_{i}']
            # Dataframe to be displayed:
            df_q_ivt = times.calculate_quantiles(
                df_times,
                [f'{u}_ivt', f'{c}_ivt', f'{u}_mt', f'{c}_mt'],
                region,
                [0.05, 0.25, 0.5, 0.75, 0.95]
                )

            # Grid.
            # This dict has separate entries for "all_patients" and
            # "nearest_unit_no_mt":
            s = 'dict_highlighted_region_unique_treatment_times'
            # Pick out a dataframe with the treatment times as index
            # and a single region's admissions as the column:
            df_times = st.session_state[s]['nearest_unit_no_mt'][[region]]
            # Scale down admissions to match redir patients only,
            # not all patients with nearest unit no mt:
            d = st.session_state['dict_pops']['redir_allowed']
            p_redir = d.loc[d['scenario'] == 'redir_accepted', 'full_population'].sum()
            df_times *= p_redir
            # Convert column to 2D grid:
            df_times_grid = times.create_time_diff_admissions_grid(df_times)

            # Store results:
            st.session_state['dict_times'][region] = {
                'quantiles': df_q_ivt,
                'grid': df_times_grid,
            }

        st.session_state['dict_mrs'] = {}
        for r, region in enumerate(df_highlighted_regions['highlighted_region']):
            # --- mRS distributions ---

            # mRS distribution summary values and bar charts:
            dict_metrics = {}
            dict_mrs_bars = {}
            for s, subgroup in enumerate(st.session_state['df_subgroups'].index):
                pops = st.session_state['dict_pops']['usual_care'][subgroup]
                dict_mrs_bars[subgroup], dict_metrics[subgroup] = (
                    mrs.calculate_mrs_bars(
                        st.session_state['dict_highlighted_region_outcomes'][subgroup],
                        region,
                        lsoa_subset,
                        pops,
                        dict_no_treatment_outcomes,
                        bar_colours,
                        options_labels
                    ))

            # Gather metrics:
            dict_df_metrics = {}
            for key in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
                df_metrics = pd.DataFrame()
                for subgroup, d in dict_metrics.items():
                    df_metrics[subgroup] = d[key]
                    df_metrics.loc['subgroup', subgroup] = (
                        st.session_state['df_subgroups'].loc[subgroup, 'label'])
                dict_df_metrics[key] = df_metrics.transpose()

            # Gather mRS dists:
            dict_df_mrs = {}
            for subgroup, dict_mrs_here in dict_mrs_bars.items():
                s_scens = []
                for scenario, dict_scen in dict_mrs_here.items():
                    s_scen = []
                    for col in ['noncum', 'cum', 'std']:
                        try:
                            s = dict_scen[col]
                            s.name = scenario
                            s_scen.append(s)
                        except KeyError:
                            pass
                    s_scens.append(pd.concat(s_scen, axis='rows'))
                df_mrs = pd.concat(s_scens, axis='columns')
                dict_df_mrs[subgroup] = df_mrs

            # Store results:
            st.session_state['dict_mrs'][region] = {
                'dict_metrics': dict_metrics,
                'dict_mrs_bars': dict_mrs_bars,
                'dict_df_metrics': dict_df_metrics,
                'dict_df_mrs': dict_df_mrs,
            }

        st.session_state['rerun_region_summaries'] = False

else:
    pass


# Map setup.
# Do this only once, not a new set for each set of summary results.
# For the selected data type to show on the maps, gather the full
# LSOA-level data. Then reshape into the raster array.
# Re-run map if the following have changed:
# "setup" section inputs; colour limits.
with containers['map_setup']:
    map_outcome = outcomes.select_outcome_type()
    # Set up colour limits:
    dicts_colours = colour_setup.select_colour_limits(map_outcome)
        # map_outcome, st.session_state['map_vlim_dict'])
    # Make colour maps:
    default_cmap_name, all_cmaps = colour_setup.select_colour_maps()
    for p, dp in dicts_colours.items():
        dicts_colours[p]['cmap'] = colour_setup.make_colour_list(
            default_cmap_name, vmin=dp['vmin'], vmax=dp['vmax'])

    outline_labels_dict = {
        'none': 'None',
        'ambo22': 'Ambulance service',
        'icb': 'Integrated Care Board',
        'isdn': 'Integrated Stroke Delivery Network',
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

# For each highlighted region, show summary results:
containers_highlighted = {}
for r, region in enumerate(df_highlighted_regions['highlighted_region']):

    # Set up region display:
    region_type = df_highlighted_regions.loc[
        df_highlighted_regions['highlighted_region'] == region,
        'region_type'].values[0]
    # Pick out label for the box:
    if region_type == 'nearest_ivt_unit':
        region_label = df_unit_services.loc[region, 'ssnap_name']
    else:
        region_label = region

    # Flag whether we need to make a new map:
    if f'maps_fig_region_{region}' in list(st.session_state.keys()):
        pass
    else:
        st.session_state['rerun_maps'] = True

    # Set up containers:
    containers_highlighted[region] = {}
    # For shorter lines:
    containers_h = containers_highlighted[region]
    with containers[f'results_{region}']:
        containers_h['top'] = st.container(border=True)
    with containers_h['top']:
        st.header(region_label)
        containers_h['h'] = st.container(horizontal=True)
        containers_h['h2'] = st.container(horizontal=True)
        containers_h['h3'] = st.container()

    # container_labels = ['redir_flow', 'redir_time', 'mrs_dists']
    with containers_h['h']:
        c = st.columns([1, 2])
        with c[0]:
            containers_h['onion'] = st.container()
            containers_h['catchment_map'] = st.container()
            containers_h['onion_extra'] = st.container()
        with c[1]:
            containers_h['redir_flow'] = st.container()
            containers_h['redir_time_top'] = st.container()
            with containers_h['redir_time_top']:
                containers_h['redir_time'] = st.container()
                c2 = st.columns(2)
                with c2[0]:
                    containers_h['redir_time_single'] = st.container()
                with c2[1]:
                    containers_h['redir_time_combo'] = st.container()
            # c2 = st.columns(2)
            # # with c2[0]:
            # with c2[1]:
            containers_h['outcomes_top'] = st.container()
            with containers_h['outcomes_top']:
                containers_h['outcomes'] = st.container()
                c2 = st.columns(2)
                with c2[0]:
                    containers_h['outcomes_perc'] = st.container()
                with c2[1]:
                    containers_h['outcomes_av'] = st.container()

                containers_h['mrs_dists'] = st.container()
                c2 = st.columns(2)
                with c2[0]:
                    containers_h['mrs_dists_left'] = st.container()
                with c2[1]:
                    containers_h['mrs_dists_right'] = st.container()

        # for label in container_labels[:3]:
        #     containers_h[label] = st.container()
    # with containers_h['h2']:
    #     for label in container_labels[3:]:
    #         containers_h[label] = st.container()
    with containers_h['h3']:
        containers_h['outcome_maps'] = st.container()
    with containers_h['outcome_maps']:
        containers_h['map_fig'] = st.container()
    with containers_h['h3']:
        containers_h['map_network'] = st.container()


    with containers_h['redir_flow']:
        st.subheader('Admissions changes')
        (containers_h['redir_flow_0'],
         containers_h['redir_flow_1']) = st.columns(2, gap=None)
        (containers_h['redir_flow_mt_select'],
         containers_h['redir_flow_data']) = st.columns([1, 3])#, gap=None)
    with containers_h['redir_time']:
        st.subheader('Time changes')
    with containers_h['outcomes']:
        st.subheader('Outcomes')


    # ----- Display results -----
    # Onion:
    df_admissions_onion = st.session_state['dict_admissions_onion'][region]
    # Pick out proportion for later display:
    s = df_admissions_onion.loc[str_this_population]
    prop_here_nearest_csc = s['admissions_nearest_csc'] / s['admissions']
    # Highlight admissions for this onion layer:
    admissions_here = s['admissions']
    admissions_here_nearest_csc = s['admissions_nearest_csc']
    admissions_here_nearest_atc = s['admissions_nearest_atc']
    with containers_h['onion']:
        st.metric(f'__{str_this_population}__:', f'{admissions_here:,.0f} patients')
        st.markdown(f'({prop_this_population:.1%} of all stroke)')
        st.markdown(f'__{prop_here_nearest_csc:.0%} have nearest unit with MT__')
    with containers_h['onion_extra']:
        # Full table:
        col_labels = {
            'admissions': 'Total',
            'admissions_nearest_csc': 'Nearest CSC',
            'admissions_nearest_atc': 'Nearest not CSC',
            }
        config = dict([
            (c, st.column_config.NumberColumn(
                col_labels[c], format='%.0f', width='small'
                ))
            for c in df_admissions_onion.columns
            ])
        config['_index'] = st.column_config.TextColumn(width='small')
        with st.expander('Admissions for all populations'):
            st.dataframe(df_admissions_onion, column_config=config)

    # Catchment map:
    with containers_h['catchment_map']:
        # Plot graph:
        plot_maps.plot_catchment(
            st.session_state['dict_map_catchment'][region]['gdf_units'],
            st.session_state['dict_map_catchment'][region]['bounds'],
            st.session_state['dict_map_catchment'][region]['catch_trace'],
            st.session_state['dict_map_catchment'][region]['gdf_region'],
            region_label
            )

    # Redirection flowcharts:
    with containers_h['redir_flow_0']:
        # c = st.columns(2, gap=None)
        # with c[0]:
        d = st.session_state['dict_networks'][region]['dict_generic']
        st.markdown('Usual care:')
        reg.plot_generic_travel_admissions(
            d['mt_usual_care'],
            d['no_mt_usual_care'],
            mt_label='MT units'
            )
    with containers_h['redir_flow_1']:
        st.markdown('Redirection available:')
        reg.plot_generic_travel_admissions(
            d['mt_redir'],
            d['no_mt_redir'],
            mt_label='MT units'
            )
    with containers_h['redir_flow_data']:
        with st.expander('Data behind the flowcharts'):
            st.markdown(
                '''Column names show where patients are going to.
                Row names are a combination of (MT/no MT),
                (usual care / redirection), and
                (patients nearest CSC / patients nearest IVT unit).
                '''
                )
            df = st.session_state['dict_networks'][region]['df_generic']
            column_config = {'_index': st.column_config.TextColumn(width=150)}
            for c in df.index:
                column_config[c] = st.column_config.NumberColumn(format='%.1f')
            st.dataframe(df.transpose(),
                         column_config=column_config,
                         height=160)

    # Redirection time change:
    with containers_h['redir_time']:
        # Summary values:
        df_q_ivt = st.session_state['dict_times'][region]['quantiles']
        t_ivt = df_q_ivt.loc[0.5, 'redir_change_ivt']
        t_mt = df_q_ivt.loc[0.5, 'redir_change_mt']

        def make_string_time_change(t):
            if t > 0:
                s = f'__:red[↑ {t:.0f}]__ minutes later'
            elif t < 0:
                s = f'__:green[↓ {abs(t):.0f}]__ minutes sooner'
            elif t == 0:
                s = 'no change.'
            return s
        s_ivt = make_string_time_change(t_ivt)
        s_mt = make_string_time_change(t_mt)

        c = st.columns(3)
        with c[0]:
            st.markdown('Median time changes:')
        with c[1]:
            s = st.container(border=True)
            with s:
                st.markdown(f'__IVT__ {s_ivt}')
        with c[2]:
            s = st.container(border=True)
            with s:
                st.markdown(f'__MT__ {s_mt}')

        with containers_h['redir_time_single']:
            with st.expander('Time statistics for one treatment'):
                st.markdown(''.join([
                    'The only patients shown are those where redirection ',
                    'changes the treatment times, and the time changes are ',
                    'when redirection is accepted.'
                ]))

                st.markdown(''.join([
                    'Each time field is considered separately. ',
                ]))
                # Display quantile table:
                st.dataframe(df_q_ivt)

        df_times_grid = st.session_state['dict_times'][region]['grid']
        with containers_h['redir_time_combo']:
            with st.expander('Time changes for both treatments'):
                st.markdown(''.join([
                    'The only patients shown are those where redirection ',
                    'changes the treatment times, and the time changes are ',
                    'when redirection is accepted.'
                ]))
                times.plot_time_diff_admissions_grid(df_times_grid.round(0))
            # with st.expander('Data behind the figure'):
                st.markdown('__Data behind the figure:__')
                st.markdown(''.join([
                    'Rows are the change in time to IVT and ',
                    'columns are the change in time to MT. ',
                    'Both are rounded to the nearest 5 minutes. ',
                ]))
                st.dataframe(df_times_grid)

    # Outcome metrics:
    cols_to_show = ['usual_care', 'redir_allowed',
                    'diff_redir_allowed_minus_usual_care']
    keys_to_show = ['mrs_0-2', 'mrs_shift']
    conts = [containers_h['outcomes_perc'], containers_h['outcomes_av']]
    with containers_h['outcomes']:
        i = 0
        for key in keys_to_show:
            column_config = {'_index': st.column_config.TextColumn(width=150)}
            for c in cols_to_show:
                column_config[c] = st.column_config.NumberColumn(
                    width=40, label=scenario_labels[c], help=scenario_help[c],
                    format=outcome_formats[key]
                    )
            with conts[i]:
                st.markdown(outcome_labels[key])
                df = st.session_state['dict_mrs'][region]['dict_df_metrics'][key].reset_index(drop=True).set_index('subgroup')
                df = df[cols_to_show]
                st.dataframe(df, column_config=column_config)
            i += 1

    # mRS dists:
    with containers_h['mrs_dists_left']:
        with st.expander('__mRS distributions__ bar charts'):
            containers_h['mrs_figs'] = st.container()
            containers_h['mrs_options'] = st.container()
        with containers_h['mrs_options']:
            def f(label):
                """Display subgroup with nice name instead of key."""
                return options_labels[label]
            st.multiselect(
                'Scenarios to display on the bar chart',
                options_labels.keys(),
                format_func=f,
                default=['usual_care', 'redir_allowed', 'no_treatment'],
                key=f'mrs_dist_options_{region}',
            )
        with containers_h['mrs_figs']:
            for s, subgroup in enumerate(st.session_state['df_subgroups'].index):
                subgroup_label = st.session_state['df_subgroups'].loc[subgroup, 'label']
                st.markdown(f'__{subgroup_label}__')
                try:
                    mrs_lists_dict = st.session_state['dict_mrs'][region]['dict_mrs_bars'][subgroup]
                    show = True
                except KeyError:
                    st.markdown('No data available.')
                    show = False
                if show:
                    mrs_lists_dict_to_show = {}
                    for k in st.session_state[f'mrs_dist_options_{region}']:
                        mrs_lists_dict_to_show[k] = mrs_lists_dict[k]
                    mrs.plot_mrs_bars(mrs_lists_dict_to_show,
                                    key='_'.join([region, subgroup]))

    with containers_h['mrs_dists_right']:
        with st.expander('Data behind the mRS bar charts'):
            for subgroup, df in st.session_state['dict_mrs'][region]['dict_df_mrs'].items():
                subgroup_label = st.session_state['df_subgroups'].loc[subgroup, 'label']
                st.markdown(subgroup_label)
                st.dataframe(df, height=160)

    # Outcome maps:
    # Gather data for maps:
    with containers_h['map_fig']:
        st.subheader('Outcome maps')
    with containers_h['map_fig']:
        subgroup_map, subgroup_map_label = maps.select_map_data(
            st.session_state['df_subgroups'], region
        )
    map_title = ''.join([
    f'{subgroup_map_label} — of {str_this_population}',
    f'({prop_this_population:.1%} of all stroke)'
    ])
    with containers_h['map_fig']:
        use_full_redir = st.toggle(
            '''In middle map, include "reject redirection" and
            "usual care" patients.''',
            value=False,
            key=f'map_full_redir_subset_region_{region}',
            on_change=set_rerun_map
            )
    redir_subset = ('redir_allowed' if use_full_redir else 'redir_accepted_only')
    if st.session_state['rerun_maps']:
        df_times = st.session_state['df_lsoa_units_times']
        if region == 'National':
            pass
        else:
            # Limit the LSOAs to only those in the selected region.
            # units_here = [n.replace('nearest_', '') for n in nearest_units]
            # mask = df_times['nearest_ivt_unit'].isin(units_here)
            df_lsoa_to_keep = reg.load_lsoa_region_lookups()
            m = df_lsoa_to_keep[region_type] == region
            s_lsoa_to_keep = df_lsoa_to_keep.loc[m, 'lsoa']
            mask = df_times['LSOA'].isin(s_lsoa_to_keep)
            df_times = df_times.loc[mask].copy()

        (st.session_state[f'map_arrs_dict_region_{region}'],
         st.session_state[f'map_vlim_dict_region_{region}']) = (
            maps.gather_map_arrays(
                st.session_state['dict_outcomes'][subgroup_map]['usual_care'],
                st.session_state['dict_outcomes'][subgroup_map][redir_subset],
                df_times,
                df_raster,
                transform_dict,
                col_map=map_outcome,
                _log_loc=containers['log_maps']
                )
        )
        st.session_state[f'map_arrs_dict_region_{region}']['pop'], vlim_dict_pop = (
            maps.gather_pop_map(df_raster, transform_dict))
        st.session_state[f'map_vlim_dict_region_{region}'] = (
            st.session_state[f'map_vlim_dict_region_{region}'] | vlim_dict_pop)

    maps_to_show = ['usual_care', 'redir_minus_usual_care']
    with containers_h['map_fig']:
        if st.toggle('Show population density map.',
                     on_change=set_rerun_map, value=True,
                     key=f'toggle_pop_map_{region}'):
            maps_to_show.append('pop')

    if st.session_state['rerun_maps']:
        # Make traces for maps:
        for col, arr in st.session_state[f'map_arrs_dict_region_{region}'].items():
            map_traces[col] = plot_maps.make_trace_heatmap(
                arr, transform_dict, dicts_colours[col], name=col)
        st.session_state[f'maps_fig_region_{region}'] = plot_maps.plot_outcome_maps(
            map_traces,
            maps_to_show,
            dicts_colours,
            all_cmaps,
            outline_name=outline_name,
            title=map_title,
            gdf_single_region=(st.session_state['dict_map_catchment'][region]['gdf_region'] if region != 'National' else None),
            region_display_name=(region_label if region != 'National' else None),
            bounds=st.session_state['dict_map_catchment'][region]['bounds'],
            )

    with containers_h['map_fig']:
        plotly_config = plot_maps.get_map_config()
        st.plotly_chart(
            st.session_state[f'maps_fig_region_{region}'],
            config=plotly_config,
            # width='content',
            key=region
            )

    with containers_h['map_network']:
        plot_maps.plot_networks(
            st.session_state['dict_networks'][region]['df_net_u'],
            st.session_state['dict_networks'][region]['df_net_r'],
            df_unit_services,
            st.session_state['dict_map_catchment'][region]['gdf_nearest_units'],
            st.session_state['dict_map_catchment'][region]['gdf_units'],
            st.session_state['dict_map_catchment'][region]['bounds'],
            st.session_state['dict_map_catchment'][region]['gdf_region'],
            region_display_name='',
            subplot_titles=['Usual care', 'Redirection available']
            )

        admissions.plot_admissions_sankey(
            st.session_state['dict_networks'][region]['df_net_u'],
            df_unit_services
            )
        admissions.plot_admissions_sankey(
            st.session_state['dict_networks'][region]['df_net_r'],
            df_unit_services
            )

st.session_state['rerun_maps'] = False

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
            redir_scens = ['usual_care', 'redirection_approved',
                           'redirection_rejected']
            treats = ['ivt', 'mt']
            cols_times = [f'{s}_{t}' for s in redir_scens for t in treats]
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
                    _log_loc=containers['log_regions'])
                )
            # Calculate the full outcomes for just this selected region type
            # but for all the nested subsets (subgroup, scenario, LSOA subset):
            st.session_state['dict_full_outcomes'] = (
                reg.calculate_nested_average_outcomes(
                    st.session_state['dict_outcomes'],
                    dict_this_region_unique_treatment_times,
                    use_highlighted_teams=False,
                    _log_loc=containers['log_full_results']
                    )
            )
        st.session_state['rerun_full_results'] = False
    else:
        pass
    with containers['full_results_setup']:
        if full_results_type == 'lsoa':
            # Show several dataframes to prevent repeated data.
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown('Unit postcode lookup')
                st.dataframe(df_unit_services['ssnap_name'].sort_index())
            with cols[1]:
                st.markdown('Travel and treatment times:')
                st.dataframe(st.session_state['df_lsoa_units_times']
                             .set_index('LSOA'))
            for subgroup, df_full in (
                    st.session_state['dict_full_outcomes'].items()):
                st.subheader(st.session_state['df_subgroups']
                             .loc[subgroup, 'label'])
                st.dataframe(df_full)
        else:
            use_lsoa_subset_full = st.toggle(
                'Exclude patients whose nearest unit provides MT.',
                value=True,
                key='full_lsoa_subset'
                )
            lsoa_subset_full = ('nearest_unit_no_mt' if use_lsoa_subset_full
                                else 'all_patients')
            for subgroup, dict_full in (
                    st.session_state['dict_full_outcomes'].items()):
                # Put the usual care and redirection data in the same df:
                dfs = []
                for scen in ['usual_care', 'redir_allowed']:
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
