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

# ----- Custom functions -----
import utilities.regions as reg
import utilities.maps as maps
import utilities.pathway as pathway
import utilities.outcomes as outcomes
import utilities.population as pop


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
    with c['units_setup']:
        c['units_map'] = st.container()
        with st.expander('Edit unit services'):
            c['units_df'] = st.container()

    c['pathway'] = st.container()
    with c['pathway']:
        c['pathway_inputs'] = st.container(horizontal=True)
    c['onion_setup'] = st.container()

    # ----- Results -----
    c['region_summaries'] = st.container()
    c['maps'] = st.container()
    with st.expander('Full data tables'):
        c['full_results'] = st.container()
    return c


containers = set_up_page_layout()

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
    df_unit_services = reg.select_unit_services()
df_lsoa_units_times = reg.find_nearest_units_each_lsoa(
    df_unit_services, _log_loc=containers['units_setup'])
with containers['units_map']:
    maps.draw_units_map(df_unit_services, df_lsoa_units_times)
unique_travel_for_ivt, unique_travel_for_mt, dict_unique_travel_pairs = (
    reg.find_unique_travel_times(df_lsoa_units_times,
                                 _log_loc=containers['units_setup'])
    )
dict_region_admissions_unique_times = (
    reg.find_region_admissions_by_unique_travel_times(
        df_lsoa_units_times, _log_loc=containers['units_setup'])
    )


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
dict_region_admissions_unique_treatment_times = (
    reg.find_region_admissions_by_unique_travel_times(
        df_lsoa_units_times, unique_travel=False,
        _log_loc=containers['pathway'])
    )
st.write(dict_region_admissions_unique_treatment_times['all_patients']['isdn'])


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
    _log_loc=containers['pathway'])

df_base_lvo_ivt_mt_better = outcomes.flag_lvo_ivt_better_than_mt(
    dict_base_outcomes['lvo_ivt'],
    dict_base_outcomes['lvo_mt'],
    unique_treatment_pairs,
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
st.write(dict_onion)

df_subgroups = pop.select_subgroups_for_results()
st.write(df_subgroups)

df_pop_usual_care, df_pop_redir = (
    pop.calculate_population_subgroup_grid(dict_onion))
st.write(df_pop_usual_care, df_pop_redir)
# df_pop_usual_care, df_pop_redir = pop.calculate_populations_for_subgroups(
#     df_pop_usual_care, df_pop_redir)


dict_outcomes = pop.calculate_unique_outcomes_onion(
    dict_base_outcomes, df_base_lvo_ivt_mt_better, dict_onion,
    df_treat_times_sets_unique,
    _log_loc=containers['onion_setup']
)


# ----- Region summaries -----
# Average the results over each geographical region.
# Find two copies of the results - one with all LSOA in the region
# and one with only LSOA whose nearest unit is not a CSC.
# --- CALCULATIONS:
# + Calculate admissions-weighted average outcomes.
dict_region_outcomes = calculate_region_outcomes(
    dict_region_unique_times, dict_base_outcomes,
    df_base_lvo_ivt_mt, _log_loc=containers['region_summaries']
    )

#MARK: Results
# ###################
# ##### RESULTS #####
# ###################
# ----- Region summaries -----
with containers['region_summaries']:
    highlighted_regions = select_highlighted_regions()
containers_highlighted_regions = st.container(horizontal=True)
df_summary = gather_summary_for_regions(
    dict_region_outcomes, highlighted_regions,
    _log_loc=containers['region_summaries']
    )  # 'Gathering mRS distributions for highlighted regions.'
df_mrs = gather_mrs_for_regions(
    dict_region_outcomes, highlighted_regions,
    _log_loc=containers['mrs_dists']
    )  # 'Gathering mRS distributions for highlighted regions.'

for r, region in enumerate(highlighted_regions):
    with containers_highlighted_regions:
        ch = st.container()
    with ch:
        display_region_summary()
        plot_region_mrs_dists()

# ----- Maps -----
# For the selected data type to show on the maps, gather the full
# LSOA-level data.
with containers['maps']:
    dict_map_options = select_map_data()
map_arrs = gather_map_data(
    dict_map_options, dict_outcomes,
    df_lsoa_units_times, _log_loc=containers['maps']  # 'Gathering data for maps.'
    )
with containers['maps']:
    plot_maps(map_arrs)


# ----- Full LSOA results -----
# Generate on request, not by default with each re-run.
with containers['full_results']:
    full_results_type = select_full_data_type()
try:
    df_full = dict_region_outcomes[full_results_type]
except KeyError:
    df_full = gather_lsoa_level_outcomes(
        dict_base_outcomes, df_base_lvo_ivt_mt, df_lsoa_units_times,
        _log_loc=containers['full_results']  # 'Gathering full LSOA-level results.'
        )
with containers['full_results']:
    st.dataframe(df_full)
