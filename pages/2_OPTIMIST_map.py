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
import utilities.calculations as calc
import utilities.regions as reg
import utilities.maps as maps


#MARK: Functions
# #####################
# ##### FUNCTIONS #####
# #####################


#MARK: Main
# ################
# ##### MAIN #####
# ################
if __name__ == '__main__':
    #MARK: Page layout
    # #######################
    # ##### PAGE LAYOUT #####
    # #######################
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
        df_unit_services, log_loc=containers['units_setup'])
    with containers['units_map']:
        maps.draw_units_map(df_unit_services, df_lsoa_units_times)
    unique_travel_for_ivt, unique_travel_for_mt = (
        reg.find_unique_travel_times(df_lsoa_units_times,
                                     log_loc=containers['units_setup'])
        )
    dict_region_unique_times = (
        reg.find_region_admissions_by_unique_travel_times(
            df_lsoa_units_times, log_loc=containers['units_setup'])
        )

    # ----- Pathway timings -----
    # Show the summary of pathway timings for each case: usual care;
    # redirection approved; redirection rejected. Show a timeline
    # image with the timings for the separate steps made clear.
    # Allow the timings to be changed with a series of widgets.
    # --- CALCULATIONS:
    # + Add up treatment times without travel for IVT and MT
    #   in each scenario.
    # + Calculate outcomes for unique treatment times for the base
    #   groups: nLVO + IVT, LVO + IVT, LVO + MT.
    # + For unique pairs of times to treatment, find when LVO + IVT
    #   is better than LVO + MT.

    with containers['pathway']:
        df_pathway_steps = select_pathway_timings()
    dict_treatment_times_without_travel = (
        calculate_treatment_times_without_travel(
            df_pathway_steps, log_loc=containers['pathway']
            )  # 'Calculated treatment times without travel.'
        )
    df_treatment_times = calculate_treatment_times(
        dict_treatment_times_without_travel,
        df_unique_times_travel,
        log_loc=containers['pathway']
        )  # 'Calculated unique treatment times.'
    dict_base_outcomes = calculate_unique_outcomes(
        df_treatment_times, log_loc=containers['pathway'])  # 'Calculated unique outcomes for base scenarios.'
    df_base_lvo_ivt_mt = flag_lvo_ivt_better_than_mt(
        dict_base_outcomes['lvo_ivt'], dict_base_outcomes['lvo_mt'],
        log_loc=containers['pathway']
        )  # 'Marked unique treatment time pairs where LVO with IVT is better than MT in base scenarios.'

    # ----- Patient population (onion layer) -----
    # Decide the patient population parameters. There are different
    # subgroups of patients in each layer of the SPEEDY onion graph.
    # The layer determines the proportion of patients with each stroke
    # type and the proportions of patients who will be redirected.
    # Inputs:
    # + nLVO / LVO proportions in this subgroup,
    # + nLVO / LVO proportions considered for redirection,
    # + sensitivity / specificity of redirection diagnostic.
    # --- CALCULATIONS:
    # + Unique time results for nLVO + LVO combo for usual care
    #   and for "redirection considered" groups.

    with containers['onion_setup']:
        dict_onion = select_onion_population()
    dict_outcomes = calculate_unique_outcomes_onion(
        dict_base_outcomes, df_base_lvo_ivt_mt,
        log_loc=containers['onion_setup']
    )


    # ----- Region summaries -----
    # Average the results over each geographical region.
    # Find two copies of the results - one with all LSOA in the region
    # and one with only LSOA whose nearest unit is not a CSC.
    # --- CALCULATIONS:
    # + Calculate admissions-weighted average outcomes.
    dict_region_outcomes = calculate_region_outcomes(
        dict_region_unique_times, dict_base_outcomes,
        df_base_lvo_ivt_mt, log_loc=containers['region_summaries']
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
        log_loc=containers['region_summaries']
        )  # 'Gathering mRS distributions for highlighted regions.'
    df_mrs = gather_mrs_for_regions(
        dict_region_outcomes, highlighted_regions,
        log_loc=containers['mrs_dists']
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
        df_lsoa_units_times, log_loc=containers['maps']  # 'Gathering data for maps.'
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
            log_loc=containers['full_results']  # 'Gathering full LSOA-level results.'
            )
    with containers['full_results']:
        st.dataframe(df_full)
