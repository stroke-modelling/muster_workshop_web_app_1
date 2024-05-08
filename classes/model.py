"""
Model class
"""

import numpy as np
import pandas as pd

# Imports from the stroke_outcome package:
from stroke_outcome.continuous_outcome import Continuous_outcome
import stroke_outcome.outcome_utilities as outcome_utilities


class Model(object):
    """
    Model class

    Attributes
    ----------
    scenario : dict
        Scenario parameters object.

    geodata : dict
        Pandas DataFrame of geographic data.
    """

    def __init__(self, scenario, geodata):
        """
        Constructor class for Model.
        """

        # Scenario
        self.scenario = scenario

        # Geodata
        self.geodata = geodata

        if self.scenario.limit_to_england:
            mask = self.geodata['England'] == 1
            self.geodata = self.geodata.loc[mask].copy(deep=True)

            # Index shenanigans.
            # I don't know why, but it quietly breaks some of the LSOA
            # when the index is mostly range index integers but has
            # gaps where we've taken out the Welsh LSOA.
            # The affected English LSOAs get all None for their outcomes
            # and show up as blank in any maps.
            self.geodata.index.name = 'rubbish'
            self.geodata = self.geodata.reset_index()
            self.geodata = self.geodata.drop(['rubbish'], axis='columns')

    def run(self):
        """
        Run the model.
        """

        # set up table and add results
        self.full_results = self.geodata.copy(deep=True)

        # Place mRS distributions in here:
        self.full_mrs_dists = self.geodata[
            ['LSOA', 'Admissions']].copy(deep=True)

        self.add_drip_ship()
        self.add_mothership()
        self.add_msu()

        self.add_diff_msu_minus_drip_ship()

        # Make non-cumulative mRS distributions:
        cols = self.full_mrs_dists.columns
        cols = [c for c in cols if c not in ['LSOA', 'Admissions']]
        for c in cols:
            new_col = f'{c}_noncum'
            new_data = self.full_mrs_dists[c]
            # Make one very long Series with one value per row:
            # Explicitly cast to float so that np.round() doesn't
            # throw up a TypeError later.
            new_data = new_data.explode().astype(float)
            # Reshape into one row per LSOA:
            new_data = new_data.values.reshape(len(self.full_mrs_dists[c]), 7)
            # Take the difference between mRS bands:
            new_data = np.diff(new_data, prepend=0.0, axis=1)
            # Round the values:
            new_data = np.round(new_data, 3)
            # Return to a single list per row:
            new_data = new_data.tolist()
            # Store:
            self.full_mrs_dists[new_col] = new_data

        # Reindex on LSOA
        self.full_results.set_index('LSOA', inplace=True)
        self.full_mrs_dists.set_index('LSOA', inplace=True)
        self.summary_results = self.full_results.describe().T

        self.save_results()


    def add_drip_ship(self):
        """Add Drip and ship times to IVT & MT, and clinical benefit"""

        # Set up outcome object
        continuous_outcome = Continuous_outcome()

        # Add drip and ship times
        self.full_results['drip_ship_ivt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_time_ambulance_response +
            self.scenario.process_ambulance_on_scene_duration +
            self.full_results['nearest_ivt_time'] +
            self.scenario.process_time_arrival_to_needle)

        self.full_results['drip_ship_mt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_time_ambulance_response +
            self.scenario.process_ambulance_on_scene_duration +
            self.full_results['nearest_ivt_time'] +
            self.scenario. transfer_time_delay +
            self.full_results['transfer_time'] +
            self.scenario.process_time_transfer_arrival_to_puncture)

        # Add clinical benefit for nLVO outcome (stroke type = 1)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['drip_ship_ivt_time']
        outcome_inputs_df['ivt_chosen_bool'] = 0
        outcome_inputs_df['onset_to_puncture_mins'] = 999999
        outcome_inputs_df['mt_chosen_bool'] = 0
        # Outcome without treatment
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())
        self.full_results['nlvo_no_treatment_mrs_0-2'] = \
            continuous_outcome.mrs_distribution_probs['no_treatment_nlvo'][2]
        self.full_results['nlvo_no_treatment_utility'] = \
            outcomes_by_stroke_type['nlvo_ivt_utility_not_treated']
        self.full_results['lvo_no_treatment_mrs_0-2'] = \
            continuous_outcome.mrs_distribution_probs['no_treatment_lvo'][2]
        self.full_results['lvo_no_treatment_utility'] = \
            outcomes_by_stroke_type['lvo_ivt_utility_not_treated']

        # Outcome with treatment
        outcome_inputs_df['ivt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())
        self.full_results['nlvo_drip_ship_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['nlvo_drip_ship_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_shift']        
        self.full_results['nlvo_drip_ship_ivt_utility'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_post_stroke']
        self.full_results['nlvo_drip_ship_ivt_utility_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_shift']
        # # One list of mRS values per row (patient) in the data.
        col = 'nlvo_ivt_each_patient_mrs_dist_post_stroke'
        outs = outcomes_by_stroke_type[col].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['nlvo_drip_ship_ivt_mrs_dists'] = outs

        # Add clinical benefit for LVO outcome (stroke type = 2)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(2, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['drip_ship_ivt_time']
        outcome_inputs_df['onset_to_puncture_mins'] = self.full_results['drip_ship_mt_time']
        # Outcome with treatment (IVT then IVT+MT)
        outcome_inputs_df['ivt_chosen_bool'] = 1
        outcome_inputs_df['mt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())

        # LVO IVT
        self.full_results['lvo_drip_ship_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_drip_ship_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_shift'] 
        self.full_results['lvo_drip_ship_ivt_utility'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_post_stroke']
        self.full_results['lvo_drip_ship_ivt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_shift']
        # LVO MT
        self.full_results['lvo_drip_ship_mt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_drip_ship_mt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_shift']  
        self.full_results['lvo_drip_ship_mt_utility'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_post_stroke']
        self.full_results['lvo_drip_ship_mt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_shift']
        # LVO IVT + MT
        self.full_results['lvo_drip_ship_ivt_mt_mrs_0-2'] = self.full_results[
            ['lvo_drip_ship_ivt_mrs_0-2', 'lvo_drip_ship_mt_mrs_0-2']].max(axis=1)
        self.full_results['lvo_drip_ship_ivt_mt_mrs_shift'] = self.full_results[
            ['lvo_drip_ship_ivt_mrs_shift', 'lvo_drip_ship_mt_mrs_shift']].min(axis=1)
        self.full_results['lvo_drip_ship_ivt_mt_utility'] = self.full_results[
            ['lvo_drip_ship_ivt_utility', 'lvo_drip_ship_mt_utility']].max(axis=1)
        self.full_results['lvo_drip_ship_ivt_mt_utility_shift'] = self.full_results[
            ['lvo_drip_ship_ivt_utility_shift', 'lvo_drip_ship_mt_utility_shift']].max(axis=1)

        # One list of mRS values per row (patient) in the data.
        outs = outcomes_by_stroke_type[
            'lvo_ivt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_drip_ship_ivt_mrs_dists'] = outs

        outs = outcomes_by_stroke_type[
            'lvo_mt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_drip_ship_mt_mrs_dists'] = outs

        # Pick the treatment that causes more mRS 0-2:
        inds = self.full_results[
            ['lvo_drip_ship_ivt_mrs_0-2', 'lvo_drip_ship_mt_mrs_0-2']].idxmax(axis=1)
        # Pick out IVT and MT results:
        col = 'lvo_ivt_each_patient_mrs_dist_post_stroke'
        outs_ivt = outcomes_by_stroke_type[col].copy()
        col = 'lvo_mt_each_patient_mrs_dist_post_stroke'
        outs_mt = outcomes_by_stroke_type[col].copy()

        # Initially copy over all the IVT data...
        outs = outs_ivt.copy()
        # ... then update with MT data where necessary:
        mask = np.where(inds.str.contains('_mt_'))
        outs[mask, :] = outs_mt[mask, :].copy()
        # Reshape as normal:
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_drip_ship_ivt_mt_mrs_dists'] = outs



    def add_mothership(self):
        """Add Mothership times to IVT & MT, and clinical benefit"""

        # Set up outcome object
        continuous_outcome = Continuous_outcome()

        # Add mothership times
        self.full_results['mothership_ivt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_time_ambulance_response +
            self.scenario.process_ambulance_on_scene_duration +
            self.full_results['nearest_mt_time'] +
            self.scenario.process_time_arrival_to_needle)

        self.full_results['mothership_mt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_time_ambulance_response +
            self.scenario.process_ambulance_on_scene_duration +
            self.full_results['nearest_mt_time'] +
            self.scenario.process_time_arrival_to_puncture)

        # Add clinical benefit for nLVO outcome (stroke type = 1)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['mothership_ivt_time']
        outcome_inputs_df['ivt_chosen_bool'] = 0
        outcome_inputs_df['onset_to_puncture_mins'] = 999999
        outcome_inputs_df['mt_chosen_bool'] = 0
        # Outcome with treatment
        outcome_inputs_df['ivt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())
        self.full_results['nlvo_mothership_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['nlvo_mothership_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_shift']        
        self.full_results['nlvo_mothership_ivt_utility'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_post_stroke']
        self.full_results['nlvo_mothership_ivt_utility_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_shift']
        # # One list of mRS values per row (patient) in the data.
        col = 'nlvo_ivt_each_patient_mrs_dist_post_stroke'
        outs = outcomes_by_stroke_type[col].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['nlvo_mothership_ivt_mrs_dists'] = outs

        # Add clinical benefit for LVO outcome (stroke type = 2)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(2, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['mothership_ivt_time']
        outcome_inputs_df['ivt_chosen_bool'] = 0
        outcome_inputs_df['onset_to_puncture_mins'] = self.full_results['mothership_mt_time']
        outcome_inputs_df['mt_chosen_bool'] = 0
        # Outcome with treatment (IVT then IVT+MT)
        outcome_inputs_df['ivt_chosen_bool'] = 1
        outcome_inputs_df['mt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())

        # LVO IVT
        self.full_results['lvo_mothership_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_mothership_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_shift']  
        self.full_results['lvo_mothership_ivt_utility'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_post_stroke']
        self.full_results['lvo_mothership_ivt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_shift']
        # LVO MT
        self.full_results['lvo_mothership_mt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_mothership_mt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_shift']  
        self.full_results['lvo_mothership_mt_utility'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_post_stroke']
        self.full_results['lvo_mothership_mt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_shift']
        # LVO IVT + MT
        self.full_results['lvo_mothership_ivt_mt_mrs_0-2'] = self.full_results[
            ['lvo_mothership_ivt_mrs_0-2', 'lvo_mothership_mt_mrs_0-2']].max(axis=1)
        self.full_results['lvo_mothership_ivt_mt_mrs_shift'] = self.full_results[
            ['lvo_mothership_ivt_mrs_shift', 'lvo_mothership_mt_mrs_shift']].min(axis=1)
        self.full_results['lvo_mothership_ivt_mt_utility'] = self.full_results[
            ['lvo_mothership_ivt_utility', 'lvo_mothership_mt_utility']].max(axis=1)
        self.full_results['lvo_mothership_ivt_mt_utility_shift'] = self.full_results[
            ['lvo_mothership_ivt_utility_shift', 'lvo_mothership_mt_utility_shift']].max(axis=1)

        # One list of mRS values per row (patient) in the data.
        outs = outcomes_by_stroke_type[
            'lvo_ivt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_mothership_ivt_mrs_dists'] = outs

        outs = outcomes_by_stroke_type[
            'lvo_mt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_mothership_mt_mrs_dists'] = outs

        # Pick the treatment that causes more mRS 0-2:
        inds = self.full_results[
            ['lvo_mothership_ivt_mrs_0-2', 'lvo_mothership_mt_mrs_0-2']].idxmax(axis=1)
        # Pick out IVT and MT results:
        col = 'lvo_ivt_each_patient_mrs_dist_post_stroke'
        outs_ivt = outcomes_by_stroke_type[col].copy()
        col = 'lvo_mt_each_patient_mrs_dist_post_stroke'
        outs_mt = outcomes_by_stroke_type[col].copy()

        # Initially copy over all the IVT data...
        outs = outs_ivt.copy()
        # ... then update with MT data where necessary:
        mask = np.where(inds.str.contains('_mt_'))
        outs[mask, :] = outs_mt[mask, :].copy()
        # Reshape as normal:
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_mothership_ivt_mt_mrs_dists'] = outs

    def add_msu(self):
        """Add MSU times to IVT & MT, clinical benefit, and utilised time"""

        # Set up outcome object
        continuous_outcome = Continuous_outcome()

        # Add MSU times
        self.full_results['msu_ivt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_msu_dispatch +
            (self.full_results['nearest_msu_time'] * self.scenario.scale_msu_travel_times) +
            self.scenario.process_msu_thrombolysis
            )

        self.full_results['msu_mt_time'] = (
            self.scenario.process_time_call_ambulance +
            self.scenario.process_msu_dispatch +
            (self.full_results['nearest_msu_time'] * self.scenario.scale_msu_travel_times) +
            self.scenario.process_msu_thrombolysis +
            self.scenario.process_msu_on_scene_post_thrombolysis +
            (self.full_results['nearest_mt_time']  * self.scenario.scale_msu_travel_times) +
            self.scenario.process_time_msu_arrival_to_puncture
            )

        self.full_results['msu_occupied_treatment'] = (
            self.scenario.process_msu_dispatch +
            (self.full_results['nearest_msu_time'] * self.scenario.scale_msu_travel_times) +
            self.scenario.process_msu_thrombolysis +
            self.scenario.process_msu_on_scene_post_thrombolysis +
            (self.full_results['nearest_mt_time']  * self.scenario.scale_msu_travel_times)
            )

        self.full_results['msu_occupied_no_treatment'] = (
            self.scenario.process_msu_dispatch +
            (self.full_results['nearest_msu_time'] * self.scenario.scale_msu_travel_times) +
            self.scenario.process_msu_on_scene_no_thrombolysis
            )

        # Add clinical benefit for nLVO outcome (stroke type = 1)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(1, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['msu_ivt_time']
        outcome_inputs_df['ivt_chosen_bool'] = 0
        outcome_inputs_df['onset_to_puncture_mins'] = 999999
        outcome_inputs_df['mt_chosen_bool'] = 0
        # Outcome with treatment
        outcome_inputs_df['ivt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())
        self.full_results['nlvo_msu_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['nlvo_msu_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_mrs_shift']        
        self.full_results['nlvo_msu_ivt_utility'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_post_stroke']
        self.full_results['nlvo_msu_ivt_utility_shift'] = \
            outcomes_by_stroke_type['nlvo_ivt_each_patient_utility_shift']
        # # One list of mRS values per row (patient) in the data.
        col = 'nlvo_ivt_each_patient_mrs_dist_post_stroke'
        outs = outcomes_by_stroke_type[col].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['nlvo_msu_ivt_mrs_dists'] = outs

        # Add clinical benefit for LVO outcome (stroke type = 2)
        # Set up input table for stroke outcome package
        outcome_inputs_df = pd.DataFrame()
        outcome_inputs_df['stroke_type_code'] = np.repeat(2, len(self.full_results))
        outcome_inputs_df['onset_to_needle_mins'] = self.full_results['msu_ivt_time']
        outcome_inputs_df['ivt_chosen_bool'] = 0
        outcome_inputs_df['onset_to_puncture_mins'] = self.full_results['msu_mt_time']
        # Outcome with treatment (IVT then IVT+MT)
        outcome_inputs_df['ivt_chosen_bool'] = 1
        outcome_inputs_df['mt_chosen_bool'] = 1
        continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
        patient_data_dict, outcomes_by_stroke_type, full_cohort_outcomes = (
            continuous_outcome.calculate_outcomes())
        # LVO IVT
        self.full_results['lvo_msu_ivt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_msu_ivt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_mrs_shift']  
        self.full_results['lvo_msu_ivt_utility'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_post_stroke']
        self.full_results['lvo_msu_ivt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_ivt_each_patient_utility_shift']
        # LVO MT
        self.full_results['lvo_msu_mt_mrs_0-2'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_dist_post_stroke'][:,2]
        self.full_results['lvo_msu_mt_mrs_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_mrs_shift']  
        self.full_results['lvo_msu_mt_utility'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_post_stroke']
        self.full_results['lvo_msu_mt_utility_shift'] = \
            outcomes_by_stroke_type['lvo_mt_each_patient_utility_shift']
        # LVO IVT + MT
        self.full_results['lvo_msu_ivt_mt_mrs_0-2'] = self.full_results[
            ['lvo_msu_ivt_mrs_0-2', 'lvo_msu_mt_mrs_0-2']].max(axis=1)
        self.full_results['lvo_msu_ivt_mt_mrs_shift'] = self.full_results[
            ['lvo_msu_ivt_mrs_shift', 'lvo_msu_mt_mrs_shift']].min(axis=1)
        self.full_results['lvo_msu_ivt_mt_utility'] = self.full_results[
            ['lvo_msu_ivt_utility', 'lvo_msu_mt_utility']].max(axis=1)
        self.full_results['lvo_msu_ivt_mt_utility_shift'] = self.full_results[
            ['lvo_msu_ivt_utility_shift', 'lvo_msu_mt_utility_shift']].max(axis=1)

        # One list of mRS values per row (patient) in the data.
        outs = outcomes_by_stroke_type[
            'lvo_ivt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_msu_ivt_mrs_dists'] = outs

        outs = outcomes_by_stroke_type[
            'lvo_mt_each_patient_mrs_dist_post_stroke'].copy()
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_msu_mt_mrs_dists'] = outs
    
        # Pick the treatment that causes more mRS 0-2:
        inds = self.full_results[
            ['lvo_msu_ivt_mrs_0-2', 'lvo_msu_mt_mrs_0-2']].idxmax(axis=1)
        # Pick out IVT and MT results:
        col = 'lvo_ivt_each_patient_mrs_dist_post_stroke'
        outs_ivt = outcomes_by_stroke_type[col].copy()
        col = 'lvo_mt_each_patient_mrs_dist_post_stroke'
        outs_mt = outcomes_by_stroke_type[col].copy()

        # Initially copy over all the IVT data...
        outs = outs_ivt.copy()
        # ... then update with MT data where necessary:
        mask = np.where(inds.str.contains('_mt_'))
        outs[mask, :] = outs_mt[mask, :].copy()
        # Reshape as normal:
        outs = np.round(outs, 5).tolist()
        self.full_mrs_dists['lvo_msu_ivt_mt_mrs_dists'] = outs

    def add_diff_msu_minus_drip_ship(self):

        scenario_types = ['msu', 'drip_ship']
        occlusion_types = ['nlvo', 'lvo']
        treatment_types = ['ivt', 'mt', 'ivt_mt']
        outcome_types = ['mrs_0-2', 'mrs_shift', 'utility', 'utility_shift']

        for occ in occlusion_types:
            for tre in treatment_types:
                for out in outcome_types:
                    # Existing column names:
                    col_scen1 = f'{occ}_{scenario_types[0]}_{tre}_{out}'
                    col_scen2 = f'{occ}_{scenario_types[1]}_{tre}_{out}'
                    # New column name for the diff data:
                    col_diff = ''.join([
                        f'{occ}_',
                        f'diff_{scenario_types[0]}_minus_{scenario_types[1]}',
                        f'_{tre}_{out}'
                    ])
                    try:
                        data_scen1 = self.full_results[col_scen1]
                        data_scen2 = self.full_results[col_scen2]
                        data_exists = True
                    except KeyError:
                        # This combination doesn't exist
                        # (e.g. nLVO with MT).
                        data_exists = False

                    if data_exists:
                        data_diff = data_scen1 - data_scen2
                        self.full_results[col_diff] = data_diff
                    else:
                        pass

    def save_results(self):
        """Save results to output folder"""

        if self.scenario.save_lsoa_results:
            self.full_results.to_csv(f'./output/lsoa_results_scen_{self.scenario.name}.csv')
            self.summary_results.to_csv(f'./output/summary_results_scen_{self.scenario.name}.csv')