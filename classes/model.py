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

    def __init__(self, dict_outcome_inputs, geodata):
        """
        Constructor class for Model.
        """
        self.dict_outcome_inputs = dict_outcome_inputs
        self.geodata = geodata

    def run(self):
        """
        Run the model.
        """

        # set up table and add results
        self.full_results = self.geodata.copy(deep=True)

        # Place mRS distributions in here:
        self.full_mrs_dists = self.geodata[
            ['LSOA', 'Admissions']].copy(deep=True)

        for scen, df_outcome_inputs in self.dict_outcome_inputs.items():
            scenario_results, scenario_mrs = self.add_scenario(
                df_outcome_inputs, scen)
            self.full_results = pd.merge(
                self.full_results, scenario_results,
                left_on='LSOA', right_on='LSOA', how='left')
            self.full_mrs_dists = pd.merge(
                self.full_mrs_dists, scenario_mrs,
                left_on='LSOA', right_on='LSOA', how='left')

        # Make non-cumulative mRS distributions:
        cols = self.full_mrs_dists.columns.values
        cols = sorted(list(set(
            ['_'.join(c.split('_')[:-1]) for c in cols
             if c not in ['LSOA', 'Admissions']]
             )))
        for c in cols:
            cols_cumsum = [f'{c}_{i}' for i in range(7)]
            cols_noncum = [f'{c}_noncum_{i}' for i in range(7)]

            new_data = self.full_mrs_dists[cols_cumsum]
            # Take the difference between mRS bands:
            new_data = np.diff(new_data, prepend=0.0, axis=1)
            # Round the values:
            new_data = np.round(new_data, 3)
            # Store:
            self.full_mrs_dists[cols_noncum] = new_data

        # Reindex on LSOA
        self.full_results.set_index('LSOA', inplace=True)
        self.full_mrs_dists.set_index('LSOA', inplace=True)
        self.summary_results = self.full_results.describe().T

        # self.save_results()

    def add_scenario(self, outcome_inputs_df, scenario_name='scen'):
        """
        Add Drip and ship times to IVT & MT, and clinical benefit


        Define column name variables to make the lines shorter.
        Used to also find 'utility' outcomes.
        """

        # Set up outcome object
        continuous_outcome = Continuous_outcome()

        # Run the outcome model twice - once with only nLVO patients
        # and once with only LVO patients. Take the relevant columns
        # of each set of results and store them in one master df.
        outcomes_by_stroke_type = {}
        outcome_inputs_df = outcome_inputs_df.copy()
        for occ_code, occ in enumerate(['nlvo', 'lvo']):
            # Stroke type code is 1 for nLVO, 2 for LVO.
            outcome_inputs_df['stroke_type_code'] = occ_code + 1
            # If any patients have nLVO and yes to MT, the outcome
            # model returns NaN. So overwrite the values here:
            if occ == 'nlvo':
                outcome_inputs_df['mt_chosen_bool'] = 0
            else:
                outcome_inputs_df['mt_chosen_bool'] = 1
            continuous_outcome.assign_patients_to_trial(outcome_inputs_df)
            patient_data_dict, outcomes_dict, full_cohort_outcomes = (
                continuous_outcome.calculate_outcomes())
            # Columns for this stroke type:
            if occ == 'nlvo':
                cols = [c for c in list(outcomes_dict.keys())
                        if occ in c]
            else:
                cols = [c for c in list(outcomes_dict.keys())
                        if ((occ in c) & ('nlvo' not in c))]
            # Store a copy of the results for this stroke type:
            for c in cols:
                outcomes_by_stroke_type[c] = outcomes_dict[c]
        # # Are there any columns that do not contain 'lvo'?
        # cols = [c for c in list(outcomes_dict.keys()) if 'lvo' not in c]
        # for c in cols:
        #     outcomes_by_stroke_type[c] = outcomes_dict[c]

        # Add new columns for the proportion with mRS<=2:
        for occ in ['lvo', 'nlvo']:
            for tre in ['ivt', 'mt']:
                try:
                    c1 = f'{occ}_{tre}_each_patient_mrs_0-2'
                    c2 = f'{occ}_{tre}_each_patient_mrs_dist_post_stroke'
                    outcomes_by_stroke_type[c1] = (
                        outcomes_by_stroke_type[c2][:, 2])
                except KeyError:
                    pass

        # Store results in here. Outcomes...
        scenario_results = pd.DataFrame()
        # ... and mRS distributions:
        scenario_mrs_dists = pd.DataFrame()
        # Include LSOA names:
        scenario_results['LSOA'] = outcome_inputs_df['LSOA']
        scenario_mrs_dists['LSOA'] = outcome_inputs_df['LSOA']
        # Include treatment times:
        scenario_results[f'{scenario_name}_ivt_time'] = (
            outcome_inputs_df['onset_to_needle_mins'])
        scenario_results[f'{scenario_name}_mt_time'] = (
            outcome_inputs_df['onset_to_puncture_mins'])
        # Include any bonus columns:
        bonus_cols = outcome_inputs_df.columns
        cols_to_remove = [
            'LSOA', 'onset_to_needle_mins', 'onset_to_puncture_mins',
            'stroke_type_code', 'ivt_chosen_bool', 'mt_chosen_bool'
            ]
        bonus_cols = [c for c in bonus_cols if c not in cols_to_remove]
        scenario_results[bonus_cols] = outcome_inputs_df[bonus_cols]

        cols_before = []
        cols_after = []
        for occ in ['lvo', 'nlvo']:
            for tre in ['ivt', 'mt']:
                if occ == 'nlvo' and tre == 'mt':
                    pass
                else:
                    # --- Outcome results ---
                    cols_before += [
                        f'{occ}_{tre}_each_patient_mrs_0-2',
                        f'{occ}_{tre}_each_patient_mrs_shift',
                        f'{occ}_{tre}_each_patient_utility_shift'
                    ]
                    cols_after += [
                        f'{scenario_name}_{occ}_{tre}_mrs_0-2',
                        f'{scenario_name}_{occ}_{tre}_mrs_shift',
                        f'{scenario_name}_{occ}_{tre}_utility_shift'
                    ]

                    # --- mRS distributions ---
                    # One list of mRS values per row (patient) in the data.
                    # Give each mRS band its own column...
                    c1 = [f'{scenario_name}_{occ}_{tre}_mrs_dists_{i}'
                          for i in range(7)]
                    # ... but in the original data it's just one column:
                    c2 = f'{occ}_{tre}_each_patient_mrs_dist_post_stroke'
                    # Round the mRS values to remove unnecessary precision:
                    outs = outcomes_by_stroke_type[c2].copy()
                    outs = np.round(outs, 5).tolist()
                    scenario_mrs_dists[c1] = outs

        for c, cb in enumerate(cols_before):
            scenario_results[cols_after[c]] = outcomes_by_stroke_type[cb]

        # --- IVT & MT results ---
        for occ in ['lvo']:
            # The results for IVT & MT use whichever is the better of
            # the separate IVT-only and MT-only results.
            # Which is better may be different for each patient.
            # Here, define "better" as more patients in the mRS<=2 band.
            # Find which is better:
            inds = scenario_results[
                [f'{scenario_name}_{occ}_ivt_mrs_0-2',
                 f'{scenario_name}_{occ}_mt_mrs_0-2'
                 ]].idxmax(axis=1)
            # "inds" has one value per patient and the value is either
            # the string (not the mRS value, not the index)
            # f'{scenario_name}_{occ}_ivt_mrs_0-2' or
            # f'{scenario_name}_{occ}_mt_mrs_0-2' depending.

            # Patients where MT is better than IVT:
            mask = np.where(inds.str.contains('_mt_'))

            # --- Outcome results ---
            for out in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
                c1 = f'{scenario_name}_{occ}_ivt_{out}'
                c2 = f'{scenario_name}_{occ}_mt_{out}'
                outs_ivt = scenario_results[c1].values.copy()
                outs_mt = scenario_results[c2].values.copy()
                # Initially copy over all the IVT data,
                # then update with MT data where necessary:
                outs = outs_ivt.copy()
                outs[mask] = outs_mt[mask].copy()
                # Place into the results df:
                c3 = f'{scenario_name}_{occ}_ivt_mt_{out}'
                scenario_results[c3] = outs

            # --- mRS distributions ---
            # Pick out IVT and MT results:
            c1 = f'{occ}_ivt_each_patient_mrs_dist_post_stroke'
            c2 = f'{occ}_mt_each_patient_mrs_dist_post_stroke'
            outs_ivt = outcomes_by_stroke_type[c1].copy()
            outs_mt = outcomes_by_stroke_type[c2].copy()
            # Initially copy over all the IVT data,
            # then update with MT data where necessary:
            outs = outs_ivt.copy()
            outs[mask, :] = outs_mt[mask, :].copy()
            # Reshape as normal:
            outs = np.round(outs, 5).tolist()
            # Place each mRS band in its own column in the new results:
            cols_mrs = [f'{scenario_name}_{occ}_ivt_mt_mrs_dists_{i}'
                        for i in range(7)]
            scenario_mrs_dists[cols_mrs] = outs

        # Sort columns alphabetically to group similar results:
        scenario_mrs_dists = scenario_mrs_dists[
            sorted(scenario_mrs_dists.columns.tolist())]
        scenario_results = scenario_results[
            sorted(scenario_results.columns.tolist())]

        return scenario_results, scenario_mrs_dists

    def save_results(self):
        """Save results to output folder"""

        if self.scenario.save_lsoa_results:
            self.full_results.to_csv(f'./output/lsoa_results_scen_{self.scenario.name}.csv')
            self.summary_results.to_csv(f'./output/summary_results_scen_{self.scenario.name}.csv')