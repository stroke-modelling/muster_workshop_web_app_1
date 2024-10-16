"""
Model class
"""

import numpy as np
import pandas as pd

# Imports from the stroke_outcome package:
from stroke_outcome.continuous_outcome import Continuous_outcome


def run(geodata, dict_outcome_inputs):
    """
    Run the model.
    """

    # set up table and add results
    full_results = geodata.copy()  # deep=True)

    # Place mRS distributions in here:
    full_mrs_dists = geodata[
        ['LSOA', 'Admissions']].copy()  # deep=True)

    for scen, df_outcome_inputs in dict_outcome_inputs.items():
        # Calculate results:
        scenario_results, scenario_mrs = add_scenario(
            df_outcome_inputs, scen)
        # scenario_mrs = add_scenario(
        #     df_outcome_inputs, scen)
        # scenario_results = add_scenario(
        #     df_outcome_inputs, scen)
        # Merge into full results dataframes:
        full_results = pd.merge(
            full_results, scenario_results,
            left_on='LSOA', right_on='LSOA', how='left',
            suffixes=['', f'_{scen}'])
        full_mrs_dists = pd.merge(
            full_mrs_dists, scenario_mrs,
            left_on='LSOA', right_on='LSOA', how='left',
            suffixes=['', f'_{scen}'])

    full_mrs_dists = convert_mrs_dists_to_noncum(full_mrs_dists)

    # Reindex on LSOA
    full_results.set_index('LSOA', inplace=True)
    full_mrs_dists.set_index('LSOA', inplace=True)
    # summary_results = full_results.describe().T
    return full_results, full_mrs_dists


def add_scenario(outcome_inputs_df, scenario_name='scen'):
    """
    Add Drip and ship times to IVT & MT, and clinical benefit


    Define column name variables to make the lines shorter.
    Used to also find 'utility' outcomes.
    """
    outcomes_by_stroke_type = run_outcome_model(outcome_inputs_df)

    mask_mt_better = find_patients_with_mt_better_than_ivt(
        outcomes_by_stroke_type,
        occ='lvo',
        )

    scenario_results = gather_outcome_results(
        outcome_inputs_df,
        outcomes_by_stroke_type,
        scenario_name,
        mask_mt_better,
        )
    scenario_mrs_dists = gather_mrs_dist_results(
        outcome_inputs_df,
        outcomes_by_stroke_type,
        scenario_name,
        mask_mt_better
        )
    return scenario_results, scenario_mrs_dists


def run_outcome_model(outcome_inputs_df):
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

    outcomes_by_stroke_type = outcomes_by_stroke_type.copy()

    del continuous_outcome

    return outcomes_by_stroke_type


def find_patients_with_mt_better_than_ivt(
        scenario_results,
        occ='lvo',
        ):
    # The results for IVT & MT use whichever is the better of
    # the separate IVT-only and MT-only results.
    # Which is better may be different for each patient.
    # Here, define "better" as more patients in the mRS<=2 band.
    # Convert input dict into DataFrame:
    cols = [
        f'{occ}_ivt_each_patient_mrs_0-2',
        f'{occ}_mt_each_patient_mrs_0-2'
        ]
    scenario_results = pd.DataFrame(
        np.array([scenario_results[c] for c in cols]).T,
        columns=cols
        )
    # Find which is better:
    inds = scenario_results[cols].idxmax(axis=1)
    # "inds" has one value per patient and the value is either
    # the string (not the mRS value, not the index)
    # f'{occ}_ivt_mrs_0-2' or
    # f'{occ}_mt_mrs_0-2' depending.

    # Patients where MT is better than IVT:
    mask = np.where(inds.str.contains('_mt_'))
    return mask


def gather_outcome_results(
        outcome_inputs_df,
        outcomes_by_stroke_type,
        scenario_name,
        mask_mt_better,
        ):
    # Store results in here. Outcomes
    scenario_results = pd.DataFrame()
    # Include LSOA names:
    scenario_results['LSOA'] = outcome_inputs_df['LSOA']
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

    cols_outcomes_before = []
    cols_outcomes_after = []
    for occ in ['lvo', 'nlvo']:
        for tre in ['ivt', 'mt']:
            if occ == 'nlvo' and tre == 'mt':
                pass
            else:
                cols_outcomes_before += [
                    f'{occ}_{tre}_each_patient_mrs_0-2',
                    f'{occ}_{tre}_each_patient_mrs_shift',
                    f'{occ}_{tre}_each_patient_utility_shift'
                ]
                cols_outcomes_after += [
                    f'{scenario_name}_{occ}_{tre}_mrs_0-2',
                    f'{scenario_name}_{occ}_{tre}_mrs_shift',
                    f'{scenario_name}_{occ}_{tre}_utility_shift'
                ]

    for c, cb in enumerate(cols_outcomes_before):
        scenario_results[cols_outcomes_after[c]] = outcomes_by_stroke_type[cb]

    # --- IVT & MT results ---
    # Only do the following for LVO:
    occ = 'lvo'
    for out in ['mrs_0-2', 'mrs_shift', 'utility_shift']:
        c1 = f'{scenario_name}_{occ}_ivt_{out}'
        c2 = f'{scenario_name}_{occ}_mt_{out}'
        outs_ivt = scenario_results[c1].values.copy()
        outs_mt = scenario_results[c2].values.copy()
        # Initially copy over all the IVT data,
        # then update with MT data where necessary:
        outs = outs_ivt.copy()
        outs[mask_mt_better] = outs_mt[mask_mt_better].copy()
        # Place into the results df:
        c3 = f'{scenario_name}_{occ}_ivt_mt_{out}'
        scenario_results[c3] = outs

    # Sort columns alphabetically to group similar results:
    scenario_results = scenario_results[
        sorted(scenario_results.columns.tolist())]
    return scenario_results


def gather_mrs_dist_results(
        outcome_inputs_df,
        outcomes_by_stroke_type,
        scenario_name,
        mask_mt_better
        ):
    # Store results in here. mRS distributions:
    scenario_mrs_dists = pd.DataFrame()
    # Include LSOA names:
    scenario_mrs_dists['LSOA'] = outcome_inputs_df['LSOA']

    cols_mrs_before = []
    cols_mrs_after = []
    for occ in ['lvo', 'nlvo']:
        for tre in ['ivt', 'mt']:
            if occ == 'nlvo' and tre == 'mt':
                pass
            else:
                # One list of mRS values per row (patient) in the data.
                # Give each mRS band its own column...
                c1 = [f'{scenario_name}_{occ}_{tre}_mrs_dists_{i}'
                        for i in range(7)]
                cols_mrs_after.append(c1)
                # ... but in the original data it's just one column:
                c2 = f'{occ}_{tre}_each_patient_mrs_dist_post_stroke'
                cols_mrs_before.append(c2)
                # Round the mRS values to remove unnecessary precision:
                outs = outcomes_by_stroke_type[c2].copy()
                outs = np.round(outs, 5).tolist()
                scenario_mrs_dists[c1] = outs

    # --- IVT & MT results ---
    # Only do the following for LVO:
    occ = 'lvo'
    # Pick out IVT and MT results:
    c1 = f'{occ}_ivt_each_patient_mrs_dist_post_stroke'
    c2 = f'{occ}_mt_each_patient_mrs_dist_post_stroke'
    outs_ivt = outcomes_by_stroke_type[c1].copy()
    outs_mt = outcomes_by_stroke_type[c2].copy()
    # Initially copy over all the IVT data,
    # then update with MT data where necessary:
    outs = outs_ivt.copy()
    outs[mask_mt_better, :] = outs_mt[mask_mt_better, :].copy()
    # Reshape as normal:
    outs = np.round(outs, 5).tolist()
    # Place each mRS band in its own column in the new results:
    cols_mrs = [f'{scenario_name}_{occ}_ivt_mt_mrs_dists_{i}'
                for i in range(7)]
    scenario_mrs_dists[cols_mrs] = outs

    # Sort columns alphabetically to group similar results:
    scenario_mrs_dists = scenario_mrs_dists[
        sorted(scenario_mrs_dists.columns.tolist())]

    return scenario_mrs_dists


def convert_mrs_dists_to_noncum(full_mrs_dists):
    # Make non-cumulative mRS distributions:
    cols = full_mrs_dists.columns.values
    cols = sorted(list(set(
        ['_'.join(c.split('_')[:-1]) for c in cols
            if c not in ['LSOA', 'Admissions']]
            )))
    for c in cols:
        cols_cumsum = [f'{c}_{i}' for i in range(7)]
        cols_noncum = [f'{c}_noncum_{i}' for i in range(7)]

        new_data = full_mrs_dists[cols_cumsum]
        # Take the difference between mRS bands:
        new_data = np.diff(new_data, prepend=0.0, axis=1)
        # Round the values:
        new_data = np.round(new_data, 3)
        # Drop the cumulative data:
        full_mrs_dists = full_mrs_dists.drop(
            cols_cumsum, axis='columns'
        )
        # Store:
        full_mrs_dists[cols_noncum] = new_data
    return full_mrs_dists
