"""
Class to run multiple scenarios
"""

import itertools
import pandas as pd
import time

from classes.model import Model
from classes.scenario import Scenario

class Scenario_runner(object):
    """
    Scenario runner
    """

    def __init__(self, scenarios, prefix=''):
        """constructor class"""

        self.scenarios = scenarios
        self.results = pd.DataFrame(); self.results.index.name = 'Scenario'
        self.scenario_values = dict()
        self.prefix = prefix

    def run(self):
        """Run through scenarios"""

        time_start = time.time()

        # Generate all scenarios:
        all_scenarios_tuples = [
            x for x in itertools.product(*self.scenarios.values())]
        # Convert list of tuples back to list of dictionaries:
        all_scenarios_dicts = [
            dict(zip(self.scenarios.keys(), p)) for p in all_scenarios_tuples]
        
        # Convert all_scenarios_dicts into a DataFrame for saving
        self.all_scenarios = pd.DataFrame.from_dict(all_scenarios_dicts)
        self.all_scenarios.index.name = 'Scenario'
        self.all_scenarios.to_csv(f'./output/scenario_list_{self.prefix}.csv')

        # Run all scenarios
        for index, scenario_to_run in enumerate(all_scenarios_dicts):
            # Estimate time remaining (minutes)
            time_elapsed = time.time() - time_start
            time_per_scenario = time_elapsed / (index + 1)
            time_remaining = time_per_scenario * (len(all_scenarios_dicts) - index)
            
            # Show progress (overwrites previous line)
            print(f'Running scenario {index+1}/{len(all_scenarios_dicts)}. Time remaining: {time_remaining/60:.2f} minutes', end='\r')

            scenario_to_run['name'] = f'{self.prefix}_{index}'
            # Set up model
            model = Model(
                scenario=Scenario(scenario_to_run),
                geodata=pd.read_csv('processed_data/processed_data.csv'))
            model.run()

            # Store results
            self.results[index] = model.summary_results['mean']

            # Delete model to free up memory
            del model

        # Save results
        self.results = self.results.T.round(5)
        # self.results.to_csv(f'./output/scenario_results_{self.prefix}.csv')