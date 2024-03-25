"""
All of the content for the Results section.
"""


def split_results_dict_by_pathway(results_dict):
    # Split results by pathway:
    results_drip_ship = dict([(key, value) for key, value in results_dict.items()
                            if 'drip_ship' in key])
    results_mothership = dict([(key, value) for key, value in results_dict.items()
                            if 'mothership' in key])
    results_msu = dict([(key, value) for key, value in results_dict.items()
                        if 'msu' in key])
    # Everything else:
    results_all_keys = results_dict.keys()
    results_all_keys -= results_drip_ship.keys()
    results_all_keys -= results_mothership.keys()
    results_all_keys -= results_msu.keys()
    results_all = {key: results_dict[key] for key in results_all_keys}

    # Remove the pathway name from the dict keys.
    # Assume that keys never end with pathway name so there is always
    # a 'pathway_' in the key somewhere.
    def rename_keys(dict, str_to_replace, replace_with=''):
        # Keep a copy of the original keys to prevent RuntimeError
        # when the keys are updated during the loop.
        keys = list(dict.keys())
        for key in keys:
            new_key = key.replace(str_to_replace, replace_with)
            dict[new_key] = dict.pop(key)
        return dict

    results_drip_ship = rename_keys(results_drip_ship, 'drip_ship_')
    results_mothership = rename_keys(results_mothership, 'mothership_')
    results_msu = rename_keys(results_msu, 'msu_')

    new_dict = {
        'all': results_all,
        'drip_ship': results_drip_ship,
        'mothership': results_mothership,
        'msu': results_msu,
    }

    return new_dict
