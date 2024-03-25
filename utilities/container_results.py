"""
All of the content for the Results section.
"""
import numpy as np


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


def make_multiindex_stroke_type(df_columns):
    # Start with 'lvo_ivt_mt' because 'lvo_ivt' starts the same,
    # checking first for 'lvo_ivt' would also pick up 'lvo_ivt_mt'.
    stroke_strings = ['lvo_ivt_mt_', 'nlvo_ivt_', 'lvo_ivt_', 'lvo_mt_']
    new_cols = np.array([[''] * len(df_columns), df_columns])
    for s in stroke_strings:
        cols = np.array(new_cols[1])
        cols_split = [c.split(s) for c in cols]
        for c, col_list in enumerate(cols_split):
            if len(col_list) > 1:
                new_cols[0][c] = s[:-1]  # Remove final underscore
                new_cols[1][c] = col_list[1]
    return new_cols
