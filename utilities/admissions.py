"""
Calculate and display admissions changes.
"""
import pandas as pd
import streamlit as st


def calculate_region_admissions_onion(
        s_admissions_psc: pd.Series,
        s_admissions_all: pd.Series,
        df_onion_pops: pd.DataFrame,
        ):
    """
    How many patients are in each onion layer in this region?

    Make sure input admissions are for the full population so that
    they can be scaled down to each onion layer.

    Inputs:
    s_admissions_psc - pd.Series. Full population admissions for
                         areas nearest a non-MT unit.
    s_admissions_all - pd.Series. Full population admissions.
    df_onion_pops    - pd.DataFrame. Onion layer data including
                       display label and proportion of all stroke.

    Returns
    -------
    df_admissions_onion - pd.DataFrame. Admission numbers for each
                          onion layer, split by nearest unit type.
    """
    # Calculate some admissions numbers for the full stroke population
    # (not the selected onion layer):
    admissions_nearest_no_mt = s_admissions_psc.sum()
    admissions = s_admissions_all.sum()
    admissions_nearest_mt = admissions - admissions_nearest_no_mt
    # Store results in here:
    cols = ['admissions', 'admissions_nearest_csc', 'admissions_nearest_atc']
    df_admissions_onion = pd.DataFrame(
        index=df_onion_pops['label'].values, columns=cols)
    # Calculate admissions for each onion layer:
    for onion_label in df_onion_pops['population'].values:
        # Pick out data for this layer:
        s = df_onion_pops.loc[df_onion_pops['population'] == onion_label]
        label = s['label'].values[0]
        prop = s['prop_of_all_stroke'].values[0]
        # Scale full admissions by this onion layer's proportion:
        a = admissions * prop
        a_csc = admissions_nearest_mt * prop
        a_atc = admissions_nearest_no_mt * prop
        # Store:
        df_admissions_onion.loc[label] = [a, a_csc, a_atc]
    return df_admissions_onion


def calculate_network_usual_care(
        df_network: pd.DataFrame, dict_pops_u: dict, mt_units: list):
    """
    Calculate tracked admissions across units in usual care.

    Inputs
    ------
    df_network  - pd.DataFrame. How many patients go to each
                  combination of stroke units for this region?
    dict_pops_u - dict. Proportions of patients with each stroke type
                  and treatment combination.

    Returns
    -------
    df_net_u - pd.DataFrame. Tracked numbers of patients at each
               combination of first and transfer units, split by
               how many patients receive thrombectomy.
    """
    prop_mt_usual_care = (
        dict_pops_u
        .loc[['lvo_mt', 'lvo_ivt_mt'], 'full_population'].sum()
        )
    # Usual care:
    df_net_u = df_network.copy()
    # Convert unit columns to generic strings:
    cols_units = [c for c in df_net_u.columns if 'unit' in c]
    df_net_u[cols_units] = df_net_u[cols_units].astype('string')
    # Combine columns that share nearest and transfer units:
    df_net_u = df_net_u.drop('nearest_mt_unit', axis='columns')
    df_net_u = df_net_u.groupby(['nearest_ivt_unit', 'transfer_unit']
                                ).sum().reset_index()
    df_net_u.insert(0, 'nearest_unit', df_net_u['nearest_ivt_unit'])
    df_net_u = df_net_u.rename(columns={'nearest_ivt_unit': 'first_unit'})
    df_net_u['admissions_catchment_to_first_unit'] = (
        df_net_u['admissions'].copy())
    # Only include transfers for patients who receive MT
    # and who aren't already at an MT unit.
    df_net_u['admissions_first_unit_to_transfer'] = (
        df_net_u['admissions'].copy() * prop_mt_usual_care)
    df_net_u['thrombectomy'] = (
        df_net_u['admissions_first_unit_to_transfer'].copy())
    mask_no_transfer = (df_net_u['first_unit'] == df_net_u['transfer_unit'])
    df_net_u.loc[mask_no_transfer, 'admissions_first_unit_to_transfer'] = 0.0
    df_net_u['nearest_unit_postcode'] = df_net_u['nearest_unit'].astype(str)
    df_net_u['nearest_unit'] = (
        'nearest_' + df_net_u['nearest_unit'].astype(str))
    # Flag whether chosen units offer MT:
    df_net_u['nearest_unit_has_mt'] = (
        df_net_u['nearest_unit_postcode'].isin(mt_units))
    df_net_u['first_unit_has_mt'] = (
        df_net_u['first_unit'].isin(mt_units))
    return df_net_u


def calculate_network_redir(
        df_network: pd.DataFrame, dict_pops_r: dict, mt_units: list):
    """
    Calculate tracked admissions across units in redirection scenario.

    Inputs
    ------
    df_network  - pd.DataFrame. How many patients go to each
                  combination of stroke units for this region?
    dict_pops_r - dict. Proportions of patients with each stroke type
                  and treatment combination.

    Returns
    -------
    df_net_r - pd.DataFrame. Tracked numbers of patients at each
               combination of first, redirected, and transfer units,
               split by how many patients receive thrombectomy.
    """
    # Pick out proportions:
    m = dict_pops_r['scenario'] != 'redir_accepted'
    n = dict_pops_r['scenario'] == 'redir_accepted'
    cols_mt = ['lvo_mt', 'lvo_ivt_mt']

    prop_no_redir = dict_pops_r[m]['full_population'].sum()
    prop_mt_redir_approved = (
        dict_pops_r[n].loc[cols_mt, 'full_population'].sum()
        ) / dict_pops_r[n]['full_population'].sum()
    prop_mt_redir_not_approved = (
        dict_pops_r[m].loc[cols_mt, 'full_population'].sum()
        ) / dict_pops_r[m]['full_population'].sum()

    # Set up results df:
    df_net_r = df_network.copy()
    # Convert unit columns to generic strings:
    cols_units = [c for c in df_net_r.columns if 'unit' in c]
    df_net_r[cols_units] = df_net_r[cols_units].astype('string')

    # No redir, similar to usual care above:
    df_net_no_redir = df_net_r.copy()
    df_net_no_redir['admissions'] *= prop_no_redir
    # Combine columns that share nearest and transfer units:
    df_net_no_redir = df_net_no_redir.drop('nearest_mt_unit', axis='columns')
    df_net_no_redir = df_net_no_redir.groupby(
        ['nearest_ivt_unit', 'transfer_unit']).sum().reset_index()
    df_net_no_redir.insert(0, 'nearest_unit',
                           df_net_no_redir['nearest_ivt_unit'])
    df_net_no_redir = df_net_no_redir.rename(
        columns={'nearest_ivt_unit': 'first_unit'})
    df_net_no_redir['admissions_catchment_to_first_unit'] = (
        df_net_no_redir['admissions'].copy())
    # Only include transfers for patients who receive MT
    # and who aren't already at an MT unit.
    df_net_no_redir['admissions_first_unit_to_transfer'] = (
        df_net_no_redir['admissions'] * prop_mt_redir_not_approved)
    df_net_no_redir['thrombectomy'] = (
        df_net_no_redir['admissions_first_unit_to_transfer'])
    mask_no_transfer = (df_net_no_redir['first_unit'] ==
                        df_net_no_redir['transfer_unit'])
    df_net_no_redir.loc[mask_no_transfer,
                        'admissions_first_unit_to_transfer'] = 0.0
    df_net_no_redir['redirected'] = 0

    # Redirected:
    df_net_redir = df_net_r.copy()
    df_net_redir['admissions'] *= (1.0 - prop_no_redir)
    df_net_redir['admissions_catchment_to_first_unit'] = (
        df_net_redir['admissions'].copy())
    df_net_redir.insert(0, 'nearest_unit',
                        df_net_redir['nearest_ivt_unit'])
    df_net_redir = df_net_redir.rename(
        columns={'nearest_mt_unit': 'first_unit'})
    df_net_redir = df_net_redir.drop('transfer_unit', axis='columns')
    df_net_redir = df_net_redir.drop('nearest_ivt_unit', axis='columns')
    df_net_redir['thrombectomy'] = (
        df_net_redir['admissions_catchment_to_first_unit'] *
        prop_mt_redir_approved
        )
    df_net_redir['transfer_unit'] = df_net_redir['first_unit']
    df_net_redir['admissions_first_unit_to_transfer'] = 0.0
    df_net_redir['redirected'] = 1

    # Combine redir and no redir:
    df_net_r = pd.concat((df_net_redir, df_net_no_redir), axis='rows')
    # Combine data for patients whose nearest unit is MT:
    df_net_r = df_net_r.groupby(
        ['nearest_unit', 'first_unit', 'transfer_unit']).sum().reset_index()
    df_net_r['nearest_unit_postcode'] = df_net_r['nearest_unit'].astype(str)
    df_net_r['nearest_unit'] = (
        'nearest_' + df_net_r['nearest_unit'].astype(str))
    # Flag whether chosen units offer MT:
    df_net_r['nearest_unit_has_mt'] = (
        df_net_r['nearest_unit_postcode'].isin(mt_units))
    df_net_r['first_unit_has_mt'] = (
        df_net_r['first_unit'].isin(mt_units))
    return df_net_r


def gather_network_units(df_net_u: pd.DataFrame, df_net_r: pd.DataFrame):
    """
    Find stroke units that exist in the networks.

    Inputs
    ------
    df_net_u - pd.DataFrame. Admissions network for usual care.
    df_net_r - pd.DataFrame. Admissions network for redirection.

    Returns
    dict_units - dict. Lists of stroke units in these networks.
                 Subsets for all units, nearest units, and MT units.
    """
    # Gather units in the network:
    cols_units = ['first_unit', 'transfer_unit']
    all_units = sorted(list(
        set(df_net_u[cols_units].values.flatten()) |
        set(df_net_r[cols_units].values.flatten())
    ))
    # Only units whose catchment area is in the selected region:
    nearest_units = sorted(list(
        set(df_net_u['nearest_unit'].values) |
        set(df_net_r['nearest_unit'].values)
    ))
    # Only MT units that can catch patients in the selected region:
    mt_units = sorted(list(
        set(df_net_u['transfer_unit'].values) |
        set(df_net_r['transfer_unit'].values)
    ))

    dict_units = {
        'all': all_units,
        'nearest': nearest_units,
        'mt': mt_units,
    }
    return dict_units


def convert_network_to_generic(df_net):
    """
    Calculate admissions for the flowchart.
    Squash the network dfs by unit into generic unit type.

    Combine all admissions for:
    + nearest unit no MT, first unit no MT
    + nearest unit no MT, first unit MT
    + nearest unit MT, first unit no MT (shouldn't happen)
    + nearest unit MT, first unit MT

    Calculate admissions separately for all patients,
    patients who receive MT, and patients with no MT.
    """
    # Drop columns that shouldn't be summed (e.g. postcodes):
    cols_to_drop = ['nearest_unit', 'first_unit', 'transfer_unit',
                    'nearest_unit_postcode']
    df_net = df_net.drop(cols_to_drop, axis='columns')
    # Sum all admissions from the same groups:
    cols_to_group = ['nearest_unit_has_mt', 'first_unit_has_mt']
    df_net = df_net.groupby(cols_to_group).sum().reset_index()
    # Mark whether these patients were redirected:
    df_net['redirected'] = (df_net['nearest_unit_has_mt'] !=
                            df_net['first_unit_has_mt'])
    # Split off patients who did not receive MT:
    df_net['admissions_catchment_to_first_unit_no_mt'] = (
        df_net['admissions_catchment_to_first_unit'] -
        df_net['thrombectomy']
    )
    return df_net


def calculate_region_admissions_generic(df_region_admissions_generic):
    """
    Input rows: first_csc, first_atc, transfer
    Input columns: (mt/no_mt)_(usual_care/redir)_(nearest_atc/nearest_csc)

    Target keys: mt_usual_care, no_mt_usual_care, mt_redir, no_mt_redir.
    Each value is a df with rows: nearest_csc, nearest_atc
    and columns: first_csc, first_atc, transfer.
    """
    keys = ['mt_usual_care', 'no_mt_usual_care', 'mt_redir', 'no_mt_redir']
    # Store results in here:
    d = {}
    for k in keys:
        cols = [c for c in df_region_admissions_generic.columns
                if c.startswith(k)]
        df_here = df_region_admissions_generic[cols].copy()
        rename_dict = dict([(c, c.replace(f'{k}_', '')) for c in cols])
        df_here = df_here.rename(columns=rename_dict)
        d[k] = df_here.transpose().fillna(0.0)
    return d


def gather_region_admissions_generic(df_net_u_gen, df_net_r_gen):
    """
    Flatten the multiple generic admissions into one dataframe.

    Useful columns in input df:
    + admissions_catchment_to_first_unit_no_mt
    + thrombectomy (=admissions_catchment_to_first_unit_mt)
    + admissions_first_unit_to_transfer

    Target column names:
    + (mt/no_mt)_(usual_care/redir)_(nearest_csc/nearest_atc)
    Target rows: first_csc, first_atc, transfer.
    Each cell has the admissions for that category.

    Inputs
    ------


    Returns
    -------
    df - pd.DataFrame. The same data as the input dict of dfs
         but flattened out for displaying on the app.
    """
    # Lookup for input dataframes:
    net_lookup = {'usual_care': df_net_u_gen, 'redir': df_net_r_gen}
    nearest_mt_labels = ['nearest_atc', 'nearest_csc']
    first_unit_labels = ['first_atc', 'first_csc']
    receive_mt_dict = {
        'no_mt': 'admissions_catchment_to_first_unit_no_mt',
        'mt': 'thrombectomy'
    }
    # Store results in here:
    df_all = pd.DataFrame(index=['first_csc', 'first_atc', 'transfer'])
    for scen, df_net in net_lookup.items():
        for nearest_mt, nearest_mt_label in enumerate(nearest_mt_labels):
            for first_unit, first_unit_label in enumerate(first_unit_labels):
                df_here = df_net[(
                    (df_net['nearest_unit_has_mt'] == nearest_mt) &
                    (df_net['first_unit_has_mt'] == first_unit)
                )]
                for receive_mt, col in receive_mt_dict.items():
                    new_col = f'{receive_mt}_{scen}_{nearest_mt_label}'
                    v = df_here[col].values[0] if len(df_here) > 0 else 0.0
                    df_all.loc[first_unit_label, new_col] = v
                    m_transfer = ((first_unit == 0) & (nearest_mt == 0) &
                                    (receive_mt == 'mt'))
                    if m_transfer:
                        col = 'admissions_first_unit_to_transfer'
                        v = df_here[col].values[0] if len(df_here) > 0 else 0.0
                        df_all.loc['transfer', new_col] = v
    df_all = df_all.fillna(0.0)
    return df_all


def select_mt_unit_here(df_unit_services, dict_network_units):
    """
    """
    dict_mt_labels = (
        df_unit_services.loc[dict_network_units['mt'], 'ssnap_name']
        .copy().to_dict()
    )
    # Sort by label:
    dict_mt_labels = {'all': 'All in this region'} | dict(sorted(
        dict_mt_labels.items(), key=lambda item: item[1]))

    def f_mt_label(lookup):
        """Display layer with nice name instead of key."""
        return dict_mt_labels[lookup]
    mt_unit_here = st.selectbox(
        'MT unit to show in the flowcharts',
        dict_mt_labels.keys(),
        format_func=f_mt_label
    )
    return mt_unit_here
