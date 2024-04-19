"""
Processing of raw geographic data for model
"""

import pandas as pd


class Geoprocessing(object):
    """
    Processing of raw geographic data for model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise geographic class.


        Attributes
        ----------

        admissions : pandas dataframe
            Raw data of stroke admissions in the UK.

        combined_data : pandas dataframe
            Combined data of all processed geographic data.
        
        hospitals : pandas dataframe
            Raw data of hospitals in the UK.

        inter_hospital_time : pandas dataframe
            Raw data of inter hospital transfer times in the UK.

        lsoa_travel_time : pandas dataframe
            Raw data of travel times between LSOAs and hospitals in the UK.
            
        nearest_ivt_unit : pandas dataframe
            Nearest IVT unit to each LSOA.

        nearest_mt_unit : pandas dataframe
            Nearest MT unit to each LSOA.

        nearest_msu_unit : pandas dataframe
            Nearest MSU unit to each LSOA.

        transfer_mt_unit : pandas dataframe
            Nearest transfer MT unit to each IVT unit.

        Methods
        -------
        collate_data
            Combine data.

        find_nearest_ivt_unit
            Find the nearest IVT unit to each LSOA.

        find_nearest_msu_unit
            Find the nearest MSU unit to each LSOA.

        find_nearest_mt_unit
            Find the nearest MT unit to each LSOA.

        find_nearest_transfer_mt_unit
            Find the nearest transfer MT unit for each IVT unit.

        load_data
            Load raw geographic data.

        run
            Run all processing methods.

        save_processed_data
            Save combined data
                
        """
        # Overwrite default values (can take named arguments or a dictionary)
        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def run(self):
        """
        Run all processing methods.
        """
        self.load_data()
        self.find_nearest_ivt_unit()
        self.find_nearest_mt_unit()
        self.find_nearest_msu_unit()
        self.find_nearest_transfer_mt_unit()
        self.collate_data()
        # self.save_processed_data()

    def collate_data(self):
        """
        Combine data
        """

        self.combined_data = pd.DataFrame()
        self.combined_data['nearest_ivt_unit'] = self.nearest_ivt_unit['unit']
        self.combined_data['nearest_ivt_time'] = self.nearest_ivt_unit['time']
        self.combined_data['nearest_mt_unit'] = self.nearest_mt_unit['unit']
        self.combined_data['nearest_mt_time'] = self.nearest_mt_unit['time']
        self.combined_data = self.combined_data.merge(
            self.transfer_mt_unit, how='left', left_on='nearest_ivt_unit', right_index=True)
        self.combined_data['nearest_msu_unit'] = self.nearest_msu_unit['unit']
        self.combined_data['nearest_msu_time'] = self.nearest_msu_unit['time']
        self.combined_data = self.combined_data.merge(
            self.admissions, how='left', left_index=True, right_index=True)

    def find_nearest_ivt_unit(self):
        """
        Find the nearest IVT unit to each LSOA.
        """

        # Limit data to hospitals providing IVT
        mask = self.hospitals['Use_IVT'] == 1
        self.IVT_hospitals = self.hospitals[mask]['Hospital_name'].tolist()
        # Use only the travel times to IVT hospitals
        travel_matrix = self.lsoa_travel_time[self.IVT_hospitals]
        # Find the value and index for the lowest travel time for each LSOA
        self.nearest_ivt_unit = pd.DataFrame()
        self.nearest_ivt_unit['unit'] = travel_matrix.idxmin(axis=1)
        self.nearest_ivt_unit['time'] = travel_matrix.min(axis=1)

    def find_nearest_msu_unit(self):
        """
        Find the nearest MSU unit to each LSOA.
        """

        # Limit data to hospitals providing MSU
        mask = self.hospitals['Use_MSU'] == 1
        self.MSU_hospitals = self.hospitals[mask]['Hospital_name'].tolist()
        # Use only the travel times to MSU hospitals
        travel_matrix = self.lsoa_travel_time[self.MSU_hospitals]
        # Find the value and index for the lowest travel time for each LSOA
        self.nearest_msu_unit = pd.DataFrame()
        self.nearest_msu_unit['unit'] = travel_matrix.idxmin(axis=1)
        self.nearest_msu_unit['time'] = travel_matrix.min(axis=1)

    def find_nearest_mt_unit(self):
        """
        Find the nearest MT unit to each LSOA.
        """

        # Limit data to hospitals providing MT
        mask = self.hospitals['Use_MT'] == 1
        self.MT_hospitals = self.hospitals[mask]['Hospital_name'].tolist()
        # Use only the travel times to MT hospitals
        travel_matrix = self.lsoa_travel_time[self.MT_hospitals]
        # Find the value and index for the lowest travel time for each LSOA
        self.nearest_mt_unit = pd.DataFrame()
        self.nearest_mt_unit['unit'] = travel_matrix.idxmin(axis=1)
        self.nearest_mt_unit['time'] = travel_matrix.min(axis=1)

    def find_nearest_transfer_mt_unit(self):
        """
        Find the nearest transfer MT unit for each IVT unit
        """

        # Use only the travel times to MT hospitals
        travel_matrix = self.inter_hospital_time[self.MT_hospitals]
        # Find the value and index for the lowest travel time for each LSOA
        self.transfer_mt_unit = pd.DataFrame()
        self.transfer_mt_unit['transfer_unit'] = travel_matrix.idxmin(axis=1)
        self.transfer_mt_unit['transfer_required'] = \
            self.transfer_mt_unit['transfer_unit'] != travel_matrix.index.tolist(
        )
        self.transfer_mt_unit['transfer_time'] = travel_matrix.min(axis=1)

    def load_data(self):
        """
        Load raw geographic data.
        """
        # self.hospitals = pd.read_csv(
        #     './data/stroke_hospitals.csv', index_col='Postcode')

        from stroke_maps.catchment import Catchment
        catchment = Catchment()
        self.hospitals = catchment.get_unit_services()

        # Rename columns to match what the rest of the model here wants.
        self.hospitals.index.name = 'Postcode'
        self.hospitals = self.hospitals.rename(columns={
            'use_ivt': 'Use_IVT',
            'use_mt': 'Use_MT',
            'use_msu': 'Use_MSU',
        })
        self.hospitals['Hospital_name'] = self.hospitals.index.copy()

        if hasattr(self, 'df_unit_services'):
            self.update_unit_services()

        self.admissions = pd.read_csv(
            './data/admissions_2017-2019.csv', index_col='area')
        self.admissions.sort_index(inplace=True)

        self.inter_hospital_time = pd.read_csv(
            './data/inter_hospital_time_calibrated.csv', index_col='from_postcode')

        self.lsoa_travel_time = pd.read_csv(
            './data/lsoa_travel_time_matrix_calibrated.csv', index_col='LSOA')
        self.lsoa_travel_time.sort_index(inplace=True)

    def update_unit_services(self):
        hospitals = self.hospitals
        hospitals = hospitals.reset_index()

        df_unit_services = self.df_unit_services
        cols_new = df_unit_services.columns
        df_unit_services = df_unit_services.reset_index()

        for col_new in cols_new:
            hospitals = pd.merge(
                hospitals, df_unit_services[['Postcode', col_new]],
                left_on='Postcode', right_on='Postcode',
                how='right', suffixes=['_old', None]
            )
        hospitals = hospitals.set_index('Postcode')
        # Remove 'old' columns:
        cols = hospitals.columns
        cols = [c for c in cols if not c.endswith('_old')]
        self.hospitals = hospitals[cols]

    def save_processed_data(self):
        """
        Save combined data
        """

        self.combined_data.to_csv(
            './processed_data/processed_data.csv', index_label='LSOA')
