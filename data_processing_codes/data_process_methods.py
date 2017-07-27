# Custom methods for processing pickup and drop off information as well as
# extra information to the basic ridership data downlowded from the Citibike
# website

import pandas as pd
import numpy as np
import datetime as dt

#data = pd.read_csv(file_name)

def add_date_information(data):
    '''
    Add datetime columns to the data in padans DataFrame format
    '''
    # Add new columns to be used for processing the original table
    data.columns = ['tripduration', 'starttime', 'stoptime',
                    'start station id', 'start station name',
                    'start station latitude', 'start station longitude',
                    'end station id', 'end station name',
                    'end station latitude', 'end station longitude',
                    'bikeid', 'usertype', 'birth year', 'gender']

    data['start_date'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.strftime("%Y%m%d")
    data['start_date_hour'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.strftime("%Y%m%d%H")
    data['start_year'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.year
    data['start_month'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.month
    data['start_day'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.day
    data['start_hour'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.hour
    data['start_dayofweek'] = pd.to_datetime(data['starttime'], \
                        infer_datetime_format=True).dt.dayofweek
    data['end_date'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.strftime("%Y%m%d")
    data['end_date_hour'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.strftime("%Y%m%d%H")
    data['end_year'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.year
    data['end_month'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.month
    data['end_day'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.day
    data['end_hour'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.hour
    data['end_dayofweek'] = pd.to_datetime(data['stoptime'], \
                        infer_datetime_format=True).dt.dayofweek
    return data

def count_pickups_dropoffs(data):
    '''
    Function to count pickups and dropoffs
    '''
    subs = data[data.usertype == 'Subscriber']
    nsubs = data[data.usertype == 'Customer']

    # Get pickups information
    pickups_subs = subs.groupby(['start_date', 'start station id'])\
                                                    .size().reset_index()
    pickups_nsubs = nsubs.groupby(['start_date', 'start station id'])\
                                                    .size().reset_index()
    pickups_subs.columns = ['date', 'st_id', 'pickups']
    pickups_nsubs.columns = ['date', 'st_id', 'pickups']

    # Get dropoffs information
    dropoffs_subs = subs.groupby(['end_date', 'end station id'])\
                                                    .size().reset_index()
    dropoffs_nsubs = nsubs.groupby(['end_date', 'end station id'])\
                                                    .size().reset_index()
    dropoffs_subs.columns = ['date', 'st_id', 'dropoffs']
    dropoffs_nsubs.columns = ['date', 'st_id', 'dropoffs']

    pickups_subs = pickups_subs.astype(int)
    pickups_nsubs = pickups_nsubs.astype(int)
    dropoffs_subs = dropoffs_subs.astype(int)
    dropoffs_nsubs = dropoffs_nsubs.astype(int)

    return [pickups_subs, pickups_nsubs, dropoffs_subs, dropoffs_nsubs]

def add_station_information(data, st_info_file):
    '''
    Function to append bike station information.
    '''
    st_info = pd.read_csv(st_info_file)
    if 'st_id' not in st_info.columns:
        print('st_id does not exit in the station info file')
    if 'st_id' not in data.columns:
        print('st_id does not exit in the data')

    data = pd.merge(data, st_info, how='left', on='st_id')

    return data

def add_weather_information(data, weather_file):
    '''
    Function to append weather information
    '''
    weather = pd.read_csv(weather_file)
    if 'DATE' not in weather.columns:
        print('DATE does not exit in the station info file')
    if 'date' not in data.columns:
        print('date does not exit in the data')
    weather['DATE'] = weather['DATE'].astype(int)
    data = pd.merge(data, weather, how='left', left_on='date', right_on='DATE')

    return data

# This function was never used in the report
def add_holiday_information(data, holiday_file):
    '''
    Append holiday information to the data.
    ***THIS WAS NOT USED IN THE REPORT"""
    '''
    holiday = pd.read_csv(holiday_file)
    if 'date' not in holiday_file.columns:
        print('DATE does not exit in the station info file')
    if 'date' not in data.columns:
        print('date does not exit in the data')

    data = pd.merge(data, holiday, how='left', on='date')

    return data

def add_extra_information(data, st_info_file, weather_file, holiday_file=None):
    '''
    Using the functions above, append a set of extra information to the data
    '''
    data = add_station_information(data, st_info_file)
    data = add_weather_information(data, weather_file)
    if holiday_file:
        data = add_holiday_information(data, holiday_file)
    return data


def count_pickups_dropoffs2(data):
    '''
    Count hourly pickups and dropoffs. Revised version.
    '''
    subs = data[data.usertype == 'Subscriber']
    nsubs = data[data.usertype == 'Customer']

    # Get pickups information
    pickups_subs = subs.groupby(['start_date_hour', 'start station id'])\
                                                    .size().reset_index()
    pickups_nsubs = nsubs.groupby(['start_date_hour', 'start station id'])\
                                                    .size().reset_index()
    pickups_subs.columns = ['date_hour', 'st_id', 'pickups']
    pickups_nsubs.columns = ['date_hour', 'st_id', 'pickups']

    # Get dropoffs information
    dropoffs_subs = subs.groupby(['end_date_hour', 'end station id'])\
                                                    .size().reset_index()
    dropoffs_nsubs = nsubs.groupby(['end_date_hour', 'end station id'])\
                                                    .size().reset_index()
    dropoffs_subs.columns = ['date_hour', 'st_id', 'dropoffs']
    dropoffs_nsubs.columns = ['date_hour', 'st_id', 'dropoffs']

    pickups_subs = pickups_subs.astype(int)
    pickups_nsubs = pickups_nsubs.astype(int)
    dropoffs_subs = dropoffs_subs.astype(int)
    dropoffs_nsubs = dropoffs_nsubs.astype(int)

    return [pickups_subs, pickups_nsubs, dropoffs_subs, dropoffs_nsubs]
