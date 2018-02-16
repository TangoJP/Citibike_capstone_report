# Process and combine all the downloaded data into a few datasets
# containing pickups and dropoffs for subscribers and non-subscribers.
# date, bike station ID and pickups or dropoffs were included so that the
# data batch to process is relatively small. The extended information will be
# appended from a separate file containing station information, etc

import pandas as pd
import numpy as np
import os
import time
from data_process_methods import (add_date_information,
                            count_pickups_dropoffs,
                            count_pickups_dropoffs2,
                            add_station_information,
                            add_weather_information)

# Start timing the process
t0 = time.time()

# Set up variables containing strings for iterating over files
file_dir = './data/downloaded/'
file_year = ['2013','2014', '2015', '2016','2017']
file_months = ['01', '02', '03', '04', '05', '06', \
               '07', '08', '09', '10', '11', '12']
file_end = '_citibike-tripdata.csv'

# iteration paramters set
counts = np.zeros(4)
total_iteration = len(file_year) * len(file_months)
iteration = 1

# Iterate over years
for i in range(len(file_year)):
    # create DFs for pickups and dropoffs for subscribers and non-subscribers
    pickups_subs = pd.DataFrame(columns=['date', 'st_id', 'pickups'])
    pickups_nsubs = pd.DataFrame(columns=['date', 'st_id', 'pickups'])
    dropoffs_subs = pd.DataFrame(columns=['date', 'st_id', 'dropoffs'])
    dropoffs_nsubs = pd.DataFrame(columns=['date', 'st_id', 'dropoffs'])

    # Same thing for hourly data
    pickups_subs_hourly = pd.DataFrame(columns=['date_hour', 'st_id', 'pickups'])
    pickups_nsubs_hourly = pd.DataFrame(columns=['date_hour', 'st_id', 'pickups'])
    dropoffs_subs_hourly = pd.DataFrame(columns=['date_hour', 'st_id', 'dropoffs'])
    dropoffs_nsubs_hourly = pd.DataFrame(columns=['date_hour', 'st_id', 'dropoffs'])

    # Iterate over months
    for j in range(len(file_months)):
        # Get file name and read in the data
        file_name = file_dir + file_year[i] + file_months[j] + file_end
        if os.path.isfile(file_name) == False:
            print(file_year[i] + file_months[j] + file_end + ' does not exist')
            iteration += 1
        else:
            data = pd.read_csv(file_name)
            print('%.1fs: Processing %d/%d, %s...' \
                         % ((time.time() - t0),
                            iteration,
                            total_iteration,
                            (file_year[i] + file_months[j] + file_end)))
            print('\tReading %s...' % file_name)

            # Use custom functions to append extra information to usage data
            print('\tAdding date information...')
            data = add_date_information(data)

            print('\tSlicing data...')
            [p_s, p_ns, d_s, d_ns] = count_pickups_dropoffs(data)
            [p_s_hourly, p_ns_hourly, d_s_hourly, d_ns_hourly] = \
                                     count_pickups_dropoffs2(data)

            print('\tAppending data...')
            pickups_subs = pd.concat([pickups_subs, p_s], axis=0)
            pickups_nsubs = pd.concat([pickups_nsubs, p_ns], axis=0)
            dropoffs_subs = pd.concat([dropoffs_subs, d_s], axis=0)
            dropoffs_nsubs = pd.concat([dropoffs_nsubs, d_ns], axis=0)

            pickups_subs_hourly = \
                    pd.concat([pickups_subs_hourly, p_s_hourly], axis=0)
            pickups_nsubs_hourly = \
                    pd.concat([pickups_nsubs_hourly, p_ns_hourly], axis=0)
            dropoffs_subs_hourly = \
                    pd.concat([dropoffs_subs_hourly, d_s_hourly], axis=0)
            dropoffs_nsubs_hourly = \
                    pd.concat([dropoffs_nsubs_hourly, d_ns_hourly], axis=0)

            iteration += 1

    # save the data for each year
    savepath = './data/processed/test2/' + file_year[i]
    pickups_subs.to_csv(savepath + '_p_s.csv')
    pickups_nsubs.to_csv(savepath + '_p_ns.csv')
    dropoffs_subs.to_csv(savepath + '_d_s.csv')
    dropoffs_nsubs.to_csv(savepath + '_d_ns.csv')

    pickups_subs_hourly.to_csv(savepath + '_p_s_hourly.csv')
    pickups_nsubs_hourly.to_csv(savepath + '_p_ns_hourly.csv')
    dropoffs_subs_hourly.to_csv(savepath + '_d_s_hourly.csv')
    dropoffs_nsubs_hourly.to_csv(savepath + '_d_ns_hourly.csv')

    print('files for ' + file_year[i] + ' saved')

print('p_subs size vs counts: % d' % len(pickups_subs))
print('p_nsubs size vs counts: % d' % len(pickups_nsubs))
print('d_subs size vs counts: % d' % len(pickups_subs))
print('d_nsubs size vs counts: % d' % len(pickups_nsubs))

print('p_subs_hourly size vs counts: % d' % len(pickups_subs))
print('p_nsubs_hourly size vs counts: % d' % len(pickups_nsubs))
print('d_subs_hourly size vs counts: % d' % len(pickups_subs))
print('d_nsubs_hourly size vs counts: % d' % len(pickups_nsubs))

print('Time of Processing: %.1fs' % (time.time() - t0))
