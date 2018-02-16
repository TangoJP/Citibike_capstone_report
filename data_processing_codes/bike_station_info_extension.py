# Extract Bike statio information and calculate distances from each station to
# various venues, including parks, schools, theaters, museums and subways

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoSeries, GeoDataFrame
from haversine import haversine

# get the [latitude, longitude] pair via station ID
def get_location_by_ID(data, station_id):
    '''
    Get geolocation of the station by station ID
    '''
    if station_id not in data.st_id.values:
        print('Error: the station ID not found in the database')
    location = data[data.st_id == station_id][['st_latitude', \
                                               'st_longitude']].values
    return tuple(location.flat)

# get the name of the station via station ID
def get_name_by_ID(data, station_id):
    '''
    Get name of the station by station ID
    '''
    if station_id not in data.st_id.values:
        print('Error: the station ID not found in the database')
    name = data[data.st_id == station_id]['name']
    return name

def calculate_minimum_distances2(stations_data, station_id, locations,
            dict_keys=['st_id', 'closest_loc_index', 'closest_distance']):
    '''
    calculate minimum distance from a bike station to a certain location
    '''
    num_rows = len(locations)
    distances = np.zeros(num_rows)
    loc1 = get_location_by_ID(stations_data, station_id)
    for i, loc2 in enumerate(locations):
        dist = haversine(loc1, loc2)
        distances[i] = dist
    min_dist = np.min(distances)
    min_dist_index = np.argmin(distances).astype(int)
    return {dict_keys[0]: station_id, dict_keys[1]: min_dist_index, dict_keys[2]: min_dist}

# Defining methods to calculate the distance to a nearest park
def get_location_by_ID2(data, station_id):
    '''
    Get geolocation of the station by station ID. Revised version
    '''
    if station_id not in data.st_id.values:
        print('Error: the station ID not found in the database')
    location = data[data.st_id == station_id][['st_latitude', 'st_longitude']].values
    return Point(location.flat[1], location.flat[0])

def calculate_distance_closest_park2(st_data, st_id, list_polygon_id, list_polygon):
    '''
    calculated minimum distance from a bike station to a park
    '''
    if len(list_polygon_id) != len(list_polygon):
        print('Error: the length of ID and plolygons don\'t match')

    point = get_location_by_ID2(st_data, st_id)

    dists = np.zeros(len(list_polygon))
    for i, pt in enumerate(list_polygon):
        dists[i] = point.distance(list_polygon.iloc[i])
    ind = np.argmin(dists)

    return {'st_id': st_id, 'polygon_id': list_polygon_id.iloc[ind], 'min_dist': dists[i]}


# Import original bike station data
print('Reading in Bike station information...')
bike_info = pd.read_csv('./data/processed/stations_info_complete.csv')

path = './data/geolocation/'

# Information on the extra-information to be added
additions = {
    'colleges':
        {'file_name': 'colleges.geojson',
         'stem': 'colleges',
         'stem_key': 'college'},
    'subways':
        {'file_name': 'subway_entrances.geojson',
         'stem': 'subways',
         'stem_key': 'subway'},
    'theaters':
        {'file_name': 'theaters.geojson',
         'stem': 'theaters',
         'stem_key': 'theater'},
    'museums':
        {'file_name': 'museums.geojson',
         'stem': 'museums',
         'stem_key': 'museum'}
 }

# iterate over information type
for addition in additions.keys():
    file_name = additions[addition]['file_name']
    file_path = path + file_name
    stem = additions[addition]['stem']
    stem_key = additions[addition]['stem_key']
    stem_name = 'closest_' + stem_key

    # read in geolocation data from the file
    print('Reading in geolocations for ' + stem + '...')
    geo = gpd.read_file(file_path)
    geo_loc_key = stem_key + '_location'
    geo[geo_loc_key] = None
    for ind in geo.index:
        lat = geo.geometry[ind].y
        lng = geo.geometry[ind].x
        geo[geo_loc_key].iloc[ind] = tuple([lat, lng])

    # Calculated minimum distances from a bike station to venues
    print('Appending geolocation information for ' + stem + '...')
    labels1 = ['st_id',
               ('closest_' + stem_key + '_ind'),
               ('closest_' + stem_key +'_distance')]
    distances = []
    for st in bike_info.st_id:
        res = calculate_minimum_distances2(bike_info, st,
                    geo[geo_loc_key], dict_keys=labels1)
        res[stem_name] = geo.iloc[res[labels1[1]]]['name']
        distances.append(res)
    additional_info = pd.DataFrame(distances)
    bike_info = pd.merge(bike_info, additional_info, on='st_id')
    bike_info = bike_info.drop(labels1[1], axis=1)

# Adding Park Information (this had to be done slightly differently
# because parks were not point locations)
# Import parks info into geopandas
parks = gpd.read_file('./data/geolocation/parks.geojson')
parks['source_id'] = parks['source_id'].astype(int)

# Calculate the distances to parks
park_distances = []
for st in bike_info.st_id:
    res = calculate_distance_closest_park2(bike_info, st,
                                            parks.source_id, parks.geometry)
    park_distances.append(res)

# Pretty up the DataFrame
p_dist = pd.DataFrame(park_distances)
p_dist = pd.merge(p_dist, parks[['source_id', 'park_name']],
                    how='left', left_on='polygon_id', right_on='source_id')
p_dist = p_dist.drop(['source_id'], axis=1)
p_dist.columns = ['closest_park_distance', 'park_source_id',
                  'st_id', 'closest_park_name']

# Add the park distances to the main table
bike_info = pd.merge(bike_info, p_dist[['st_id', 'closest_park_name',
                            'closest_park_distance']], on='st_id')

# Save file
print('Saving to a file...')
bike_info = bike_info.sort_values('st_id', ascending=True)
bike_info.to_csv('./data/processed/stations_info_extended_complete.csv',
                 index=False)
print('...PRCESSING COMPLETE')
