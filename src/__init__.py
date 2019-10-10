# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:33:52 2019

@author: Jingjing Tang
"""

import os
import numpy as np
import pandas as pd

def check_create_dir( strDir ):

	if not os.path.exists( strDir ):
		os.makedirs( strDir )
	return strDir

def check_file(strPath):
	try:
		with open(strPath):
			s = True
	except IOError:
		s = False
	return s
    
def check_data(strpath):
        
    data = pd.read_csv(strpath,sep = ',')
    data = data.drop_duplicates()
    data.columns = data.columns.str.lower()
    for nec in ['location', 'target', 'score', 'competition_week']:
        if nec not in data.columns:
            return nec
    if 'logscore' in data.columns:
        data['score'] = np.e**data['logscore']
    elif 'score' in data.columns:
        data = data.rename(columns={"score": "logscore"})
        data['score'] = np.e**data['logscore']
    else:
        return -2
    return data

def check_targets(data):
    targets = np.unique(data['target'])
    
    target_key = []
    target_pal = []
    for target in targets:
        target_key.append(target)
        if 'onset' in target:
            target_pal.append('Season onset')
        elif 'pkwk' in target or 'peak week' in target:            
            target_pal.append('Season peak week')
        elif 'pkpr' in target or 'percentage' in target:
            target_pal.append('Season peak percentage')
        elif '1' in target:
            target_pal.append('1 wk ahead')
        elif '2' in target:
            target_pal.append('2 wk ahead')
        elif '3' in target:
            target_pal.append('3 wk ahead')
        elif '4' in target:
            target_pal.append('4 wk ahead')
    target_lut = dict(zip(map(str, target_key), target_pal))
    data['target'] = data['target'].map(target_lut)
    return data

def check_locations(data):
    data['location'] = data['location'].str.lower()
    locations = np.unique(data['location'])
    
    if 'us national' in locations or 'us' in locations:
        datatype = 'Region'
#        loc_order = ['HHS Region %d'%i for i in range(1,11)] + ['US National']
    elif 'overall' in locations:
        datatype = 'Hospitalization'
#        loc_order = ['0-4 yr', '5-17 yr', '18-49 yr', '50-64 yr', '65+ yr', 'Overall']
    else:
        datatype = 'State'
#        loc_order = np.sort(np.unique(df['location']))
    
    location_key = []
    location_pal = []
    
    if datatype == 'Region':
        for loc in locations:
            location_key.append(loc)
            if 'us' in loc:
                location_pal.append('US National')
            elif "1" in loc and '10' not in loc:
                  location_pal.append('HHS Region 1') 
            else:
                for j in range(10, 1, -1):
                    if str(j) in loc:
                        location_pal.append('HHS Region %d'%j)
                        break
        location_lut = dict(zip(map(str, location_key), location_pal))
        data['location'] = data['location'].map(location_lut)
    else:
        data['location'] = data['location'].apply(lambda x: str.capitalize(x))
    
    data = data.sort_values('competition_week', ascending = True)
    
    return datatype

        