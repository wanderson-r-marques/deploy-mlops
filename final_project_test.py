#!/usr/bin/env python
# coding: utf-8

import requests
'''
cat_cols = ['size_class', 'fire_origin', 'det_agent_type', 'initial_action_by', 'fire_type', 'fire_position_on_slope',
            'weather_conditions_over_fire', 'fuel_type', 'day_period', 'seasons', 'forest_protection_area']

X_cols = ['assessment_hectares', 'current_size', 'size_class', 'fire_location_latitude', 'fire_location_longitude',
            'fire_origin', 'det_agent_type', 'initial_action_by', 'fire_type', 'fire_position_on_slope', 
            'weather_conditions_over_fire', 'fuel_type', 'bh_hectares', 'uc_hectares', 'ex_hectares', 
            'fire_start_day', 'fire_start_month', 'fire_start_year', 'day_period', 'seasons','control_time',
            'extinction_time', 'extinction_delta', 'extinction_efficiency', 'forest_protection_area']
'''
dict_28266 = {
    'assessment_hectares': 0.01, 
    'current_size': 0.01, 
    'size_class': 'A',
    'fire_location_latitude': 56.829483, 
    'fire_location_longitude': -111.5526, 
    'fire_origin': 'Provincial Land',
    'det_agent_type': 'UNP',
    'initial_action_by': 'HAC',
    'fire_type': 'Surface',
    'fire_position_on_slope': 'Flat',
    'weather_conditions_over_fire': 'Clear',
    'fuel_type': 'M2',
    'bh_hectares': 0.01,
    'uc_hectares': 0.01,
    'ex_hectares': 0.01,
    'fire_start_day': 28,
    'fire_start_month': 5,
    'fire_start_year': 2018,
    'day_period': 'afternoon',
    'seasons': 'spring',
    'control_time': 4605,
    'extinction_time': 4638,
    'extinction_delta': 33,
    'extinction_efficiency': 46.38,
    'forest_protection_area': 'Fort McMurray'}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=dict_28266)
print()
print(response.json())
