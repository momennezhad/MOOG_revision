"""
URL scraper - At the moment, this scapes the first 5 URLS of the Italy, Germany and USA data
"""

__author__ = "Sang Young Noh (sangyoung123@googlemail.com)"
__version = "1.0.1"
__date__ = "01/05/2018"
__copyright__ = "Copyright (c) 2018 Sang Young Noh """
__license__ = "Python"

# Generic Mathematical imports

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Googlesearch imports
from googlesearch.googlesearch import GoogleSearch
from googlesearch import search

# iterative library imports

from tqdm import tqdm
from itertools import islice
import json

# Read the MOOG data

data = pd.read_csv("MOOG_data.csv")
#interesting_cols = ['Country','Customer','Family','Market','Application Market','Customer Type','Moog Company','Costs (US$)','Sales (US$)','Invoice', 'Date','Focus Market','Business Sector','Business Unit','Sub Family']
#df = data[interesting_cols]

# Separate the data into Italy, USA and Germany customers in MOOG

df_Italy = data.loc[data['Moog Company'] == 'Italy']
df_Italy['Profit'] = df_Italy['Sales (US$)'] - df_Italy['Costs (US$)']

# Germany data                                                                                                                                       
df_Germany = data.loc[data['Moog Company'] == 'Germany']
df_Germany['Profit'] = df_Germany['Sales (US$)'] - df_Germany['Costs (US$)']

# USA/Canada data

df_USA = data.loc[data['Moog Company'] == 'ICD']
df_USA['Profit'] = df_USA['Sales (US$)'] - df_USA['Costs (US$)']

# Sort by ascending valus                                                                                                                            
df_USA = df_USA.sort_values(by = ['Profit'], ascending = False)
df_Italy = df_Italy.sort_values(by = ['Profit'], ascending = False)
df_Germany = df_Germany.sort_values(by = ['Profit'], ascending = False)

# Clip any -ve profits to 0                                                                                                                          
df_USA['Profit'] = (df_USA['Profit']).clip_lower(0)
df_Germany['Profit'] = (df_Germany['Profit']).clip_lower(0)
df_Italy['Profit'] = (df_Italy['Profit']).clip_lower(0)

# Remove any values that are 0                                                                                                                       
df_Italy = df_Italy[df_Italy['Profit'] > 0]
df_Germany = df_Germany[df_Germany['Profit'] > 0]
df_USA = df_USA[df_USA['Profit'] > 0]

USA_Company_list = []
for i, j in enumerate(df_USA['Customer']):
    A = j.split(" ")[:-1]
    A = " ".join(A) 
    USA_Company_list.append(A)

Italy_Company_list = []
for i, j in enumerate(df_Italy['Customer']):
    A = j.split(" ")[:-1]
    A = " ".join(A) 
    Italy_Company_list.append(A)
     
Germany_Company_list = []
for i, j in enumerate(df_Germany['Customer']):
    A = j.split(" ")[:-1]
    A = " ".join(A) 
    Germany_Company_list.append(A)

# Assign classes in the case for USA_Company_list

def profit_class_USA(val):

    if val > df_USA['Profit'].quantile(0.75):
        return 1
    elif val > df_USA['Profit'].quantile(0.5) and val < df_USA['Profit'].quantile(0.75):
        return 2
    elif val > df_USA['Profit'].quantile(0.25) and val < df_USA['Profit'].quantile(0.5):
        return 3 
    elif val < df_USA['Profit'].quantile(0.25):
        return 4

# Assign classes in the case for Germany_Company_list

def profit_class_GER(val):

    if val > df_Germany['Profit'].quantile(0.75):
        return 1
    elif val > df_Germany['Profit'].quantile(0.5) and val < df_Germany['Profit'].quantile(0.75):
        return 2
    elif val > df_Germany['Profit'].quantile(0.25) and val < df_Germany['Profit'].quantile(0.5):
        return 3 
    elif val < df_Germany['Profit'].quantile(0.25):
        return 4

# Assign classes in the case for Italy_Company_list
    
def profit_class_ITALY(val):
    if val > df_Italy['Profit'].quantile(0.75):
        return 1
    elif val > df_Italy['Profit'].quantile(0.5) and val < df_Italy['Profit'].quantile(0.75):
        return 2
    elif val > df_Italy['Profit'].quantile(0.25) and val < df_Italy['Profit'].quantile(0.5):
        return 3 
    elif val < df_Italy['Profit'].quantile(0.25):
        return 4
    
# Assign four classes for classification based on quartile profit data
    
df_USA['class'] = df_USA['Profit'].apply(profit_class_USA)    
df_Germany['class'] = df_Germany['Profit'].apply(profit_class_Germany)    
df_Italy['class'] = df_Italy['Profit'].apply(profit_class_Italy)    

# Write seperate dicts to json

USA_class_dict = dict(zip(df_USA['Customer'], df_USA['class']))
Germany_class_dict = dict(zip(df_Germany['Customer'], df_Germany['class']))
Italy_class_dict = dict(zip(df_Italy['Customer'], df_Italy['class']))

with open('USA_class.json', 'w') as fp:
    json.dump(USA_class_dict, fp)

with open('Germany_class.json', 'w') as fp:
    json.dump(Germany_class_dict, fp)

with open('Italy_class.json', 'w') as fp:
    json.dump(Italy_class_dict, fp)

# Isolate unique names 
    
ger = set(Germany_Company_list)
it = set(Italy_Company_list)
US = set(USA_Company_list)
     
#link_df = pd.read_csv("all_urls_with_url_analysis.csv")
#Scraping_company_search_list = set(Company_list)    

# --------------------------------------------------- #

company_url_links = {}

iterator = islice(it,0,800,2)

for link in tqdm(iterator):
    #print link
    company_URL_list = []
    link = str(link)
    for url in search(link, stop = 5):
     #   print link, url
     #   company_url_top_20_links[link] = []
     #   company_url_top_20_links[link].append(url) 
         company_URL_list.append(url)
         company_url_links[link] = company_URL_list
         print link, url
                
with open('it.json', 'w') as fp:
        json.dump(company_url_links, fp)
    
      
