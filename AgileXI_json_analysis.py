__author__ = "Sang Young Noh (sangyoung123@googlemail.com)"
__version = "0.0.1"
__date__ = "08/03/2018"
__copyright__ = "Copyright (c) 2018 Sang Young Noh """
__license__ = "Python"


"""
In this file, we are looking for URL links scraped from scrapy that may not be 
coordinated with the company name - for example, 'Mitsubishi' may have a facebook
link corresponding to it that has been scraped, which may add a bias to the weighting 
we will implement to the series of strings we will parse.

Hence, this code acts as the pipeline to get rid of all the excess codes in that regard
"""
import json
import numpy as np
from difflib import SequenceMatcher # String sequence matcher - used for matching up string similarities. 

def scrapedJSONreader(json_file)
    """
    Function for reading the Scraped URLs in JSON format 
    """ 
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data 

"""
Function checks for similarity between the company name and the URLs 
"""    
def similar(a,b):
    return SequenceMatcher(None, a, b).ratio()

class 


data_domain = {}
data_score = {}
data_URL = {}

for key in data.keys():
     domain_list = []
     score_list = []
     for URL in data[key]: # Searching through the 
        #print key, URL.split("/")[2].split(".")[1:] # print the URL and the name of the URL 
        domain_list.append(".".join(str(x) for x in URL.split("/")[2].split(".")[1:]))
        print URL.split(".")[1], key 
        score_list.append(similar(URL.split(".")[1], key.lower()))
        print URL.split(".")[1], key, similar(URL.split(".")[1].lower(), key.lower()), URL.split("/")[2].split(".")[1:]
        data_domain[key] = domain_list
        data_score[key] = score_list
        
A = [] 
B = []
C = []
for key in data_domain.keys():
   for index, val in enumerate(data_domain[key]):
       #print key, index, data[key][index], data_score[key][index]
       if data_score[key][index] >= 0.4:
          A.append(data[key][index])
          B.append(data_domain[key][index])
          C.append(key)

domain_country = []                          
for link in B:
    domain_country.append(".".split(link)[-1])
    #print link
    
Test_URL_domain_list_scored = zip(C,A,B,domain_country) 
Test_URL_domain_list_scored_np = np.array(Test_URL_domain_list_scored)
    
dict_final = {}
for i,j in enumerate(Test_URL_domain_list_scored_np[:,0]):
    url_domain = []
    #print Test_URL_domain_list_scored[i][2]
    if len(Test_URL_domain_list_scored[i][2]) > 3: # Get rid of all erroneous domains
       if Test_URL_domain_list_scored[i][-2].split(".")[len(Test_URL_domain_list_scored[i][-2].split("."))-1] == 'com':
           url_domain = [Test_URL_domain_list_scored[i][1], Test_URL_domain_list_scored[i][2]]
           dict_final[Test_URL_domain_list_scored[i][0]] = url_domain 

            
 #Test_URL_domain_list_scored[i][-2].split(".")[len(Test_URL_domain_list_scored[i][-2].split("."))-1]
  #  if Test_URL_domain_list_scored[i][-2].split(".")[len(Test_URL_domain_list_scored[i][-2].split("."))-1] == 'de':
  #      url_domain = [Test_URL_domain_list_scored[i][1], Test_URL_domain_list_scored[i][2]]
  #      dict_final[Test_URL_domain_list_scored[i][0]] = url_domain 

for key,value in  dict_final.iteritems():
    print key, value



with open('test_data.json','w') as fp:
    json.dump(dict_final,fp)
