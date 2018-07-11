"""
Implementation of the TF-IDF algorithm to the  
"""

__author__ = "Sang Young Noh (sangyoung123@googlemail.com)"
__version = "1.0.1"
__date__ = "30/04/2018"
__copyright__ = "Copyright (c) 2018 Sang Young Noh """
__license__ = "Python"

import pandas as pd
import numpy as np
import json

from difflib import SequenceMatcher
from tqdm import tqdm

# NLTK library part

from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
from collections import defaultdict
import math

# Countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Pipeline module

from sklearn.pipeline import Pipeline
from sklearn import metrics

# Classifiers

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Cross validation

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Language detection

from langdetect import detect

# Defining the stopwords 

stop_words = stopwords.words('english') + stopwords.words('italian') + stopwords.words('german') + list(punctuation) + list('0123456789')

class Scraping_TFIDF(object):
    """
    """
    def __init__(self):
        self.IT_Company_list = [] 
        self.USA_Company_list = []
        self.GER_Company_list = []
        
        # Classifications loaded 
        self.GER_class = json.load(open('Germany_class.json'))
        self.IT_class = json.load(open('Italy_class.json'))
        self.USA_class = json.load(open('USA_class.json'))

        self.USA_Total_Company_Data = {}
        self.GER_Total_Company_Data = {}
        self.IT_Total_Company_Data = {}

    def TFIDF_similarity(self,string1,string2):
        return SequenceMatcher(None, string1, string2).ratio()

    def TFIDF_load_result(self):
        """
        Load the json files that has the scraped data
        """ 
        GER_data = json.load(open('ger_result.json'))
        USA_data = json.load(open('USA_result.json'))
        IT_data = json.load(open('it_result.json'))

    def TFIDF_load_company_list(self):
        """
        Load the company list data
        """
        self.IT_list = json.load(open('it.json'))
        self.GER_list = json.load(open('ger.json'))
        self.USA_list = json.load(open('USA.json'))


    def TFIDF_hash_table(self):    
        for keys, value in self.IT_list.iteritems():
            self.IT_Company_list.append(keys)
        # Ditto for Germany
        for keys, value in self.GER_list.iteritems():
            self.GER_Company_list.append(keys)

        for keys, value in self.USA_list.iteritems():
            self.US_Company_list.append(keys)

    def TFIDF_reallocate_class(self):
        self.GER_class_final = {} 
        self.IT_class_final = {}
        self.USA_class_final = {}

        for key, item in self.GER_class.iteritems():
            newkey =  key.split(" ")[:-1]
            newkey = " ".join(newkey)
            self.GER_class_final[newkey] = item

        # Reallocating the Italian data

        for key, item in self.IT_class.iteritems():
            newkey =  key.split(" ")[:-1]
            newkey = " ".join(newkey)
            self.IT_class_final[newkey] = item

        # Reallocating the USA data 

        for key, item in self.USA_class.iteritems():
            newkey =  key.split(" ")[:-1]
            newkey = " ".join(newkey)
            self.USA_class_final[newkey] = item

    def TFIDF_final_reallocation(self):
        for company_names in Germany_Company_list:
            Company_array = []
            for dat in ger_data:
                URL_TEXT_data = [] 
                if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
                URL_TEXT_data = dat['TEXT'] # Append the text 
                # Need to put in an additional filter here
                token_URL = word_tokenize(URL_TEXT_data)
                long_words = [w for w in token_URL if len(w) > 10]
                if not long_words:
                    pass
                else:
                    long_words = " ".join(long_words)
                    Company_array.append(long_words) # Append to the greater array
             
            GER_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 


        for company_names in USA_Company_list:
            Company_array = []
            for dat in USA_data:
                URL_TEXT_data = [] 
                if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
                    URL_TEXT_data = dat['TEXT'] # Append the text 
                    # Need to put in an additional filter here
                    token_URL = word_tokenize(URL_TEXT_data)
                    long_words = [w for w in token_URL if len(w) > 10]
                    if not long_words:
                        pass
                    else:
                        long_words = " ".join(long_words)
                        Company_array.append(long_words) # Append to the greater array
             
            USA_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 


        for company_names in Italy_Company_list:
            Company_array = []
            for dat in Italy_data:
                URL_TEXT_data = [] 
                if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
                    URL_TEXT_data = dat['TEXT'] # Append the text 
                    # Need to put in an additional filter here
                    token_URL = word_tokenize(URL_TEXT_data)
                    long_words = [w for w in token_URL if len(w) > 10]
                    if not long_words:
                        pass
                    else:
                        long_words = " ".join(long_words)
                    Company_array.append(long_words) # Append to the greater array
             
        IT_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 



# JSON files including the scraped results

ger_data = json.load(open('ger_result.json'))
USA_data = json.load(open('USA_result.json'))
Italy_data = json.load(open('it_result.json'))


# JSON files including the company names

Italy_list = json.load(open('it.json'))
Germany_list = json.load(open('ger.json'))
USA_list = json.load(open('USA.json'))

# Arrays including the company abbreviations
Italy_Company_list = [] 
USA_Company_list = []
Germany_Company_list = []

# Include in italy array the company names from which
# we can scan the data

for keys, value in Italy_list.iteritems():
    Italy_Company_list.append(keys)

# Ditto for Germany
    
for keys, value in Germany_list.iteritems():
    Germany_Company_list.append(keys)

# Ditto for the USA

for keys, value in USA_list.iteritems():
    USA_Company_list.append(keys)


USA_Total_Company_Data = {}
GER_Total_Company_Data = {}
IT_Total_Company_Data = {}

# The classes each of the companies belong to in each country


ger_class_final = {} 
it_class_final = {}
USA_class_final = {}

# Reallocating the German data

for key, item in ger_class.iteritems():
    newkey =  key.split(" ")[:-1]
    newkey = " ".join(newkey)
    ger_class_final[newkey] = item

# Reallocating the Italian data

for key, item in it_class.iteritems():
    newkey =  key.split(" ")[:-1]
    newkey = " ".join(newkey)
    it_class_final[newkey] = item

# Reallocating the USA data 

for key, item in USA_class.iteritems():
    newkey =  key.split(" ")[:-1]
    newkey = " ".join(newkey)
    USA_class_final[newkey] = item

# Implementing TFIDF on the USA data

class_list = []

for company_names in Germany_Company_list:
    Company_array = []
    for dat in ger_data:
         URL_TEXT_data = [] 
         if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
           URL_TEXT_data = dat['TEXT'] # Append the text 
           # Need to put in an additional filter here
           token_URL = word_tokenize(URL_TEXT_data)
           long_words = [w for w in token_URL if len(w) > 10]
           if not long_words:
               pass
           else:
              long_words = " ".join(long_words)
              Company_array.append(long_words) # Append to the greater array
             
    GER_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 


for company_names in USA_Company_list:
    Company_array = []
    for dat in USA_data:
         URL_TEXT_data = [] 
         if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
           URL_TEXT_data = dat['TEXT'] # Append the text 
           # Need to put in an additional filter here
           token_URL = word_tokenize(URL_TEXT_data)
           long_words = [w for w in token_URL if len(w) > 10]
           if not long_words:
               pass
           else:
              long_words = " ".join(long_words)
              Company_array.append(long_words) # Append to the greater array
             
    USA_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 


for company_names in Italy_Company_list:
    Company_array = []
    for dat in Italy_data:
         URL_TEXT_data = [] 
         if similar(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
           URL_TEXT_data = dat['TEXT'] # Append the text 
           # Need to put in an additional filter here
           token_URL = word_tokenize(URL_TEXT_data)
           long_words = [w for w in token_URL if len(w) > 10]
           if not long_words:
               pass
           else:
              long_words = " ".join(long_words)
              Company_array.append(long_words) # Append to the greater array
             
    IT_Total_Company_Data[company_names] = Company_array # assign dict to each company/URL 

    
train_key_data = []
train_class_data = []

for company in GER_Total_Company_Data.keys():
    for line in GER_Total_Company_Data[company]:
        train_key_data.append(line)
        train_class_data.append(ger_class_final[company])


# English data 

ENG_trans = []
ENG_LAB = []
ENG_TEXT = [] 

for key in USA_Total_Company_Data.keys():
    for line in USA_Total_Company_Data[key]:
        ENG_trans.append('eng')
        ENG_LAB.append(USA_class_final[key])
        ENG_TEXT.append(line)

# Germany data 
GER_trans = []
GER_LAB = []
GER_TEXT = [] 

for key in GER_Total_Company_Data.keys():
    for line in GER_Total_Company_Data[key]:
        GER_trans.append(detect(line))
        GER_LAB.append(ger_class_final[key])
        GER_TEXT.append(line)
        
# Italy data
IT_trans = []
IT_LAB = []
IT_TEXT = [] 
for key in IT_Total_Company_Data.keys():
    for line in IT_Total_Company_Data[key]:
        IT_trans.append(detect(line))
        IT_LAB.append(it_class_final[key])
        IT_TEXT.append(line)

for i in ENG_TEXT:
    print "English " + " " + i

for i in GER_TEXT:
    print "German " + " " + i

for i in IT_TEXT:
    print "Italian " + " " + i

    
        
# Allocate all the scraped data into the company, with the company abbreviation as the
# dict keys - Join all the strings into a single entry

GER_collated_data = {}
ENG_collated_data = {}
IT_collated_data = {}

for name in GER_Total_Company_Data.keys():  
    joined_text =[]
    for dat in GER_Total_Company_Data[name]:
        joined_text.append(dat)
    joined_text = '\n'.join(joined_text)
    GER_collated_data[name] = joined_text

for name in USA_Total_Company_Data.keys():  
    joined_text =[]
    for dat in USA_Total_Company_Data[name]:
        joined_text.append(dat)
    joined_text = '\n'.join(joined_text)
    ENG_collated_data[name] = joined_text

for name in IT_Total_Company_Data.keys():  
    joined_text =[]
    for dat in IT_Total_Company_Data[name]:
        joined_text.append(dat)
    joined_text = '\n'.join(joined_text)
    IT_collated_data[name] = joined_text

    
# Get rid of dictionary entries where we don't get any entries because of lack of string consistency 

ENG_collated_data = dict((k, v) for k, v in ENG_collated_data.iteritems() if v)
GER_collated_data = dict((k, v) for k, v in GER_collated_data.iteritems() if v)
IT_collated_data = dict((k, v) for k, v in IT_collated_data.iteritems() if v)

new = [] # Need to change the name of this - new is not specific enough 

# Listing the scraped data into a separate list, excluding
# empty dict entries

##for key, data in collated_data.iteritems():
#   new.append(collated_data[key])
 
#for key, data in collated_data.iteritems():
#    class_list.append(ger_class_final[key])

    
# Naive Bayes approach pipeline

text_clf_Bayes = Pipeline([('vect', CountVectorizer(stop_words=stop_words)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])

# SVM approach pipeline

text_clf_SVM_ENG = Pipeline([('vect', CountVectorizer(stop_words = stop_words)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=1000, learning_rate=0.5)),
])
text_clf_SVM_GER = Pipeline([('vect', CountVectorizer(stop_words = stop_words)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=1000, learning_rate=0.5)),
])



# Random Tree approach pipeline

# text_clf_RTree = Pipeline([('vect', CountVectorizer(stop_words=stop_words)),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', RandomForestClassifier(max_depth=10, max_features=4)),
#])

# Gradient boosting
# XGboost

# 3 clasifiers
X_train_ENG, X_test_ENG, y_train_ENG, y_test_ENG = train_test_split(ENG_TEXT, ENG_LAB, test_size=0.33, random_state=42)
text_clf_SVM_ENG.fit(X_train_ENG, y_train_ENG)

X_train_GER, X_test_GER, y_train_GER, y_test_GER = train_test_split(GER_TEXT, GER_LAB, test_size=0.33, random_state=42)
text_clf_SVM_GER.fit(X_train_GER, y_train_GER)

X_train_IT, X_test_IT, y_train_IT, y_test_IT = train_test_split(IT_TEXT, IT_LAB, test_size=0.33, random_state=42)
text_clf_SVM_IT.fit(X_train_IT, y_train_IT)

# Initial prediction 
#predicted_SVM = text_clf_SVM.predict(X_test)
#print (np.mean(predicted_SVM == y_test))

#### TEST DATA ### 
#
test_dat = json.load(open('test_result.json'))
test_company_dat = pd.read_excel('Copy of VDMA List for Scoring 200318 final.xlsx')
test_company_list = [] 

for name in test_company_dat['Company Name']:
    test_company_list.append(name)
                           
TOTAL_COMPANY_DATA = {}


for company in test_company_list:
    Company_array = []
    for line in test_dat:
        if similar(line['URL'].split('.')[1].lower(), company.lower()) > 0.5:
            URL_TEXT_data = line['TEXT']
            token_URL = word_tokenize(URL_TEXT_data)
            long_words = [w for w in token_URL if len(w) > 10]
            if not long_words:
               pass
            else:
                long_words = " ".join(long_words)
                Company_array.append(long_words) # Append to the greater array 
    TOTAL_COMPANY_DATA[company] = Company_array
     
TOTAL_COMPANY_DATA = dict((k, v) for k, v in TOTAL_COMPANY_DATA.iteritems() if v)
test_key_data = []
class_data = []


USA_test_dat = json.load(open('final_result.json'))
USA_test_company_dat = pd.read_excel('US_Target_List 200318.xls')
USA_test_company_list = [] 

for name in USA_test_company_dat['Company Name']:
    USA_test_company_list.append(name)
                           
USA_TOTAL_COMPANY_DATA = {}


for company in USA_test_company_list:
    Company_array = []
    for line in USA_test_dat:
        if similar(line['URL'].split('.')[1].lower(), company.lower()) > 0.5:
            URL_TEXT_data = line['TEXT']
            token_URL = word_tokenize(URL_TEXT_data)
            long_words = [w for w in token_URL if len(w) > 10]
            if not long_words:
               pass
            else:
                long_words = " ".join(long_words)
                Company_array.append(long_words) # Append to the greater array 
    USA_TOTAL_COMPANY_DATA[company] = Company_array
     
USA_TOTAL_COMPANY_DATA = dict((k, v) for k, v in TOTAL_COMPANY_DATA.iteritems() if v)
USA_test_key_data = []
USA_class_data = []

TC = {}
for key in TOTAL_COMPANY_DATA.keys():
    A = []
    TC = TOTAL_COMPANY_DATA[key][:20]

US_TC = {}
for key in USA_TOTAL_COMPANY_DATA.keys():
    A = []
    US_TC = USA_TOTAL_COMPANY_DATA[key][:20]

TC_class = {}
for key in TC.keys():
    class_TC = []
    for line in TC[key]:
        class_TC.append(text_clf_SVM_GER.predict([line])[0])
    TC_class[key] = class_TC

USA_TC_class = {}
for key in tqdm(US_TC.keys()):
    class_TC = []
    for line in tqdm(US_TC[key]):
        class_TC.append(text_clf_SVM_ENG.predict([line])[0])
    USA_TC_class[key] = class_TC

for key in TOTAL_COMPANY_DATA.keys():
    trans = []
    for line in TOTAL_COMPANY_DATA[key]:
        try:
            trans.append(detect(line))
        except:
            language = "error"
            pass
    translation[key] = trans




