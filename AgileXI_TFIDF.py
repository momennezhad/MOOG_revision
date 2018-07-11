"""

Implementation of the TF-IDF algorithm to the scraped data from MOOG

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

# Cross validation and learning curves

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# RMSE metric

from sklearn.metrics import mean_squared_error

# Language detection

from langdetect import detect
import string

# Defining the stopwords - accumulation of the english, italin and german stop words, along with the punctuation and numbers

stop_words = stopwords.words('english') + stopwords.words('italian') + stopwords.words('german') + list(punctuation) + list('0123456789')

# Plot learning curves for the model

def plot_learning_curve(pipeline, X, y):
    
    X,train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        y_train_predict = pipeline.predict(X_train[:m])
        y_val_predict = pipeline.predict(X_val[:m])

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

        return train_errors, val_errors


# String one-hot vectorizer

def string_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] 
                  for letter in strng]
    return vector

class Scraping_TFIDF(object):
    """
    """
    def __init__(self):
        """
        """
        self.IT_Company_list = [] 
        self.USA_Company_list = []
        self.GER_Company_list = []
        
        # Classifications loaded 
        self.GER_class = json.load(open('data/class_JSON/Germany_class.json'))
        self.IT_class = json.load(open('data/class_JSON/Italy_class.json'))
        self.USA_class = json.load(open('data/class_JSON/USA_class.json'))

        self.USA_Total_Company_Data = {}
        self.GER_Total_Company_Data = {}
        self.IT_Total_Company_Data = {}

         # Class labels for each URL        

         self.ENG_LAB = []
        self.GER_LAB = []
        self.IT_LAB = []

        # Parsed Text

        self.ENG_TEXT = []
        self.GER_TEXT = []
        self.IT_TEXT = []

    def TFIDF_similarity(self,string1,string2):
        """
        String filter to check whether the company names match the URL pattern
        """
        return SequenceMatcher(None, string1, string2).ratio()

    def TFIDF_load_result(self):
        """
        Load the json files that has the scraped data
        """ 
        self.GER_data = json.load(open('data/result/ger_result.json'))
        self.USA_data = json.load(open('data/result/USA_result.json'))
        self.IT_data = json.load(open('data/result/it_result.json'))

    def TFIDF_load_company_list(self):
        """
        Load the company list data
        """
        self.IT_list = json.load(open('data/Company_list/it.json'))
        self.GER_list = json.load(open('data/Company_list/ger.json'))
        self.USA_list = json.load(open('data/Company_list/USA.json'))


    def TFIDF_hash_table(self):    
        """

        """
        for keys, value in self.IT_list.iteritems():
            self.IT_Company_list.append(keys)
        # Ditto for Germany
        for keys, value in self.GER_list.iteritems():
            self.GER_Company_list.append(keys)

        for keys, value in self.USA_list.iteritems():
            self.US_Company_list.append(keys)

    def TFIDF_reallocate_class(self):
        """
        """
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
        """
        """
        for company_names in self.GER_Company_list:
            Company_array = []
            for dat in self.GER_data:
                URL_TEXT_data = [] 
                if self.TFIDF_similarity(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
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


        for company_names in self.USA_Company_list:
            Company_array = []
            for dat in self.US_data:
                URL_TEXT_data = [] 
                if self.TFIDF_similarity(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
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

        """
        """
        for company_names in self.IT_Company_list:
            Company_array = []
            for dat in self.IT_data:
                URL_TEXT_data = [] 
                if self.TFIDF_similarity(dat['URL'].split('.')[1].lower(), company_names.lower()) > 0.5:
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
        
    def TFIDF_clean(self):
        """
        """
        ENG_collated_data = dict((k, v) for k, v in ENG_collated_data.iteritems() if v)
        GER_collated_data = dict((k, v) for k, v in GER_collated_data.iteritems() if v)
        IT_collated_data = dict((k, v) for k, v in IT_collated_data.iteritems() if v)
        
    def TFIDF_ML_pipelines(self):
        """
        """
        # 3 train-test split of data
        
        X_train_ENG, X_test_ENG, y_train_ENG, y_test_ENG = train_test_split(ENG_TEXT, ENG_LAB, test_size=0.33, random_state=42)
        text_clf_SVM_ENG.fit(X_train_ENG, y_train_ENG)

        X_train_GER, X_test_GER, y_train_GER, y_test_GER = train_test_split(GER_TEXT, GER_LAB, test_size=0.33, random_state=42)
        text_clf_SVM_GER.fit(X_train_GER, y_train_GER)

        X_train_IT, X_test_IT, y_train_IT, y_test_IT = train_test_split(IT_TEXT, IT_LAB, test_size=0.33, random_state=42)
        text_clf_SVM_IT.fit(X_train_IT, y_train_IT)

        # Naive Bayes approach pipeline

        self.text_clf_Bayes = Pipeline([('vect', CountVectorizer(stop_words=stop_words)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),
         ])

        # Adaptive boosting approach pipeline - English - Warning, this will take very long

        self.text_clf_adaboost_ENG = Pipeline([('vect', CountVectorizer(stop_words = stop_words)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=1000, learning_rate=0.5)),
          ])

         # Adaptive boosting approach pipeline - German  - Warning, this will take very long 

        self.text_clf_adaboost_GER = Pipeline([('vect', CountVectorizer(stop_words = stop_words)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=1000, learning_rate=0.5)),
                                              ])

    def TFIDF_text_label_full(self):
        """
        """
        for key in self.USA_Total_Company_Data.keys():
            for line in self.USA_Total_Company_Data[key]:
                self.ENG_LAB.append(self.USA_class_final[key])
                self.ENG_TEXT.append(line)

        for key in self.GER_Total_Company_Data.keys():
            for line in self.GER_Total_Company_Data[key]:
                self.GER_LAB.append(self.GER_class_final[key])
                self.GER_TEXT.append(line)
        
            for key in self.IT_Total_Company_Data.keys():
                for line in self.IT_Total_Company_Data[key]:
                    self.IT_LAB.append(self.IT_class_final[key])
                    self.IT_TEXT.append(line)

    def TFIDF_ML_train(self, X_GER, X_ENG):
        """
        """
        X_train_ENG, X_test_ENG, y_train_ENG, y_test_ENG = train_test_split(self.ENG_TEXT, self.ENG_LAB, test_size=0.33, random_state=42)
        self.text_clf_adaboost_ENG.fit(X_train_ENG, y_train_ENG)

        X_train_GER, X_test_GER, y_train_GER, y_test_GER = train_test_split(self.GER_TEXT, self.GER_LAB, test_size=0.33, random_state=42)
        self.text_clf_adaboost_GER.predict.fit(X_train_GER, y_train_GER)
        
        self.predicter_values_GER = self.text_clf_adaboost_GER.predict(X_train_GER) # Values to do cross-validation check 
        self.predicter_values_ENG = self.text_clf_adaboost_ENG.predict(X_train_ENG) # Values to do cross-validation check 

    def TFIDF_ML_test_data_processing(self):
        """
        """
        # Load German test data 

        GER_test_company_dat = pd.read_excel('Copy of VDMA List for Scoring 200318 final.xlsx') # Initial excel data
        GER_test_dat = json.load(open('test_result.json')) # Scraped data from test excel 
        GER_test_company_list = [] 
        self.GER_test_URL_filter = {}

        # Load USA test data

        USA_test_company_dat = pd.read_excel('US_Target_List 200318.xls')
        USA_test_dat = json.load(open('final_result.json'))
        USA_test_company_list = [] 
        self.USA_test_URL_filter = {}
 
        # Append all the German company names

        for name in GER_test_company_dat['Company Name']:
            GER_test_company_list.append(name)

        for company in self.GER_test_company_list:
            Company_array = []
            for line in test_dat:
                if self.TFIDF_similarity(line['URL'].split('.')[1].lower(), company.lower()) > 0.5:
                    URL_TEXT_data = line['TEXT']
                    token_URL = word_tokenize(URL_TEXT_data)
                    long_words = [w for w in token_URL if len(w) > 10]
                    if not long_words:
                        pass
                    else:
                        long_words = " ".join(long_words)
                    Company_array.append(long_words) # Append to the greater array 
            self.GER_test_URL_filter[company] = Company_array
        self.GER_test_URL_filter = dict((k, v) for k, v in TOTAL_COMPANY_DATA.iteritems() if v) # remove empty entries

        # Append all the USA company names
        
        for name in USA_test_company_dat['Company Name']:
            USA_test_company_list.append(name)
                           
       
        for company in USA_test_company_list:
            Company_array = []
            for line in USA_test_dat:
                if self.TFIDF_similarity(line['URL'].split('.')[1].lower(), company.lower()) > 0.5:
                    URL_TEXT_data = line['TEXT']
                    token_URL = word_tokenize(URL_TEXT_data)
                    long_words = [w for w in token_URL if len(w) > 10]
                    if not long_words:
                        pass
                    else:
                        long_words = " ".join(long_words)
                Company_array.append(long_words) # Append to the greater array 
            self.USA_test_URL_filter[company] = Company_array     
        self.USA_test_URL_filter = dict((k, v) for k, v in TOTAL_COMPANY_DATA.iteritems() if v)

    def TFIDF_FINAL_label(self,truncation_limit):
        """
        """
        GER_TC = {}
        US_TC = {}

        for key in self.GER_test_URL_filter.keys():
            GER_TC = self.GER_test_URL_filter[key][:truncation_limit]
            
        for key in self.USA_test_URL_filter.keys():
            US_TC = USA_TOTAL_COMPANY_DATA[key][:truncation_limit]

        GER_TC_class = {}
        USA_TC_class = {}
         
        for key in tqdm(GER_TC.keys()):
            class_TC = []
            for line in tqdm(TC[key]):
                class_TC.append(text_clf_SVM_GER.predict([line])[0])
                GER_TC_class[key] = class_TC
                
        for key in tqdm(US_TC.keys()):
            class_TC = []
            for line in tqdm(US_TC[key]):
                class_TC.append(text_clf_SVM_ENG.predict([line])[0])
                USA_TC_class[key] = class_TC

        return GER_TC_class, USA_TC_class          




