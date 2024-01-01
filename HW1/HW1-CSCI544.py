#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
import contractions 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


warnings.simplefilter(action='ignore', category=Warning)


#Read Data
data = pd.read_csv('data.tsv',on_bad_lines='skip', sep="\t",usecols=['star_rating','review_body'])


#Create 2 classes
data['class'] = data['star_rating'].apply(lambda x: 1 if x in [1,2,3] else 2)


#We form two classes and select 50000 reviews randomly from each class.
balanced_data = pd.DataFrame(columns=data.columns)
for rating in data['class'].unique():
    class_set = data[data['class'] == rating]
    random_sample = class_set.sample(n=50000)
    balanced_data = pd.concat([balanced_data,random_sample]) 


# Average before data cleaning
review_col = balanced_data['review_body']
average_before = sum(len(str(review)) for review in review_col)/len(review_col)

#Data Cleaning


#Pre-processing

# Converting into lower case
balanced_data['review_body'] = balanced_data['review_body'].str.lower()
# Remove HTML tags
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: re.sub(r'<.*?>','',str(d)))
# Remove URLs
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: re.sub(r'https?://\S+www.\S+', '', d))
# Remove non alphabetical character
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: re.sub(r'[^a-zA-Z\s]','',d))
# Remove extra spaces
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: ' '.join(d.split()))
# Perform contractions
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: contractions.fix(d))

# Average after data cleaning
review_col = balanced_data['review_body']
average_after = sum(len(str(review)) for review in review_col)/len(review_col)
print(f"{average_before}, {average_after}")


# remove the stop words 
stopwords = set(stopwords.words('english'))
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: ' '.join([word for word in d.split() if word not in stopwords])) 



# perform lemmatization  
lemmatizer = WordNetLemmatizer()
balanced_data['review_body'] = balanced_data['review_body'].apply(lambda d: ' '.join([lemmatizer.lemmatize(word) for word in d.split()]))


# 
review_col = balanced_data['review_body']
average_after_preprocessing = sum(len(str(review)) for review in review_col)/len(review_col)
print(f"{average_after}, {average_after_preprocessing}")


# TF-IDF and BoW Feature Extraction
X = balanced_data['review_body']
Y = balanced_data['class']

tfidf_vectorizer = TfidfVectorizer()
tfidf_X = tfidf_vectorizer.fit_transform(X)
X_traintf, X_testtf, Y_traintf, Y_testtf = train_test_split(tfidf_X, Y, test_size=0.2)

count_vectorizer = CountVectorizer()
bow_X = count_vectorizer.fit_transform(X)
X_trainbow, X_testbow, Y_trainbow, Y_testbow = train_test_split(bow_X, Y, test_size=0.2)


#Perceptron Using Both Features

#Perceptron using TFIDF
perceptron_tfidf = Perceptron()
perceptron_tfidf.fit(X_traintf, Y_traintf.astype('int'))

y_tf = perceptron_tfidf.predict(X_testtf)

precision_tf = precision_score(Y_testtf.astype('int'), y_tf.astype('int'), average='binary')
recall_tf = recall_score(Y_testtf.astype('int'), y_tf.astype('int'), average='binary')
f1_tf = f1_score(Y_testtf.astype('int'), y_tf.astype('int'), average='binary')

print(f"{precision_tf :.4f} {recall_tf:.4f} {f1_tf:.4f}")

#Perceptron using BOW
perceptron_bow = Perceptron()
perceptron_bow.fit(X_trainbow, Y_trainbow.astype('int'))

y_bow = perceptron_bow.predict(X_testbow)

precision_bow = precision_score(Y_testbow.astype('int'),y_bow.astype('int'), average='binary')
recall_bow = recall_score(Y_testbow.astype('int'),y_bow.astype('int'), average='binary')
f1_bow = f1_score(Y_testbow.astype('int'),y_bow.astype('int'), average='binary')

print(f"{precision_bow :.4f} {recall_bow:.4f} {f1_bow:.4f}")


#SVM Using Both Features

#Linear SVM using TFIDF
svm_model_tf = LinearSVC()
svm_model_tf.fit(X_traintf, Y_traintf.astype('int'))


y_tf_svm = svm_model_tf.predict(X_testtf)

# Precision
precision_tf_svm = precision_score(Y_testtf.astype('int'), y_tf_svm.astype('int'), average='binary')

# Recall
recall_tf_svm = recall_score(Y_testtf.astype('int'), y_tf_svm.astype('int'), average='binary')

# F1 score
f1_tf_svm = f1_score(Y_testtf.astype('int'), y_tf_svm.astype('int'), average='binary')

print(f"{precision_tf_svm :.4f} {recall_tf_svm:.4f} {f1_tf_svm:.4f}")


# Linear SVM using BOW
svm_model_bow = LinearSVC()
svm_model_bow.fit(X_trainbow, Y_trainbow.astype('int'))

y_bow_svm = svm_model_bow.predict(X_testbow)

# Precision
precision_bow_svm = precision_score(Y_testbow.astype('int'),y_bow_svm.astype('int'), average='binary')

# Recall
recall_bow_svm = recall_score(Y_testbow.astype('int'),y_bow_svm.astype('int'), average='binary')

# F1 score
f1_bow_svm = f1_score(Y_testbow.astype('int'),y_bow_svm.astype('int'), average='binary')

print(f"{precision_bow_svm :.4f} {recall_bow_svm:.4f} {f1_bow_svm:.4f}")


#Logistic Regression Using Both Features

#Logistic Regression using TFIDF
logisctic_regression_tfidf = LogisticRegression()
logisctic_regression_tfidf.fit(X_traintf, Y_traintf.astype('int'))

y_tf_reg = logisctic_regression_tfidf.predict(X_testtf)

# Precision
precision_tf_reg = precision_score(Y_testtf.astype('int'), y_tf_reg.astype('int'), average='binary')

# Recall
recall_tf_reg = recall_score(Y_testtf.astype('int'), y_tf_reg.astype('int'), average='binary')

# F1 score
f1_tf_reg = f1_score(Y_testtf.astype('int'), y_tf_reg.astype('int'), average='binary')

print(f"{precision_tf_reg :.4f} {recall_tf_reg:.4f} {f1_tf_reg:.4f}")


#Logistic Regression using BOW
logisctic_regression_bow = LogisticRegression(max_iter = 400)
logisctic_regression_bow.fit(X_trainbow, Y_trainbow.astype('int'))

y_bow_reg = logisctic_regression_bow.predict(X_testbow)

# Precision
precision_bow_reg = precision_score(Y_testbow.astype('int'), y_bow_reg.astype('int'), average='binary')

# Recall
recall_bow_reg = recall_score(Y_testbow.astype('int'), y_bow_reg.astype('int'), average='binary')

# F1 score
f1_bow_reg = f1_score(Y_testbow.astype('int'), y_bow_reg.astype('int'), average='binary')

print(f"{precision_bow_reg :.4f} {recall_bow_reg:.4f} {f1_bow_reg:.4f}")


#Naive Bayes Using Both Features

#Naive Bayes on TF-IDF
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_traintf, Y_traintf.astype('int'))

y_tfidf_nb = nb_tfidf.predict(X_testtf)

# Precision
precision_tfidf_nb = precision_score(Y_testtf.astype('int'), y_tfidf_nb.astype('int'), average='binary')

# Recall
recall_tfidf_nb = recall_score(Y_testtf.astype('int'), y_tfidf_nb.astype('int'), average='binary')

# F!1 score
f1_tfidf_nb = f1_score(Y_testtf.astype('int'), y_tfidf_nb.astype('int'), average='binary')

print(f"{precision_tfidf_nb :.4f} {recall_tfidf_nb:.4f} {f1_tfidf_nb:.4f}")


# Naive Bayes on BOW
nb_bow = MultinomialNB()
nb_bow.fit(X_trainbow, Y_trainbow.astype('int'))

y_bow_nb = nb_tfidf.predict(X_testbow)

# Precision
precision_bow_nb = precision_score(Y_testbow.astype('int'), y_bow_nb.astype('int'), average='binary')

# Recall
recall_bow_nb = recall_score(Y_testbow.astype('int'), y_bow_nb.astype('int'), average='binary')

# F1 score
f1_bow_nb = f1_score(Y_testbow.astype('int'), y_bow_nb.astype('int'), average='binary')

print(f"{precision_bow_nb :.4f} {recall_bow_nb:.4f} {f1_bow_nb:.4f}")

