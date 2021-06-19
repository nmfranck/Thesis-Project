# Thesis-Project: Using machine learning techniques to analyze and predict open-ended survey data
## Overview
This thesis has as main goal to investigate if predictive machine learning techniques can be used to solve a multi-label classification problem based on online hotel reviews. 
The labels that are attempted to be predicted are features present in the response of a hotel to a review. These labels are modelled using both pre-coded features extracted by humand hand as well as using topics extracted by automated topic modelling.

## Code and resources used
Python Version: 3.8  
Packages: pandas, sklearn, xgboost, seaborn, hyperopt, gensim, spacy, nltk  
Nested cross validation: https://github.com/rasbt/stat451-machine-learningfs20/blob/master/L11/code/11-eval4-algo__nested-cv_verbose1.ipynb

## EDA
Investigated the distribution of pre-coded features and labels. The distribution of the labels revealed that some labels had imbalance.  
- Distribution of pre-coded features  
![alt text](https://github.com/nmfranck/Thesis-Project/blob/main/distribution%20of%20pre-coded%20features.png "Distribution pre-coded features")
- Distribution of labels 

## Text cleaning and preprocessing
1. Removal of non-alphanumeric characters  
2. Removal of stop words  
3. Applying lemmatizer on the words

Transform cleaned text to document-term matrix using:  
   - Bag of words   
   - TF-IDF

## Modelling 
In order to evaluate the models a nested cross validation approach is taken.   
The main performance used is the F1-score, together with the recall and precision since the data showed imbalance in the labels and the F1-score is more robust to this kind of data. Next to these metrics also the hamming loss and accuracy is reported.   

Three different predictive algorithms are tested:  
-  Naïve Bayes  
-  (weighted) Random Forest  (Optimized using GridsearchCV)
-  XGBoost                   (Optimized using Bayesian optimization with TPE and expected improvement)

Two different topic models are used:  
-  Latent Semantic Analysis
-  Latent Dirichlet Allocation

## Model Performance 
1. Using pre-coded features 
-  Naïve Bayes: F1-score= 13,09%
-  (weighted) Random Forest: F1-score= 48,28%
-  XGBoost: F1-score= 48,05%
2. Using LSA
-  Naïve Bayes: F1-score= 36,12% (BOW) / 42,89% (TFIDF)    
-  (weighted) Random Forest: F1-score= 61,99% (BOW) / 64,70% (TFIDF)   
-  XGBoost: F1-score= 62,12% (BOW) / 62,86% (TFIDF)  
3. Using LDA
-  Naïve Bayes: F1-score= 48,12% (BOW) / 46,46% (TFIDF)   
-  (weighted) Random Forest: F1-score= 60,03% (BOW) / 60,10% (TFIDF)  
-  XGBoost: F1-score= 59,00% (BOW) / 52,85% (TFIDF)  

## Identified topics
I used the random forest model with LSA and TFIDF (best performance + tuned number of topics close to optimal number of topics according to coherence score) to find the most important topics using the 'feature importances'. This allowed to interpret these most important topics and create insight in what motivates the owner of a hotel to respond positively to an online review. 

## Conclusion
The modelling results showed that these methods obtain good performance on common labels. Performance on the labels with an extreme imbalance (<1% of the observations had a label) was considerably lower. This performance could be increased by using resampling techniques. The results showed that the investigated methods can be used for a multi-label classification task based on open-ended survey data and can also be used to identify important topics that allow to generate insights in the online reviews.  





