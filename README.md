# Spam-Detector
Classification of Ham and Spam using Naive Bayes, Multinomial Naive Bayes and Logistic Regression


Language:
Python 2.7

Required Libraries:
io
sys
re
pandas
time
numpy
sklearn
argparse

Command Synatx:
python classify.py -o option_no -train_dir ‘train_set_directory_name’ -test_dir ‘test_ set_directory_name’

Command example:
python classify.py -o 3 -train_dir ‘hw2_train’ -test_dir ‘hw2_test’ 

Please select options according to following list:

1. Multinomial Naive Bayes on the Bag of words model
2. Discrete Naive Bayes on the Bernoulli model
3. Logistic Regression on Bag of words
4.  Logistic Regression on Bernoulli models
5. SGDClassifier on Bag of words
6. SGDClassifier on Bernoulli models
