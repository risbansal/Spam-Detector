import os
import io
import sys
import re
import pandas as pd
import time
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import argparse


start_time = time.time()


def open_files(path):
    data_list = []
    if not os.path.isdir(path):
        sys.exit("Input path is not a directory")
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        try:
            reader = io.open(filename,'rb')
            file_data = reader.read()
            file_data = str(file_data)
            file_data = re.findall(r'[A-Za-z]+', file_data)
            #file_data = re.sub('\s+', ' ', file_data)
            #file_data = file_data.split()
            data_list.append(file_data)

        except IOError:
            sys.exit("Cannot read file")


    return data_list


def get_vocab(ham_list, spam_list):

    vocab_list = []

    for doc in ham_list:
        for word in doc:
            if word not in vocab_list:
                vocab_list.append(word)

    for doc in spam_list:
        for word in doc:
            if word not in vocab_list:
                vocab_list.append(word)

    vocab_list = vocab_list[3:]


    return vocab_list


def bag_matrix(vocab_list, ham_list, spam_list):
    h_count_mat = []
    s_count_mat = []

    for doc in ham_list:
        t = dict(zip(list(vocab_list), [0] * len(vocab_list)))
        for word in doc:
            if word in t.keys():
                t[word] = 1
        h_count_mat.append(t)

    for doc in spam_list:
        t = dict(zip(list(vocab_list), [0] * len(vocab_list)))
        for word in doc:
            if word in t.keys():
                t[word] = 1
        s_count_mat.append(t)

    h_df = pd.DataFrame(h_count_mat)
    s_df = pd.DataFrame(s_count_mat)

    h_df["Y"] = [1 for i in range(len(h_df.index))]
    s_df["Y"] = [0 for i in range(len(s_df.index))]

    hs_df = pd.concat([h_df, s_df], ignore_index=True)

    return hs_df

def ber_matrix(vocab_list, ham_list, spam_list):
    h_count_mat = []
    s_count_mat = []


    for doc in spam_list:
        t = dict(zip(list(vocab_list), [0] * len(vocab_list)))
        for word in doc:
            if word in t.keys() and t[word] != 1:
                t[word] += 1
        s_count_mat.append(t)

    for doc in ham_list:
        t = dict(zip(list(vocab_list), [0] * len(vocab_list)))
        for word in doc:
            if word in t.keys() and t[word] != 1:
                t[word] += 1
        h_count_mat.append(t)

    s_df = pd.DataFrame(s_count_mat)
    h_df = pd.DataFrame(h_count_mat)

    s_df["Y"] = [0 for i in range(len(s_df.index))]
    h_df["Y"] = [1 for i in range(len(h_df.index))]

    hs_df = pd.concat([h_df, s_df], ignore_index=True)

    return hs_df

def multi_nb(bag_count):
    prob_class = [0,0]
    ham_c = bag_count["Y"].sum()
    total = len(bag_count.index)
    spam_c = total - ham_c
    prob_class[1] = np.log2(np.true_divide(spam_c, total))
    prob_class[0] = np.log2(np.true_divide(ham_c, total))
    ham_prob_dict = dict(zip(list(bag_count.columns[:-1]), [0]*len(bag_count.columns[:-1])))
    spam_prob_dict = dict(zip(list(bag_count.columns[:-1]), [0]*len(bag_count.columns[:-1])))
    ham_prob = np.ones(len(bag_count.columns[:-1]))
    spam_prob = np.ones(len(bag_count.columns[:-1]))
    tot_h_sum = total
    tot_s_sum = total

    for i, col in enumerate(bag_count.columns[:-1]):

        tot_s_sum = tot_h_sum + (bag_count[col][bag_count["Y"] == 0].sum() + 1)
        tot_h_sum = tot_h_sum + (bag_count[col][bag_count["Y"] == 1].sum() + 1)
        spam_prob[i] = spam_prob[i] + bag_count[col][bag_count["Y"] == 0].sum()
        ham_prob[i] = ham_prob[i] + bag_count[col][bag_count["Y"] == 1].sum()


    ham_prob = np.log2(ham_prob) - np.log2(tot_h_sum)
    spam_prob = np.log2(spam_prob) - np.log2(tot_s_sum)

    for i, word in enumerate(list(bag_count.columns[:-1])):
        ham_prob_dict[word] = ham_prob[i]
        spam_prob_dict[word] = spam_prob[i]


    return ham_prob_dict, spam_prob_dict, prob_class

def ber_train(ber_count):
    prob_class = [0,0]
    ham_c = ber_count["Y"].sum()
    total = len(ber_count.index)
    spam_c = total - ham_c
    prob_class[1] = np.log2(np.true_divide(spam_c, total))
    prob_class[0] = np.log2(np.true_divide(ham_c, total))
    ham_prob_dict = dict(zip(list(ber_count.columns[:-1]), [0]*len(ber_count.columns[:-1])))
    spam_prob_dict = dict(zip(list(ber_count.columns[:-1]), [0]*len(ber_count.columns[:-1])))
    ham_prob = np.ones(len(ber_count.columns[:-1]))
    spam_prob = np.ones(len(ber_count.columns[:-1]))
    tot_h_sum = 0
    tot_s_sum = 0

    for i, col in enumerate(ber_count.columns[:-1]):


        spam_prob[i] = spam_prob[i] + ber_count[col][ber_count["Y"] == 0].sum()
        ham_prob[i] = ham_prob[i]  + ber_count[col][ber_count["Y"] == 1].sum()

    tot_s_sum = np.log2(len(ber_count[ber_count["Y"] == 1].index) + 2)
    tot_h_sum = np.log2(len(ber_count[ber_count["Y"] == 0].index) + 2)


    ham_prob = np.log2(ham_prob) - tot_h_sum
    spam_prob = np.log2(spam_prob) - tot_s_sum

    for i, word in enumerate(list(ber_count.columns[:-1])):
        ham_prob_dict[word] = ham_prob[i]
        spam_prob_dict[word] = spam_prob[i]


    return ham_prob_dict, spam_prob_dict, prob_class

def mnb_tester(ham_test_list, spam_test_list, ham_prob_dict, spam_prob_dict, prob_class):


    classify_ham = [0] * len(ham_test_list)
    classify_spam = [0] * len(spam_test_list)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for doc_ham, doc_spam in zip(ham_test_list, spam_test_list):
        total_prob_ham_h = prob_class[0]
        total_prob_spam_h = prob_class[1]
        total_prob_ham_s = prob_class[0]
        total_prob_spam_s = prob_class[1]
        doc_index_h = ham_test_list.index(doc_ham)
        doc_index_s = spam_test_list.index(doc_spam)

        for word_h, word_s in zip(doc_ham, doc_spam):
            if word_h in ham_prob_dict.keys():
                total_prob_ham_h = total_prob_ham_h + ham_prob_dict[word_h]
            if word_s in ham_prob_dict.keys():
                total_prob_ham_s = total_prob_ham_s + ham_prob_dict[word_s]

            if word_h in spam_prob_dict.keys():
                total_prob_spam_h = total_prob_spam_h + spam_prob_dict[word_h]
            if word_s in spam_prob_dict.keys():
                total_prob_spam_s = total_prob_spam_s + spam_prob_dict[word_s]

        if total_prob_ham_h > total_prob_spam_h:
            classify_ham[doc_index_h] = 1
        if total_prob_ham_s > total_prob_spam_s:
            classify_spam[doc_index_s] = 1

    tp = np.sum(classify_ham)
    fn = len(classify_ham) - tp

    fp = np.sum(classify_spam)
    tn = len(classify_spam) - fp



    accuracy = np.true_divide((tp + tn), (tp + tn + fp + fn))
    precision = np.true_divide(tp, (tp + fp))
    recall = np.true_divide(tp, (tp + fn))
    fscore = np.true_divide((2 * recall * precision), (precision + recall))
    return accuracy, precision, recall, fscore


def ber_tester(ber_count, ham_test_list, spam_test_list, ham_prob_dict, spam_prob_dict, prob_class):


    classify_ham = [0] * len(ham_test_list)
    classify_spam = [0] * len(spam_test_list)

    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for doc_ham, doc_spam in zip(ham_test_list, spam_test_list):
        total_prob_ham_h = prob_class[0]
        total_prob_spam_h = prob_class[1]
        total_prob_ham_s = prob_class[0]
        total_prob_spam_s = prob_class[1]
        doc_index_h = ham_test_list.index(doc_ham)
        doc_index_s = spam_test_list.index(doc_spam)

        for word in ber_count.columns[:-1]:
            if word in doc_ham:
                total_prob_ham_h = total_prob_ham_h + ham_prob_dict[word]
            else:
                total_prob_ham_h = total_prob_ham_h + (1 - ham_prob_dict[word])

            if word in doc_spam:
                total_prob_ham_s = total_prob_ham_s + ham_prob_dict[word]
            else:
                total_prob_ham_s = total_prob_ham_s + (1 - ham_prob_dict[word])

            if word in doc_ham:
                total_prob_spam_h = total_prob_spam_h + spam_prob_dict[word]
            else:
                total_prob_ham_h = total_prob_spam_h + (1 - spam_prob_dict[word])

            if word in doc_spam:
                total_prob_spam_s = total_prob_spam_s + spam_prob_dict[word]
            else:
                total_prob_spam_s = total_prob_spam_s + (1 - spam_prob_dict[word])

        if total_prob_ham_h > total_prob_spam_h:
            classify_ham[doc_index_h] = 1
        if total_prob_ham_s > total_prob_spam_s:
            classify_spam[doc_index_s] = 1


    tp = np.sum(classify_ham)
    fn = len(classify_ham) - tp

    fp = np.sum(classify_spam)
    tn = len(classify_spam) - fp

    accuracy = np.true_divide((tp + tn), (tp + tn + fp + fn))
    precision = np.true_divide(tp, (tp + fp))
    recall = np.true_divide(tp, (tp + fn))
    fscore = np.true_divide((2 * recall * precision), (precision + recall))
    return accuracy, precision, recall, fscore


def log_reg(count_matrix):

    weights = np.zeros(len(count_matrix.columns[:-1]))
    bias = 0
    lr = 0.1
    l1 = 0.06

    for e in range(1000):
        grad_ascent = np.zeros(len(count_matrix.columns[:-1]))
        grad_bias = 0
        for i, doc in count_matrix.iloc[:, :-1].iterrows():

            z = bias + (weights * np.array(doc)).sum()
            if z < 37:
                prob = np.true_divide(1, 1 + np.exp(z))
            else:
                prob = 0.5
            grad_ascent = grad_ascent + np.array(doc) * (count_matrix["Y"][i] -prob)  - (l1 * np.array(doc)).sum()
            grad_bias = grad_bias + (count_matrix["Y"][i] - prob)

        weights = weights + (lr * grad_ascent)
        bias = bias + (lr * grad_bias)


    return weights, bias

def log_reg_test(weights, bias, count_matrix_test):

    pred = np.zeros(len(count_matrix_test.iloc[:,:-1].index))
    for i, doc in count_matrix_test.iloc[:, :-1].iterrows():

        z = bias + (weights * np.array(doc)).sum()
        pred[i] = np.true_divide(1, 1 + np.exp(z))

    check = np.array(count_matrix_test["Y"]) == pred
    accuracy = np.true_divide(check.sum(), len(check))


    return accuracy

def sgd(count_matrix, bag_count_test):

    X_train = np.array(count_matrix.iloc[:, :-1])
    Y_train = np.array(count_matrix.iloc[:, -1])
    X_test = np.array(bag_count_test.iloc[:, :-1])
    Y_test = np.array(bag_count_test["Y"])

    lr = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    nj =np.array([-1,-2, -3])
    epoch = np.array([500])


    mod = linear_model.SGDClassifier()
    grid = GridSearchCV(estimator=mod, param_grid=dict(alpha=lr, n_jobs = nj,max_iter = epoch))
    grid.fit(X_train, Y_train)
    #print(grid.best_score_)


    # clf.fit(X_train, Y_train)
    pred = grid.predict(X_test)
    check = Y_test == pred
    a = np.true_divide(check.sum(), len(check))


    return a



def  main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', type=int, default = 1)
    parser.add_argument('-train_dir', '--trd', type=str, default = 1)
    parser.add_argument('-test_dir', '--td', type=str, default = 1)


    arg = parser.parse_args()

    option = arg.option
    train_dir = arg.trd
    test_dir = arg.td

    #Data

    ham_train = train_dir + "/ham"
    spam_train = train_dir + "/spam"

    ham_test = test_dir + "/ham"
    spam_test = test_dir + "/spam"

    #Getting word list from data
    ham_list = open_files(ham_train)
    spam_list = open_files(spam_train)

    ham_test_list = open_files(ham_test)
    spam_test_list = open_files(spam_test)

    #Getting Vocabulary
    vocab_list = get_vocab(ham_list, spam_list)

    #Bag of words matrix
    bag_count = bag_matrix(vocab_list, ham_list, spam_list)
    bag_count_test = bag_matrix(vocab_list, ham_test_list, spam_test_list)


    #Bernoulli Matrix
    ber_count = ber_matrix(vocab_list,ham_list, spam_list)
    ber_count_test = ber_matrix(vocab_list, ham_test_list, spam_test_list)

    if option == 1:
        # Multinomial Naive Bayes
        ham_prob_dict, spam_prob_dict, prob_class = multi_nb(bag_count)

        # Bag of Word NB Accuracy
        accuracy_bow, pr_bow, r_bow, fs_bow = mnb_tester(ham_test_list, spam_test_list, ham_prob_dict, spam_prob_dict,
                                                         prob_class)
        print("Accuracy of Bag of Words:", accuracy_bow)
        print ("Precision of Bag of Words:", pr_bow)
        print ("Recall of Bag of Words:", r_bow)
        print ("F1 Score of Bag of Words:", fs_bow)

    if option == 2:

        #Discrete Naive Bayes
        ham_prob_dict2, spam_prob_dict2,prob_class2 = ber_train(ber_count)


        #Bernoulli NB Accuracy
        accuracy_ber, pr_ber, r_ber, fs_ber = ber_tester(ber_count, ham_test_list, spam_test_list, ham_prob_dict2, spam_prob_dict2, prob_class2)
        print("Accuracy of Bernoulli:", accuracy_ber)
        print ("Precision of Bernoulli:", pr_ber)
        print ("Recall of Bernoulli:", r_ber)
        print ("F1 Score of Bernoulli:", fs_ber)

    if option == 3:

        #Logistic Regression using BOW
        weights_bow, bias_bow = log_reg(bag_count)
        accuracy_lr_bow = log_reg_test(weights_bow, bias_bow, bag_count_test)
        print ("Accuracy for Logistic Regression using Bag of Words:", accuracy_lr_bow)


    if option == 4:

        # Logistic Regression using Bernoulli
        weights_ber, bias_ber = log_reg(ber_count)
        accuracy_lr_ber = log_reg_test(weights_ber, bias_ber, ber_count_test)
        print ("Accuracy for Logistic Regression using Bernoulli:", accuracy_lr_ber)

    if option == 5:
        #SGD for BOW
        ac_sgd_bow = sgd(bag_count, bag_count_test)
        print("Accuracy for SGD using Bag of Words:", ac_sgd_bow)

    if option == 6:
        # SGD for Bernoulli
        ac_sgd_ber = sgd(ber_count, ber_count_test)
        print("Accuracy for SGD using Bernoulli", ac_sgd_ber)


    etime = time.time() - start_time
    print("Execution time:", etime)

main()