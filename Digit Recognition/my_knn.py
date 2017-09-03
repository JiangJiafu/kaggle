# encoding=utf8
import numpy as np
import csv
import operator


def toInt(array):
    """Change the string to int"""
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def nomalizing(array):
    m, n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


def load_train_data():
    lst = []
    with open('train.csv', 'r') as csv_file:
        lines = csv.reader(csv_file)
        num = 0
        for line in lines:  # 先只用前10000条训练集
            lst.append(line)  # 42001*785
            # num += 1
            # if num > 4000:
            #     break
        del (lst[0])
        lst = np.array(lst)
        label = lst[:, 0]
        data = lst[:, 1:]
        return nomalizing(toInt(data)), toInt(label)  # label 42000*1  data 42000*784


def load_test_data():
    lst = []
    with open('test.csv', 'r') as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            lst.append(line)
    del (lst[0])
    data = np.array(lst)
    return nomalizing(toInt(data))


def save_result(result):
    with open('result.csv', 'wb') as my_file:
        my_writer = csv.writer(my_file)
        my_writer.writerow(['ImageId', 'Label'])
        for i in range(len(result)):
            my_writer.writerow([i + 1, result[i]])


def classify(k, test_record, train_data, labels):
    """
    :param k: 
    :param test_record: 1 * n 
    :param train_data: m * n
    :param labels: m * 1
    :return: classifiy result
    """

    test_record = np.mat(test_record)
    data_set = np.mat(train_data)
    labels = np.mat(labels)
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(test_record, (data_set_size, 1)) - data_set
    sq_diff_mat = np.array(diff_mat) ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        voteIlabel = labels[0, sorted_dist_indicies[i]]
        class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def my_knn(k=10):
    # read the training data and label
    train_data, labels = load_train_data()
    test_data = load_test_data()
    m, n = np.shape(test_data)
    result_list = []
    for i in range(m):
        classifier_result = classify(k, test_data[i], train_data, labels)
        result_list.append(classifier_result)
        print 'test record %d, result:%d ' % (i, classifier_result)
        i += 1
        if i % 1000 == 0:
            print '%d %% finished' % (i * 100 / m)
    save_result(result_list)


if __name__ == '__main__':
    """This script uses KNN algorithm to solve the problem on kaggle 
    https://www.kaggle.com/c/digit-recognizer.
    
    Before running this script, you should download the "test.csv" and
    "train.csv" from https://www.kaggle.com/c/digit-recognizer, and 
    put them in the same directory of this script. Then run
    python my_knn.py
     
    Note: I run this script on my own PC(Intel(R) Core(TM) i5-2400 3.10GHz, 4GB memory), 
    it takes about 7 hours to finish.
    
    """
    my_knn()