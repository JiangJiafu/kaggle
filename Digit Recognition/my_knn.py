# encoding=utf8
import cProfile
import numpy as np
import csv
import operator


def toInt(array):
    """将字符串转成int"""
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def nomalizing(array):
    """归一化"""
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


def __cmd(a, b):
    if a['value'] < b['value']:
        return -1
    elif a['value'] > b['value']:
        return 1
    else:
        return 0


def calculate_similarity(test_record, train_record):
    """calculate the similarity of test record and the train record.
    
    Returns:
        the similarity value.
    """

    value = 0
    for i in xrange(784):
        if test_record[i] > 0:
            if train_record[i] > 0:
                value += 1
            else:
                value -= 1
        else:
            if train_record[i] > 0:
                value -= 1
    return value


def old_classify(k, test_record, train_data, labels):
    """
    :param k: 
    :param test_record: 1 * n 
    :param train_data: m * n
    :param labels: m * 1
    :return: classifiy result
    """
    value_list = []
    j = 0
    record_num = len(train_data)
    for i in range(record_num):
        train_record = train_data[i]
        # print for test
        j += 1
        # if j % 1000 == 0:
        #     print 'j:%d' % j
        # print 'len value_list:%d' % len(value_list)
        value = calculate_similarity(test_record, train_record)
        if value <= 0:
            continue
        value_list.append({'value': value, 'label': labels[0][i]})
    # print value_list
    value_list.sort(cmp=__cmd)
    value_list = value_list[0:k]
    # print 'value list2:', value_list
    label0_num = len([x for x in value_list if x['label'] == 0])
    label1_num = len([x for x in value_list if x['label'] == 1])
    label2_num = len([x for x in value_list if x['label'] == 2])
    label3_num = len([x for x in value_list if x['label'] == 3])
    label4_num = len([x for x in value_list if x['label'] == 4])
    label5_num = len([x for x in value_list if x['label'] == 5])
    label6_num = len([x for x in value_list if x['label'] == 6])
    label7_num = len([x for x in value_list if x['label'] == 7])
    label8_num = len([x for x in value_list if x['label'] == 8])
    label9_num = len([x for x in value_list if x['label'] == 9])
    label_num = [label0_num, label1_num, label2_num, label3_num, label4_num, label5_num, label6_num, label7_num,
                 label8_num, label9_num]
    max_label_num = label_num[0]
    ret = 0
    for i in range(1, 10):
        if label_num[i] > max_label_num:
            ret = i
            max_label_num = label_num[i]

    return ret


def saveResult(result):
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
        # if i == 2:
        #     break
    saveResult(result_list)


if __name__ == '__main__':
    # read_train_data()
    cProfile.run("my_knn()")
