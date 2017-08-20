# encoding=utf8
import heapq
import cProfile
import numpy as np
import csv
import matplotlib.pyplot as plt


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


def read_train_data():
    lst = []
    with open('train.csv', 'r') as csv_file:
        lines = csv.reader(csv_file)
        num = 0
        for line in lines:  # 先只用前10000条训练集
            lst.append(line)  # 42001*785
            num += 1
            if num >= 10001:
                break
        lst.remove(lst[0])  # remove the head line
        lst = np.array(lst)
        label = lst[:, 0]
        data = lst[:, 1:]
        return nomalizing(toInt(data)), toInt(label)  # label 1*42000  data 42000*784


def read_test_data():
    with open('test.csv', 'r') as csv_file:
        return csv.reader(csv_file)


def my_knn(k=10):
    # read the training data and label
    train_data, labels = read_train_data()
    csv_file2 = open('test.csv', 'r')
    test_data_list = csv.reader(csv_file2)
    is_head_line = True
    i = 0
    for test_record in test_data_list:
        if is_head_line:
            is_head_line = False
            continue
        if i in [9, 10, 11, 12, 13]:
            print 'test record %d, result:%d ' % (i, calculate(k, test_record, train_data, labels))
        i += 1


def __cmd(a, b):
    if a['value'] < b['value']:
        return -1
    elif a['value'] > b['value']:
        return 1
    else:
        return 0


def calculate_similarity(test_record, train_record):
    """calculate the test record and the train record.
    
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


def calculate(k, test_record, train_data, labels):
    value_list = []
    test_record = np.array(test_record, dtype='uint8')
    j = 0
    record_num = len(train_data)
    for i in range(record_num):
        train_record = train_data[i]
        # print for test
        j += 1
        if j % 1000 == 0:
            print 'j:%d' % j
            print 'len value_list:%d' % len(value_list)
        value = calculate_similarity(test_record, train_record)
        if value <= 0:
            continue
        value_list.append({'value': value, 'label': labels[0][i]})
    print value_list
    value_list.sort(cmp=__cmd)
    value_list = value_list[0:k]
    print 'value list2:', value_list
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


if __name__ == '__main__':
    # read_train_data()
    cProfile.run("my_knn()")
