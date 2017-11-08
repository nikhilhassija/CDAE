import numpy
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile

def load_data():
    '''
    load data from MovieLens 100K Dataset
    http://grouplens.org/datasets/movielens/

    Note that this method uses ua.base and ua.test in the dataset.

    :return: train_users, train_x, test_users, test_x
    :rtype: list of int, numpy.array, list of int, numpy.array
    '''
    path = get_file('ml-1m.zip', origin='http://files.grouplens.org/datasets/movielens/ml-1m.zip')
    with ZipFile(path, 'r') as ml_zip:
        max_item_id  = -1
        train_history = {}
        count = 0
        with ml_zip.open('ml-1m/ratings.dat', 'r') as file:
            for line in file:
                if(count<0.8*6040):
                    user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('::')
                    if int(user_id) not in train_history:
                        train_history[int(user_id)] = [int(item_id)]
                    else:
                        train_history[int(user_id)].append(int(item_id))

                    if max_item_id < int(item_id):
                        max_item_id = int(item_id)

        count =0
        test_history = {}
        with ml_zip.open('ml-1m/ratings.dat', 'r') as file:
            for line in file:
                if(count>=0.8*6040):
                    user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('::')
                    if int(user_id) not in test_history:
                        test_history[int(user_id)] = [int(item_id)]
                    else:
                        test_history[int(user_id)].append(int(item_id))

    max_item_id += 1 # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)
        train_x[i] = numpy.sum(mat, axis=0)

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
