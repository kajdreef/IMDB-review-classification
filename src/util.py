import csv
import numpy as np
from numpy import savetxt, loadtxt, save, load
from scipy.sparse import save_npz, load_npz
import os.path


def curry(func):
    def f(Xtr, Xte):
        return [[func(review) for review in Xtr], [func(review) for review in Xte]]
    return f

def output_to_csv(file_name, head, data):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(head)
        writer.writerows(data)


def output_features_to_file(Xtr, Ytr, Xte, Yte, output_path='./data/'):
    if type(Xtr) is np.ndarray:
        savetxt('{}/training_features_array.feat'.format(output_path), Xtr, fmt='%f')
        savetxt('{}/test_features_array.feat'.format(output_path), Xte, fmt='%f')
    else: 
        save_npz('{}/training_features_sparse'.format(output_path), Xtr)
        save_npz('{}/test_features_sparse'.format(output_path), Xte)
    
    savetxt('{}/training.target'.format(output_path), Ytr.astype(int), fmt='%i')
    savetxt('{}/test.target'.format(output_path), Yte.astype(int), fmt='%i')


def load_features_from_file(input_path='./data/'):

    if os.path.isfile('{}/test_features_array.feat'.format(input_path)):
        print("Load as np array...")
        Xtr = loadtxt('{}/training_features_array.feat'.format(input_path), dtype=float)
        Xte = loadtxt('{}/test_features_array.feat'.format(input_path), dtype=float)
    else: 
        print("Load as Sparse matrix...")
        Xtr = load_npz('{}/training_features_sparse.npz'.format(input_path))
        Xte = load_npz('{}/test_features_sparse.npz'.format(input_path))

    Ytr = loadtxt('{}/training.target'.format(input_path), dtype=int)
    Yte = loadtxt('{}/test.target'.format(input_path), dtype=int)
    
    print("Xtr.shape: {}", Xtr.shape)
    print("Xte.shape: {}", Xte.shape)
    return Xtr, Ytr, Xte, Yte
