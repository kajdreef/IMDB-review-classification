import csv
from numpy import savetxt, loadtxt
from scipy.sparse import save_npz, load_npz

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
    save_npz('{}/training_features'.format(output_path), Xtr)
    save_npz('{}/test_features'.format(output_path), Xte)
    savetxt('{}/training.target'.format(output_path), Ytr.astype(int), fmt='%i')
    savetxt('{}/test.target'.format(output_path), Yte.astype(int), fmt='%i')


def load_features_from_file(input_path='./data/'):
    Xtr = load_npz('{}/training_features.npz'.format(input_path))
    Xte = load_npz('{}/test_features.npz'.format(input_path))
    Ytr = loadtxt('{}/training.target'.format(input_path), dtype=int)
    Yte = loadtxt('{}/test.target'.format(input_path), dtype=int)

    return Xtr, Ytr, Xte, Yte
