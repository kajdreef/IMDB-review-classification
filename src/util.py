import csv

def curry(func):
    def f(Xtr, Xte):
        return [[func(review) for review in Xtr], [func(review) for review in Xte]]
    return f

def output_to_csv(file_name, head, data):
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(head)
        writer.writerows(data)