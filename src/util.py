def curry(func):
    def f(Xtr, Xte):
        return [[func(review) for review in Xtr], [func(review) for review in Xte]]
    return f