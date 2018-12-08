from sklearn.model_selection import cross_validate


def cross_validation(learner, Xtr, Ytr, folds=10, n_jobs=10, fit_params=None):
    """
    
    """

    return cross_validate(learner, Xtr, Ytr, cv=folds, n_jobs=n_jobs)

