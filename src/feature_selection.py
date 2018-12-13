from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2

def extract_lsa(n_components):
    """ Take the tf_idf matrix as input and sizes it down using 

    """
    def __lsa(Xtr, Xte):
        svd = TruncatedSVD(n_components)
        lsa = make_pipeline(svd, Normalizer(copy=False))

        lsa.fit(Xtr)
        Xtr_lsa = lsa.transform(Xtr)
        Xte_lsa = lsa.transform(Xte)

        return Xtr_lsa, Xte_lsa

    return __lsa


def k_best(Ytr, Yte, k_best):
    def __k_best(Xtr, Xte):
        Select = SelectKBest(chi2, k=k_best).fit(Xtr, Ytr)
        XtrS = Select.transform(Xtr)
        XteS = Select.transform(Xte)

        return XtrS, XteS
    return __k_best
