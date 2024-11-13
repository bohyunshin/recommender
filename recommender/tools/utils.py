import scipy
import time
import warnings
import numpy as np

def nonzeros(m, row):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]

def check_random_state(random_state):
    """Validate the random state.

    Check a random seed or existing numpy RandomState
    and get back an initialized RandomState.

    Parameters
    ----------
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    """
    # if it's an existing random state, pass through
    if isinstance(random_state, np.random.RandomState):
        return random_state
    # otherwise try to initialize a new one, and let it fail through
    # on the numpy side if it doesn't work
    return np.random.RandomState(random_state)

class ParameterWarning(Warning):
    pass

def check_csr(user_items):
    if not isinstance(user_items, scipy.sparse.csr_matrix):
        class_name = user_items.__clas__.__name__
        start = time.time()
        user_items = user_items.tocsr()
        warnings.warn(
            f"Method expects CSR input, and was passed {class_name} instead. "
            f"Converting to CSR took {time.time() - start} seconds",
            ParameterWarning,
        )
    return user_items


def mapping_dict(vals):
    u = sorted(list(set(vals)))
    mapping = {v:i for i,v in enumerate(u)}
    return mapping