import time
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def nonzeros(m: csr_matrix, row: int):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


def check_random_state(random_state: int):
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


def check_csr(user_items: csr_matrix):
    if not isinstance(user_items, csr_matrix):
        class_name = user_items.__clas__.__name__
        start = time.time()
        user_items = user_items.tocsr()
        warnings.warn(
            f"Method expects CSR input, and was passed {class_name} instead. "
            f"Converting to CSR took {time.time() - start} seconds",
            ParameterWarning,
        )
    return user_items


def mapping_dict(vals: NDArray):
    u = sorted(list(set(vals)))
    mapping = {v: i for i, v in enumerate(u)}
    return mapping


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0


def binarize(
    c: NDArray,
):
    """
    Binarizes input NDArray making dataset as implicit.
    In implicit dataset, even if an user interacted with an item greater than one time,
    the user is regarded as interacting with an item one time.
    This model assumption inherently has disadvantage, therefore fixes it with `alpha` parameter.

    Args:
         c (NDArray): NDArray to be binarized.

    Returns (NDArray):
        Binarized NDArray.
    """
    return np.where(c >= 1, 1, 0)
