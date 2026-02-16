import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "recommender.libs.utils._csr_ops",
        ["recommender/libs/utils/_csr_ops.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    packages=[],
    ext_modules=cythonize(extensions),
)
