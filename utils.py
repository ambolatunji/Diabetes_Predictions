import numpy as np 
import pandas as pd 
from pandas.api.types import is_numeric_dtype
# from pandas_profiling import ProfileReport



def isNumerical(col):
    return is_numeric_dtype(col)



if __name__ == '__main__':
    main()
    