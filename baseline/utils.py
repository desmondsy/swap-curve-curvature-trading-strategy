#!/usr/local/bin/python3.7
import numpy as np
import pandas as pd

def normalize_L(L):
    """Normalizes an entire Laplacian Matrix of size (N x N x s).

    Args:
        L (ndarray): Entire Laplacian Matrix of size (N x N x S).

    Note: 
        N is the number of tenors in the yield curve (15), and s
        is the numer of strides backtested 

    Returns:
        L_norm: Returns the normalized Laplacian.
    """    
    L_norm = np.zeros((L.shape[0],L.shape[1],L.shape[2]))
    for i in range(L.shape[2]):
        for j in range(L.shape[0]):
            L_norm[j,:,i] = L[j,:,i] / np.max(L[j,:,i])

    return L_norm

def getData(start_date, end_date, currency):
    """Returns a pandas.DataFrame object of the swap curve in specified currency denomination

    Args:
        start_date (str): Starting range of dataset to be used.
        end_date (str): Ending range of dataset to be used.
        currency (str): Specified currency denomination.

    Returns:
        df: pandas.DataFrame object of swap curve.
    """    
    datapath = r'/Users/desmond/OneDrive - Imperial College London/FYP/Python/cleanedscripts/data/swap_curve_set1.xlsx'

    df = pd.read_excel(datapath,
                    index_col=0,
                    header=[0,1,2])

    df.index = pd.to_datetime(df.index,format='%Y-%m-%d') 
    df = df.sort_index()
    df.columns = df.columns.droplevel(2)
    df = df.loc[start_date:end_date,[currency]].ffill().dropna()

    return df