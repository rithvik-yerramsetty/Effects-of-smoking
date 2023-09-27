import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

if not os.path.exists('../plots'):
    os.makedirs('../plots')



def import_data(path):
    with open(path, 'r') as f:
        num_cols = 0
        data = []
        for ind, line in enumerate(f.readlines()):
            if(ind == 0):
                num_cols = len(line.split())
            else:
                row_cols = line.split()
                relevant_data = list(map(float, row_cols[1:49]))
                data += [relevant_data]
        
    
    data = np.array(data, dtype=np.float64)
    return data

def get_N_matrix():
    r = [[1]*24 + [0]*24, [0]*24 + [1]*24, [0]*12 + [1]*12 + [0]*12 + [1]*12, [1]*12 + [0] *12 + [1]*12 + [0]*12 ]
    N = np.array(r, dtype=int).T
    
    return N

def get_D_matrix():
    r = [[0]*12 + [1]*12 + [0]*24, [1]*12 + [0]*36, [0]*36 + [1]*12, [0]*24 + [1]*12 + [0]*12]
    D = np.array(r, dtype=int).T
    
    return D
    
def compute_p_values(data, N, D):
    NTN_pinv = np.linalg.pinv(N.T@N)
    DTD_pinv = np.linalg.pinv(D.T@D)

    S_N = N@NTN_pinv@(N.T)
    S_D = D@DTD_pinv@(D.T)

    I = np.eye(N = S_N.shape[0])

    p_values = []
    i = 1
    for X in data:
        numerator = X.T @(I - S_N)@ X
        denominator = X.T @(I - S_D)@ X + 1e-7
        if(math.isnan(denominator) or denominator == 0):
            print(denominator)
        k = (X.shape[0] - np.linalg.matrix_rank(D) )/ (np.linalg.matrix_rank(D) - np.linalg.matrix_rank(N))
        f_value = k*((numerator/denominator) - 1)
        
        p_value = 1 - sp.stats.f.cdf(f_value, dfn=1, dfd=44)
        p_values.append(p_value)
        
    return p_values

def get_pvalues(data):
    N = get_N_matrix()
    D = get_D_matrix()

    p_values = compute_p_values(data, N, D)

    return p_values

def main(args):
    data = import_data(args['data_path'])
    p_values = get_pvalues(data)

    counts, bins = np.histogram(p_values, 25)
    plt.hist(bins[:-1], bins, weights=counts, edgecolor='black')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Histogram of p-values')
    plt.savefig(args['plot_path'])

if __name__ == '__main__':
    args = {
        "data_path": "../data/Raw Data_GeneSpring.txt",
        "plot_path": "../plots/plot.png",  # ensure that the path exists
    }
    main(args)