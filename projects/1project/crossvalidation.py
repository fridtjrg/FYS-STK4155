from sklearn.model_selection import train_test_split
from random import randint
import numpy as np
from ols_solver import OLS_solver


def group_indeces(n, random_groupsize = False, sections = 10):
    """Creates index pairs, sectioning off a range of indexes into smaller groups

    Args:
        n (int): Length of list to be grouped
        random_groupsize (bool): Whether groupsizes are equal or random

    Returns:
        list: List of pairs (start_index, end_index) for each group
    """

    if sections*3 > n:
        raise ValueError("n must be at least 3 times greater than the number of sections")
    
    inddex_cutoffs = [0]
    if random_groupsize:     #Randomly sized groups
        while len(inddex_cutoffs) < sections:
            new_randint = randint(2,n-2)
            if  all(x not in inddex_cutoffs for x in [new_randint-1,new_randint,new_randint+1]):
                inddex_cutoffs.append(new_randint)
        inddex_cutoffs.append(n)    
    else:   #Evenly sized groups
        const_group_size = n//sections
        for i in range(1,sections-1):
            inddex_cutoffs.append(i*const_group_size)
        inddex_cutoffs.append(n)        
    inddex_cutoffs.sort()

    #Make index pairs
    index_pairs = []
    for i in range(len(inddex_cutoffs)-1):
        index_pairs.append((inddex_cutoffs[i], inddex_cutoffs[i+1]-1))

    return index_pairs


def combine_groups(index_pairs, data):
    """Combines the index pairs into one np.array

    Args:
        index_pairs (list): List of pairs (start_index, end_index) for each group
        data (np.array): data to be indexed and recombined

    Returns:
        np.array(): combined np.array from the given indexes
    """

    #print("Index pairs: ",index_pairs)
    #Only one pair "automatically" removes top list layer... >:(
    
    if index_pairs.__class__ is tuple:
        index_pairs = [index_pairs]
        #print("Input tuple converted to list(tuple): ",index_pairs)

    #Get length of combined groups ndarray
    len_comb_groups = 0
    for pairs in index_pairs:
        len_comb_groups+=pairs[1]-pairs[0]+1
    #print("Lenght of combined groups: ",len_comb_groups)

    if len(data.shape) == 1:
        combined_groups = np.ndarray(shape=(len_comb_groups,))
        #print("Shape of combined groups: ", combined_groups.shape)

        """if index_pairs[0][0] == 0:
            combined_groups[0:index_step+index_interval+1] = data[pairs[0]:pairs[1]+1]
            index_step = index_pairs[0][1]-index_pairs[0][0]
        """
        index_step = 0
        for i in range(len(index_pairs)):
            pairs = index_pairs[i]
            #print("Pairs: ",pairs)
            index_interval = pairs[1]-pairs[0] 
            combined_groups[index_step:index_step+index_interval+1] = data[pairs[0]:pairs[1]+1]
            #print(combined_groups)
            index_step+=index_interval+1


    elif len(data.shape) == 2:
        combined_groups = np.ndarray(shape=(len_comb_groups,data.shape[1]))

        index_step = 0
        for i in range(len(index_pairs)):
            pairs = index_pairs[i]
            #print("Pairs: ",pairs)
            index_interval = pairs[1]-pairs[0] 
            for j in range(index_interval+1):
                combined_groups[index_step+j,:] = data[pairs[0]+j,:]
            #combined_groups[index_step:index_step+index_interval+1,:] = data[pairs[0]:pairs[1]+1,:]
            #print(combined_groups)
            index_step+=index_interval+1

    return combined_groups


def cross_validation(k, designmatrix, datapoints, random_groupsize = False):
    """Divides the dataset into k folds of n groups and peforms cross validation

    Args:
        k (int): number of folds
        n_groups (int): number of groups
        designmatrix (np.array(n,m)): The design matrix
        datapoints (np.array(n)): datapoints
        random_groupsize (bool, optional): Whether group size is even or random. Defaults to False.

    Returns:
        [type]: [description]
    """
    
    #Defaults to using 2/10 groups for testing
    #Currently only tested for 5 folds, 10 groups

    n = len(datapoints)
    n_groups = 2*k
    index_pairs = group_indeces(n,True, n_groups)
    #print("Index pairs: ",index_pairs)

    CV_stats = []

    ols_beta_set = []
    MSE_train_set = []
    MSE_test_set = []

    for i in range(0,n_groups-1,2):

        test_index_pairs = []
        test_index_pairs.append(index_pairs[i])
        test_index_pairs.append(index_pairs[i+1])
        #print("TEST index pairs: ",test_index_pairs)
        train_index_pairs = list(index_pairs)
        del train_index_pairs[i]
        del train_index_pairs[i]
        #print("TRAIN index pairs: ",train_index_pairs)

        X_train = combine_groups(train_index_pairs, designmatrix)
        X_test = combine_groups(test_index_pairs, designmatrix)
        z_train = combine_groups(train_index_pairs, datapoints)
        z_test = combine_groups(test_index_pairs, datapoints)

        ols_beta, MSE_train, MSE_test = OLS_solver(X_train, X_test, z_train, z_test)
        ols_beta_set.append(ols_beta)
        MSE_train_set.append(MSE_train)
        MSE_test_set.append(MSE_test)
    
    return np.mean(ols_beta_set), np.mean(MSE_train_set), np.mean(MSE_test_set)
