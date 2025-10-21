from sklearn.neighbors import KernelDensity
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class DensityDifference:
    def __init__(self, alpha, bandwidth=400, kernel='gaussian', sample_size=10000, disable_tqdm=False, n_jobs=10):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.sample_size = sample_size
        self.disable_tqdm = disable_tqdm
        self.kdeneg = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kdepos = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.alpha = alpha
        self.n_jobs=n_jobs

    def fit(self, x_train, y_train, concatenate=True):

        ### Healthy samples ###
        allneg_sample = []
        for i, xt in enumerate(x_train): #[y_train == 0]:
            if y_train[i] == 0:
                xt_array = np.array(xt)
                
                # 1. get the number of cells in the dataset
                num_cells = xt_array.shape[0]
                
                # 2. perform a permutation of num_cells (no replacement)
                # Ex: if num_cells = 5 → [3, 1, 4, 0, 2]
                perm_index = np.random.permutation(num_cells)
                
                # 3. Selects only the first sample_size elements
                # Ex: if self.sample_size = 2 → [3, 1]
                selected_index = perm_index[:self.sample_size]
                
                # 4. Usa gli indici selezionati per estrarre le righe corrispondenti dall'array
                # La virgola e i due punti (:, ) significano "prendi tutte le colonne"
                neg_sample = xt_array[selected_index, :]

                #neg_sample = xt[np.random.permutation(xt.shape[0])[:self.sample_size],:] # samples with replacements from each sample with a disease
                allneg_sample.append(neg_sample)
        allneg_sample_concat = np.concatenate(allneg_sample) # concatanates all arrays in a single array 
        self.kdeneg.fit(allneg_sample_concat)  # analyzes the samples using gaussian kernel function 

        print(f'all_neg: {len(allneg_sample)}')
        print(f'all_neg: {len(allneg_sample[0])}')
        
        ### Samples of donor with disease ###
        allpos_sample = []
        allpos_train = []

        counter = 0
        
        # for each non-healthy donor
        # tqdm shows the progress bar
        for xt in tqdm(x_train[y_train == 1], disable=self.disable_tqdm):

            print(f'Iteration {counter}: shape={xt.shape}\n')

            kp = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
            counter += 1
        
            # performs a sample of 2*sample_size cells
            perm = np.random.permutation(xt.shape[0])
            
            print(f'perm1 {len(perm)}')
            print(f'next: {2*self.sample_size}')
            
            perm = perm[:2*self.sample_size]
            print(f'perm2 {len(perm)}')
            
            perm_train = perm[:self.sample_size]
            perm_sample = perm[self.sample_size:]
            print(f'perm_train: {perm_train}')
            print(f'perm_sample: {perm_sample}')
            ap_train = xt[perm_train,:]
            ap_sample = xt[perm_sample,:]

            print(f'check ap_train : {len(ap_train)}')
            print(f'check ap_sample : {len(ap_sample)}')
            allpos_train.append(ap_train)
            allpos_sample.append(ap_sample)

        allpos_train_concat = np.concatenate(allpos_train)
        
        self.kdepos.fit(allpos_train_concat)
        if concatenate:
            allpos_sample = np.concatenate(allpos_sample)
            allneg_sample = allneg_sample_concat
            print(f'all_neg2: {len(allneg_sample)}')
        else:
            allpos_sample = np.stack(allpos_sample)
            allneg_sample = np.stack(allneg_sample)
            print(f'all_neg3: {len(allneg_sample)}')
            
        print(len(allpos_sample))
        print(f'neg: {len(allneg_sample)}')
        return allpos_sample, allneg_sample

    def score_samples(self, X, return_scores=False):
        X_split = np.array_split(X, self.n_jobs)

        #computes the scores of negative samples (healthy). Parallel is uset to work with multiple iterations at the same time
        score_negs = Parallel(n_jobs=self.n_jobs)(delayed(self.kdeneg.score_samples)(x) for x in tqdm(X_split, disable=self.disable_tqdm))
        score_neg = np.concatenate(score_negs)

        #computes the scores of positive samples (blast)
        score_poss = Parallel(n_jobs=self.n_jobs)(delayed(self.kdepos.score_samples)(x) for x in tqdm(X_split, disable=self.disable_tqdm))
        score_pos = np.concatenate(score_poss)

        diff =  1 - np.exp(np.log(1 - self.alpha) + score_neg - score_pos)

        if return_scores:
            return diff, (score_pos, score_neg)
        return diff
