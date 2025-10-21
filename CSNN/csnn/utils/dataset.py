import torch
from torch.utils.data import Dataset
from glob import glob
import math
from sklearn.model_selection import train_test_split

def splitting_debug(train_list, valid_list):
    """Checks if splits are done correctly"""
    counter = 0
    for fn in train_list:
        if fn in train_list and fn in valid_list:
            fn_list.append(fn)
            counter += 1

    if counter != 0:
        print('Warning! Data Leakage...')
        print(f'{counter} files have been found in both datasets: \n{fn_list} ')
    else:
        print(f'Train dataset has {len(train_list)} files. Valid dataset has {len(valid_list)} files.')

def retrieve_patient_index(fn):
    """Extract Patient ID from path"""
    id = fn[::-1].index("\\")
    fn = fn[::-1][:id][::-1]
    _id = fn.index('_')
    patient_index = fn[:_id]
    return patient_index

def retrieve_unhealthy_datasets_length(patients_fn):
    """Retrieve length of unhealthy donors (since new dataset generation create smaller number of new datasets for healthy donors)"""
    max_l = 0 
    for key, value in patients_fn.items():
        if len(value) > max_l:
            max_l = len(value) # check the max len to extract datasets that comes from healthy patients
    return max_l

def retrieve_healthy_patients(patients_fn, pat_indices):
    """Extracts healthy donors and remove the from those that have to be permuted"""
    max_l = retrieve_unhealthy_datasets_length(patients_fn)
    healthy_patients = []
    for key, value in patients_fn.items():
        if len(value) < max_l:
            healthy_patients.append(key) # store healthy patients
            pat_indices.remove(key) # remove healthy patients
    return healthy_patients, pat_indices

def final_variable_extraction(patients_perm_idx, healthy_perm_idx, pat_indices, patients_fn):
            """Extrats variables needed from the model"""
            #retrieve patients indices from permutation
            perm = []
            for idx in patients_perm_idx:
                perm.append(pat_indices[idx])

            # extracts paths from fictionaty of patients
            patients_fn_list = []
            for pat in list(perm):
                patients_fn_list += patients_fn[str(pat)]

            # adds one healthy patient at the beginnig and one at the end (hence one in the training and one in the validation in eah fold)
            n_test = len(patients_fn_list)
            add_n_test = 0
    
            for i, h_pat in enumerate(healthy_perm_idx):

                if i % 2: #if odd append ( if i % 2 == 0, then i % 2 == 0 is False, hence even)
                    patients_fn_list += patients_fn[str(h_pat.item())]
                    
                else: # if even put at the start 
                    patients_fn_list = patients_fn[str(h_pat.item())] + patients_fn_list
                    add_n_test += len(patients_fn[str(h_pat.item())])
            
            n_test += add_n_test
    
            return patients_fn_list, n_test, perm
    
class PointsetDataset(Dataset):
    '''Extracts datasets and its properties'''
    def __init__(self, datasets, sample_size=None,
                 ignore_proportions=False, train=True,  test = False,
                 train_proportions=None, fold=None,
                 random_state=42):
        
        self.datasets = datasets
        self.sample_size = sample_size
        self.ignore_proportions = ignore_proportions
        self.fold=fold

        if test: # if test datasets have to be processed, then train have to set as false
            train = False
            
        # retrieves all datapath 
        files = []
        pat_indices = []
        patients_fn = {}
        
        for data in self.datasets:
            for fn in sorted(glob(f"{data}/*")): # for datafile in the directory
                files.append(fn)

                # let the procedure works only for train and validation
                if not test:
                    pat_id = retrieve_patient_index(fn) 

                    # divide datasets per donor and collect them in a dictionary
                    if pat_id not in patients_fn.keys(): 
                        patients_fn[pat_id] = []
                    patients_fn[pat_id].append(fn)
                
        self.indices = files #files_path

        if not test:

            pat_indices = list(patients_fn.keys())
            healthy_patients, pat_indices = retrieve_healthy_patients(patients_fn, pat_indices)
            
            if train_proportions is not None:
                n_train = int(math.ceil(train_proportions * len(pat_indices))) #takes the number of files to assign to the train 
                n_test = int((1 - train_proportions) * len(pat_indices))

            elif fold is not None:
                if fold < 2:
                    print('Minimum is 2 folds. Fold set to 2')
                    fold = 2
                
                n_train = int((len(path_indices) / train_proportions)*(fold-1))
                n_test = len(path_indices) - n_train
    
            # generates permutations permutations of dpatients indices
            patients_perm_idx = torch.randperm(len(pat_indices), generator=torch.Generator().manual_seed(random_state)) 
            healthy_perm_idx = torch.randperm(len(healthy_patients), generator=torch.Generator().manual_seed(random_state)) 

            self.indices, n_test, perm = final_variable_extraction(patients_perm_idx, healthy_perm_idx, pat_indices, patients_fn)
            
        else:
            if train_proportions is not None:
                n_train = int(math.ceil(train_proportions * len(self.indices)))
                n_test = int(math.ceil((1-train_proportions) * len(self.indices)))
                print(len(self.indices))
    
            perm = torch.randperm(len(self.indices), generator=torch.Generator().manual_seed(random_state))
    


        # common to train, validation and testing
        if fold is not None:
                # for each fold, file paths are retrieved from files list and data is loaded. ['y'] is the key from the dict saved in that file
                # the following code retrieves the y's of the datasets
                ys = []
                for idx in self.indices:
                    y = torch.load(idx, weights_only=False)["y"] 
                    ys.append(y)
                # [torch.load(idx, weights_only=False)["y"] for idx in self.indices]
    
                # splits indices in train and validation (test on the leaved out fold or percentage)
                train_index, test_index = train_test_split(list(range(len(self.indices))), stratify=ys, test_size=n_test, random_state=random_state)
                
                splitting_debug(train_index, test_index)
                
                if train:
                    self.indices = [self.indices[i] for i in train_index] 
                    print(f'train self_indices: {self.indices}')
                else:
                    self.indices = [self.indices[i] for i in test_index]
                    print(f'valid self_indices: {self.indices}')
                    
                
        elif train_proportions is not None:
                if train:
                    self.indices = [self.indices[i] for i in perm[:n_train]] # retrieves indices of data files for train
                else: # validation
                    self.indices = [self.indices[i] for i in perm[n_train:]] 
            
        # list of data (list of tensors)
        self.data = [torch.load(fn, weights_only=False) for fn in self.indices]

        # if sample size is not None, resampling is performed (no replaement here)
        if self.sample_size is not None:
            for data in self.data:
                perm = torch.randperm(data["X"].shape[0])[:self.sample_size]
                data["X"] = data["X"][perm,:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data = self.data[index]
        X = data["X"]
        y = data["y"]
        if (not self.ignore_proportions) and "proportion" in data:
            prop = data["proportion"]
            return X, y, prop
        return X, y

    @property
    def X(self):
        return torch.stack([c['X'] for c in self.data])

    @property
    def y(self):
        return torch.tensor([c['y'] for c in self.data])

    @property
    def proportion(self):
        return torch.tensor([c['proportion'] for c in self.data])

    @property
    def idx(self):
        return torch.tensor([c['idx'] for c in self.data])

if __name__ == "__main__":
    dataset_train = PointsetDataset(["data/B-ALL/positive_prop", "data/B-ALL/negative_prop"], sample_size=300000, train_proportions=0.75, train=True)
    dataset_valid = PointsetDataset(["data/B-ALL/positive_prop", "data/B-ALL/negative_prop"], sample_size=300000, train_proportions=0.75, train=False)

    print(dataset_train.X.shape)
    print(dataset_valid.X.shape)
    print(set(dataset_train.idx).isdisjoint(set(dataset_valid.idx)))

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    # for batch in dataloader:
    #     print(batch)
