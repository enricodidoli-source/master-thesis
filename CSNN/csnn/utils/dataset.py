import torch
from torch.utils.data import Dataset
from glob import glob
import math
from sklearn.model_selection import train_test_split

class PointsetDataset(Dataset):
    '''Extracts datasets and its properties'''
    def __init__(self, datasets, sample_size=None,
                 ignore_proportions=False, train=True,
                 train_proportions=None, fold=None,
                 random_state=42):
        self.datasets = datasets
        #print(datasets)
        self.sample_size = sample_size
        self.ignore_proportions = ignore_proportions
        self.fold=fold
        #self.indices = [fn for d in self.datasets for fn in glob("%s/*" % d)]
       # print('ok')
        files = []
        for data in self.datasets:
            #print(f'data:{data}')
            for fn in glob("%s/*" % data): # for datafile in the directory
                #print(f'fn: {fn}')
                files.append(fn)
        self.indices = files #number of files
        #print(self.indices)
        #print(1)
        if train_proportions is not None:
            n_train = int(math.ceil(train_proportions * len(self.indices))) #takes the number of files to assign to the train 
            n_test = int(math.ceil((1 - train_proportions) * len(self.indices)))
            #print(n_test)
            #print(len(self.indices))

        perm = torch.randperm(len(self.indices), generator=torch.Generator().manual_seed(random_state)) # generates permutations permutations of data indices

        if fold is not None:
            ys = [torch.load(idx, weights_only=False)["y"] for idx in self.indices]

            train_index, test_index = train_test_split(list(range(len(self.indices))), stratify=ys, test_size=n_test, random_state=random_state)

            if train:
                self.indices = [self.indices[i] for i in train_index] 
            else:
                self.indices = [self.indices[i] for i in test_index]
                
        elif train_proportions is not None:
            if train:
                self.indices = [self.indices[i] for i in perm[:n_train]] # retrieves indices of data files for train
            else: # validation
                self.indices = [self.indices[i] for i in perm[n_train:]] 


        self.data = [torch.load(fn, weights_only=False) for fn in self.indices]

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
