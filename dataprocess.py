import os 
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataSplit(object):
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class SingleLoader(Dataset):
    def __init__(self, path):
        self.data_list = []
        self.file2memory(path)

    def __getitem__(self, index):
        text, note, spec_prev, spec = self.data_list[index].values()

        x = (text, note)
        y_prev = spec_prev
        y = spec

        return x, y_prev, y

    def __len__(self):
        return len(self.data_list)

    def file2memory(self, path):
        for basename in sorted(os.listdir(path)):
            data = torch.load(os.path.join(path, basename))
            self.data_list.extend(data)

class MultiLoader(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_index = 0
        self.file_indices = torch.load(os.path.join(path, 'indices.pt'))

    def __getitem__(self, index):
        file_list = sorted(os.listdir(self.path))
        filename = os.path.join(self.path, file_list[self.file_index])
        data = torch.load(filename)

        return_index = index
        if self.file_index > 0:
            return_index = index - self.file_indices[self.file_index - 1]

        text, note, spec_prev, spec = data[return_index].values()

        x = (text, note)
        y_prev = spec_prev
        y = spec

        return x, y_prev, y

    def __len__(self):
        return self.file_indices[-1]

class InferLoader(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        text, note = self.data[index].values()

        return (text, note)

    def __len__(self):
        return len(self.data)

def load_train(config):
    if config.data_mode == 'single':
        Loader = SingleLoader
    elif config.data_mode == 'multi':
        Loader = MultiLoader
    else:
        raise AssertionError('Please use valid data mode, \'single\' or \'multi\'.')

    dataset_train = Loader(os.path.join(config.feature_path, 'train'))
    dataset_valid = Loader(os.path.join(config.feature_path, 'valid')) 

    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size,
                                  shuffle=True, num_workers=config.num_proc)
    dataloader_valid = DataLoader(dataset_valid, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=config.num_proc)

    return DataSplit(dataloader_train, dataloader_valid, None)

def load_infer(data, config):
    dataset = InferLoader(data)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return dataloader