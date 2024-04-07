from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class CustomDataSet(Dataset):
    """Load data under folders"""
    def __init__(self, args, train=True):
        self.main_dir = args.main_dir 
        self.task = args.task 
        self.augmentation = args.augmentation

        if train:
            data_path = self.main_dir + self.task + "/data_train.npy"
            label_path = self.main_dir + self.task + "/label_train.npy"
        else:
            data_path = self.main_dir + self.task + "/data_test.npy"
            label_path = self.main_dir + self.task + "/label_test.npy"
        
        self.data = torch.from_numpy(np.load(data_path))
        self.label = torch.from_numpy(np.load(label_path)).to(torch.long) # in cls task, (N,), in seg task, (N, 10000), N is the number of objects
        
    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        point_set = self.data[idx]
        if self.augmentation:
            theta = torch.tensor(np.random.uniform(0, np.pi*2))
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]]
            )
            point_set[:, [0, 2]] = torch.mm(point_set[:, [0, 2]], rotation_matrix)
            point_set += torch.randn_like(point_set) * 0.02 # random jitter

        return point_set, self.label[idx]

def get_data_loader(args, train=True):
    """
    Creates training and test data loaders
    """
    dataset = CustomDataSet(args=args, train=train)
    dloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)
    
    return dloader
