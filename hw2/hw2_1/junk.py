from dataloader import WeirdDataset
from torch.utils.data import DataLoader

if __name__=='__main__':
    loader = DataLoader(WeirdDataset('testing_label.json', 'testing_data'), batch_size=1)
    for X,y in loader:
        file_name, data = X
        print(type(file_name), type(data))
        print(file_name)
        break


