from dataloader import WeirdDataset
from torch.utils.data import DataLoader

if __name__=='__main__':
    loader = DataLoader(WeirdDataset('testing_label.json', 'testing_data'), batch_size=2)
    
    s = set()
    for X,y in loader:
        file_name, data = X
        print(type(file_name), type(data))
        
        for x in file_name:
            s.add(x)
        print(file_name)

        print(y.shape)

    print(len(s))


