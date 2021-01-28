import load_data
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def vote(model, dataset_path, Participant_ID, target_df):
    '''
    model: a trained model on 5s samples
    '''
    # filenames = files from Participant_ID
    # dataset = DaicWOZDataset(filenames, dataset_path, target_df)
    # dataloader = DataLoader(dataset)
    # Make model not train
    # for step, (X, y) in dataloader: # y should theoretically always be the same
        # output = model.predict(X), blabla
        # outputs.append(output)
    # mean = output.mean()
    

class AggregatorDataLoader(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset)

if __name__ == "__main__":
    dataset_path = os.path.join('dataset_cut')
    target_df_path = os.path.join('targets.csv')
    target_df = pd.read_csv(target_df_path)

    filenames = os.listdir(dataset_path)
    dataset = load_data.DaicWOZDataset(filenames, dataset_path, target_df)
    dataLoader = AggregatorDataLoader(dataset)
    for x, y in enumerate(dataLoader):
        print(y)