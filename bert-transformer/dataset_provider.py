from models.custom_dataset import CustomDataset
from bert_config import DEFAULT_CFG
import pandas as pd

if __name__ == "__main__":
    file_path = '../dataset/enron_spam_data.csv'
        
    df = pd.read_csv(file_path)
    df["label"] = df["Spam/Ham"]
    new_df = df[["Subject", "Message", "label"]]

    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)

    train_dataset.to_csv('../dataset/train.csv', index=False)

    test_dataset.to_csv('../dataset/test.csv', index=False)
