import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel
from enron_dataloader import CustomDataset, DEFAULT_CFG
from models.bert import BERTClass
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import pandas as pd


def import_dataset(file_path=None):
    if file_path is None:
        file_path = "./dataset/enron_spam_data.csv"
    df = pd.read_csv(file_path)
    df["label"] = df["Spam/Ham"]
    new_df = df[["Subject", "Message", "label"]]
    print(new_df.head())
    return new_df


def train_bert(dataset_df, tokenizer, cfg):
    device = 'cuda' if cuda.is_available() else 'cpu'
    # Creating the dataset and dataloader for the neural network

    train_size = 0.95
    train_dataset = dataset_df.sample(frac=train_size, random_state=200)
    test_dataset = dataset_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.sample(frac = 0.05, random_state=200)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(dataset_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, cfg.MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, cfg.MAX_LEN)

    train_params = {"batch_size": cfg.TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": cfg.VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Training
    for epoch in tqdm(range(cfg.EPOCHS)):
        # for each batch do this
        model.train()
        for _,data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float).unsqueeze(1)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def validation(epoch):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float).unsqueeze(1)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    # Validation
    # for epoch in range(cfg.EPOCHS):
    outputs, targets = validation(0)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


if __name__ == "__main__":
    dataset_df = import_dataset("./dataset/enron_spam_data.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    config = DEFAULT_CFG
    
    train_bert(dataset_df, tokenizer, config)
