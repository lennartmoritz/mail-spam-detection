import os
import os.path as osp
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel
from models.custom_dataset import CustomDataset
from models.bert import BERTClass
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
# DEFAULT CONFIGURATION
from bert_config import DEFAULT_CFG

def import_dataset(file_path=None):

    if file_path is None:
        file_path = '../dataset/enron_spam_data.csv'
        
    df = pd.read_csv(file_path)
    df["label"] = df["Spam/Ham"]
    new_df = df[["Subject", "Message", "label"]]
    #print(new_df.head())
    return new_df

def create_datasets(dataset_df, tokenizer):
    train_size = 0.999
    train_dataset = dataset_df.sample(frac=train_size, random_state=200)
    test_dataset = dataset_df.drop(train_dataset.index).reset_index(drop=True)
    
    # TODO remove this, when whole dataset should be used for training
    # current only use a fraction of the dataset to train for performance reasons
    train_dataset = train_dataset.sample(frac = 0.001, random_state=200)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(dataset_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, DEFAULT_CFG.MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, DEFAULT_CFG.MAX_LEN)

    return training_set, testing_set

def create_loaders(training_set, testing_set):
    train_params = {"batch_size": DEFAULT_CFG.TRAIN_BATCH_SIZE, "shuffle": True, "num_workers": 0}
    test_params = {"batch_size": DEFAULT_CFG.VALID_BATCH_SIZE, "shuffle": True, "num_workers": 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    return training_loader, testing_loader

def train(device, model, optimizer, epoch):
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
    #
    os.makedirs("./models/weights/", exist_ok=True)
    #model.save_pretrained(f"./models/weights/spam_bert_base_epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"./models/weights/spam_bert_base_epoch_{epoch}.pth")
    

def validate():
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

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

if __name__ == "__main__":
    # enable GPU if available
    device = 'cuda' if cuda.is_available() else 'cpu'

    # import the SPAM/HAM data 
    dataset_df = import_dataset("../dataset/enron_spam_data.csv")

    # use the pretrained bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    
    # create a training set and a testing set from our SPAM/HAM data
    training_set, testing_set = create_datasets(dataset_df, tokenizer)

    # create the respective dataloaders
    training_loader, testing_loader = create_loaders(training_set, testing_set)

    # initialize the BERT Model
    model = BERTClass()
    model.to(device)

    # load model if it was already saved
    #os.makedirs("./models/weights/")
    if osp.exists("./models/weights/spam_bert_base.pth"):
        model.load_state_dict(torch.load("./models/weights/spam_bert_base.pth"))


    # init the optimizer
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=DEFAULT_CFG.LEARNING_RATE)

    DO_TRAIN = False
    if DO_TRAIN:

        # train the model for as many times as defined in config
        for epoch in range(DEFAULT_CFG.EPOCHS):
            train(device, model, optimizer, epoch)

        #model.save_pretrained(f"./models/weights/spam_bert_base.pth")
        torch.save(model.state_dict(), "./models/weights/spam_bert_base.pth")
    
    # evaluate the performance of the model
    outputs, targets = validate()
    outputs = np.array(outputs) >= 0.5

    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    
