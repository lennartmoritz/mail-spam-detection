import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel
from easydict import EasyDict as edict

# Sections of config

# Defining some key variables that will be used later on in the training
DEFAULT_CFG = edict({
    "MAX_LEN": 256,
    "TRAIN_BATCH_SIZE": 8,
    "VALID_BATCH_SIZE": 4,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-05,
})

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        # self.comment_text = dataframe.comment_text
        self.subject = dataframe.Subject
        self.message = dataframe.Message
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.subject)

    def __getitem__(self, index):
        # Remove "\n" from subject and message
        subject = str(self.subject[index])
        subject = " ".join(subject.split())
        message = str(self.message[index])
        message = " ".join(message.split())

        combined_mail = subject + " " + message

        inputs = self.tokenizer.encode_plus(
            combined_mail,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index] == "spam", dtype=torch.float)
        }