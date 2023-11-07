from transformers import BertTokenizer, BertModel
import pandas as pd

def load_spam_sample(file_path=None):
    """Load a sample from the dataset"""
    if file_path is None:
        file_path = "./dataset/enron1/spam/0006.2003-12-18.GP.spam.txt"
    text_content = None
    with open(file_path, "r") as file:
        text_content = file.read()
    return text_content

def load_model_and_run(input_text=None):
    """Load and run the model"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased")
    if input_text is None:
        input_text = "Replace me by any text you'd like."
    encoded_input = tokenizer(input_text, return_tensors='pt')
    outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state
    # _, last_hidden_states = model(**encoded_input)

    print("encoded tokens: \t", encoded_input["input_ids"].shape)
    # print(encoded_input)
    # print("type: \t", type(last_hidden_states))
    # print("len: \t", len(last_hidden_states))
    try:
        print("shape: \t", last_hidden_states.shape)
    except:
        pass
    print("output: \t", last_hidden_states)
    return last_hidden_states

def import_dataset(file_path=None):
    if file_path is None:
        file_path = "./dataset/enron_spam_data.csv"
    df = pd.read_csv(file_path)
    df['label'] = df['Spam/Ham']
    new_df = df[['Subject', 'Message', 'label']]
    print(new_df.head())
    return new_df
    


if __name__ == "__main__":
    text = load_spam_sample()
    print(text)
    load_model_and_run(text)
    # import_dataset()



# TODO:
# - Still have to fix max input tokens of 512
#   - Bert automatically truncates text according to https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification
# - Add multi classifer layer with training capabilities
#   - Notebook already available at https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb#scrollTo=DegHNyIEQxB2
# - Ressources: https://huggingface.co/docs/transformers/model_doc/bert#overview