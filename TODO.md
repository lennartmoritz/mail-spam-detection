# Open Tasks

## Data Import

Import the enron data. enron_to_csv.py should probably work, the current issue is the download from the web. Immediate workaround: Download the dataset manually and then use the script to convert it to csv. 

## SpamAssasin Dataset

Look into preprocessing the SpamAssasin Dataset, so it can be used as a training- and/or test-dataset. 

Get first "Subject:"
Find first empty line, where the next line does not start with placeholder

Email begins afterwards

Canceled: ~~IF first token is <html> -> apply beautifulsoup~~

adapt csv converter and dataloader, so spamassasin can be loaded in to the dataloader

## Data Exploration 

Look into the dataset and look for specific qualities such as 

- Differences between SPAM and HAM Emails 
  - word count 
  - reply chains 
  - formatting

- Differences between ENRON and SpamAssasin

## Saving the model and Inference

Save the trained model.

## Train the model with the complete dataset 

## Visualize Attention of the model (Optional)

https://github.com/jessevig/bertviz 

## Other forms of visualization? 

## SpamBERT auf huggingface

## compare accuracy of different epochs ? --> Visualization?



