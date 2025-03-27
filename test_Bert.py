import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset

# Load the datasets
train_df = pd.read_csv("train_dataset.csv")
eval_df = pd.read_csv("eval_dataset.csv")

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_df))

# Preprocessing and creating custom dataset
class CorrectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data.iloc[index]['sentence']
        corrections = self.data.iloc[index]['corrections']
        
        encoded_inputs = tokenizer(sentence, corrections, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

train_dataset = CorrectionDataset(train_df, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):  # You should adjust the number of epochs
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Placeholder loss calculation (MSE as an example)
        target_labels = torch.zeros_like(outputs.logits)  # Replace with your actual target labels
        loss = torch.nn.MSELoss()(outputs.logits, target_labels)
        
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
