# tutorial: https://www.geeksforgeeks.org/toxic-comment-classification-using-bert/
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set up GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv(r"toxic comment classification\data\train.csv")
print(data.head())

# Visualize class distribution
column_labels = data.columns.tolist()[2:]
label_counts = data[column_labels].sum().sort_values()

# Plot class distribution
plt.figure(figsize=(7, 5))
ax = sns.barplot(x=label_counts.values, y=label_counts.index, palette='viridis')
plt.xlabel('Number of Occurrences')
plt.ylabel('Labels')
plt.title('Distribution of Label Occurrences')
plt.show()

# Create subsets based on toxic and clean comments
train_toxic = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]

# Number of toxic and clean comments
num_toxic = len(train_toxic)
num_clean = len(train_clean)

# Plot toxic and clean comments distribution
plot_data = pd.DataFrame({'Category': ['Toxic', 'Clean'], 'Count': [num_toxic, num_clean]})
plt.figure(figsize=(7, 5))
ax = sns.barplot(x='Count', y='Category', data=plot_data, palette='viridis')
plt.xlabel('Number of Comments')
plt.ylabel('Category')
plt.title('Distribution of Toxic and Clean Comments')
ax.tick_params()
plt.show()

print("Toxic data:", train_toxic.shape)
print("Nontoxic data:", train_clean.shape)

# Balance data
print("Balancing...")
train_clean_sampled = train_clean.sample(n=16225, random_state=42)
dataframe = pd.concat([train_toxic, train_clean_sampled], axis=0)
dataframe = dataframe.sample(frac=1, random_state=42)

print("Toxic data:", train_toxic.shape)
print("Balanced Nontoxic data:", train_clean_sampled.shape)
print("Dataframe:", dataframe.shape)

# Split data into training, testing sets & validation sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataframe['comment_text'], dataframe.iloc[:, 2:], test_size=0.25, random_state=42)

# Validation set
test_texts, val_texts, test_labels, val_labels = train_test_split(
    test_texts, test_labels, test_size=0.5, random_state=42)

# Token and Encode Function
def tokenize_and_encode(tokenizer, comments, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for comment in comments:
        encoded_dict = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)

    return input_ids, attention_masks, labels

# Token Initialization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model = model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Tokenize and Encode for training, testing, and validation sets
input_ids, attention_masks, labels = tokenize_and_encode(tokenizer, train_texts, train_labels.values)
test_input_ids, test_attention_masks, test_labels = tokenize_and_encode(tokenizer, test_texts, test_labels.values)
val_input_ids, val_attention_masks, val_labels = tokenize_and_encode(tokenizer, val_texts, val_labels.values)

print('Training Comments :', train_texts.shape)
print('Input Ids         :', input_ids.shape)
print('Attention Mask    :', attention_masks.shape)
print('Labels            :', labels.shape)

# Create DataLoader for the balanced dataset
batch_size = 64
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Testing set
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Validation set
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print('Batch Size :', train_loader.batch_size)
Batch = next(iter(train_loader))
print('Each Input ids shape :', Batch[0].shape)
print('Input ids :\n', Batch[0][0])
print('Corresponding Decoded text:\n', tokenizer.decode(Batch[0][0]))
print('Corresponding Attention Mask :\n', Batch[1][0])
print('Corresponding Label:', Batch[2][0])

# Set up learning rate scheduler
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define accumulation steps
accumulation_steps = 2

# Train the model (without checkpointing)
def train_model(model, train_loader, optimizer, scheduler, device, num_epochs, accumulation_steps):
    model.train()
    print("Training...")
    total_loss = 0
    num_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Gradient accumulation
            total_loss += loss.item()
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate
            
            # Print current batch information
            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{num_batches}, Training Loss: {total_loss / (i + 1):.4f}')

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / num_batches:.4f}')

# Call the function to train the model
train_model(model, train_loader, optimizer, scheduler, device, num_epochs=3, accumulation_steps=2)

# Evaluate the Model
def evaluate_model(model, test_loader, device):
    model.eval()

    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        print("Evaluating...")
        for batch in test_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_probs_batch = torch.sigmoid(outputs.logits)
            predicted_probs.append(predicted_probs_batch.cpu().numpy())

            true_labels_batch = labels.cpu().numpy()
            true_labels.append(true_labels_batch)

    true_labels = np.concatenate(true_labels, axis=0)
    predicted_probs = np.concatenate(predicted_probs, axis=0)
    predicted_labels = (predicted_probs > 0.5).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='micro')
    recall = recall_score(true_labels, predicted_labels, average='micro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

# Call the function to evaluate the model on the test data
evaluate_model(model, test_loader, device)

# Save the tokenizer and model in the same directory
output_dir = "projects/toxic comment classification/Saved_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load the tokenizer and model from the saved directory
model_name = r"toxic comment classification\Saved_model"
Bert_Tokenizer = DistilBertTokenizer.from_pretrained(model_name)
Bert_Model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)

# Function to predict user input
def predict_user_input(input_text, model, tokenizer, device):
    user_input = [input_text]
    user_encodings = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")
    user_dataset = TensorDataset(user_encodings['input_ids'], user_encodings['attention_mask'])
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)
    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    labels_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = dict(zip(labels_list, predicted_labels[0]))
    return result

# Call the function with the correct arguments
text = input("Enter text: ")
result = predict_user_input(input_text=text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=device)
print(result)
