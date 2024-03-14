# Toxic-Comment-Classification
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

Tutorial: https://www.geeksforgeeks.org/toxic-comment-classification-using-bert/

This code implements a toxic comment classification model using DistilBERT

## Setup and Data Loading
- Imports necessary libraries including NumPy, pandas, PyTorch, Transformers, and Seaborn.
- Loads the training data from a CSV file, visualizes the class distribution, and balances the dataset.

## Tokenization and Encoding
- Tokenizes and encodes the comments using the DistilBERT tokenizer.
- Prepares input tensors (input_ids) and attention masks for the training, testing, and validation datasets.

## Model Initialization
- Initializes a DistilBERT model for sequence classification.
- Moves the model to the available device (CPU or GPU).

## Training Loop
- Defines a training function that iterates over the training DataLoader.
- Computes the loss, performs gradient accumulation, and updates model parameters using AdamW optimizer.
- Adjusts the learning rate using a linear scheduler.

## Evaluation
- Defines an evaluation function to assess the model's performance on the test dataset.
- Computes accuracy, precision, and recall scores for the classification task.

## Saving and Loading
- Saves the trained model and tokenizer to a specified directory.
- Loads the saved model and tokenizer for inference.

## Inference
- Implements a function to predict the toxicity labels for user input text using the loaded model and tokenizer.
- Displays the predicted labels for the provided text input.

## Additional Information
- the training takes a long time, so if you have already trained the model, it would be better to load the saved trained model for the inference
