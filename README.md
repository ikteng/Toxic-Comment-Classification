# Toxic-Comment-Classification
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

Tutorial: https://www.geeksforgeeks.org/toxic-comment-classification-using-bert/

This code essentially performs the following tasks: loading data, preparing the dataset, training a DistilBERT model for toxic comment classification, and providing a way to predict toxicity for user-input comments.

## 1. Load Data and Visualize Class Distribution:
- Reads the training data from a CSV file using Pandas.
- Displays the first few rows of the dataset.
- Visualizes the distribution of toxic comment labels using bar plots.


## 2. Create Subsets and Balance Data:
- Creates subsets of toxic and clean comments based on label sums.
- Balances the data by sampling an equal number of clean comments.
- Concatenates toxic and sampled clean comments to create a balanced dataset.


## 3. Split Data into Training, Testing, and Validation Sets:
- Splits the data into training, testing, and validation sets using scikit-learn's train_test_split function.


## 4. Tokenize and Encode Function:
- Defines a function (tokenize_and_encode) to tokenize and encode comments using the DistilBERT tokenizer.


## 5. Token Initialization and Model Setup:
- Initializes the DistilBERT tokenizer and model for sequence classification.
- Sends the model to the specified device (GPU or CPU).


## 6. Initialize Optimizer and Tokenize/Encode Data:
- Sets up the AdamW optimizer for training.
- Tokenizes and encodes the training, testing, and validation sets.


## 7. Create DataLoaders:
- Creates DataLoader objects for the balanced training set, testing set, and validation set.


## 8. Set up Learning Rate Scheduler:
- Sets up a linear learning rate scheduler.


## 9. Define Accumulation Steps:
- Defines the number of accumulation steps for gradient accumulation during training.


## 10. Train the Model:
- Defines a function (train_model) to train the DistilBERT model using the DataLoader for the training set.
- Prints training loss information during each epoch.


## 11. Evaluate the Model:
- Defines a function (evaluate_model) to evaluate the trained model on the testing set.
- Calculates and prints accuracy, precision, and recall scores.


## 12. Save Model and Tokenizer:
- Saves the trained model and tokenizer in the specified directory.


## 13. Load Model and Tokenizer from Saved Directory:
- Loads the saved model and tokenizer from the directory.


## 14. Function to Predict User Input:
- Defines a function (predict_user_input) to predict toxic comment labels for user input.
- Uses the loaded model and tokenizer.


## 15. User Input Prediction:
- Takes user input and calls the function to predict toxic comment labels.
- Prints the predicted labels.


### Things to improve:
1. I wish the training would be faster and more optimized.

