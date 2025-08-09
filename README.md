# LLM-Classification-Finetuning
This repository contains a starter notebook for fine-tuning a Large Language Model (LLM) for a classification task. The notebook leverages the KerasNLP and Keras libraries to predict which LLM response a human judge will prefer in a head-to-head battle.

The approach outlined is a "Shared Weight" strategy, similar to how Multiple Choice Question (MCQ) models are trained. We use the DebertaV3 model and implement mixed precision for faster training and inference.

1.  **Setup and Configuration:** Imports necessary libraries and defines a configuration class (`CFG`) for hyperparameters.
2.  **Data Loading and Preprocessing:** Loads the training and test datasets, performs basic preprocessing like extracting the first prompt/response from lists and converting winner columns to a single class label. It also creates combined 'prompt_response_a' and 'prompt_response_b' columns.
3.  **Data Splitting and Dataloaders:** Splits the training data into training and validation sets and defines a function to build TensorFlow datasets.
4.  **Model Definition:** Builds a classifier model using a DebertaV3 backbone and a custom classification head to process the prompt-response pairs.
5.  **Model Compilation:** Compiles the model with an Adam optimizer, categorical crossentropy loss, and categorical accuracy metric.
6.  **Training (Attempted):** Attempts to train the model using the `fit` method, but encounters a `ValueError: Invalid dtype: object` error. This indicates an issue with the data format being passed to the model training function. The learning rate scheduler and model checkpoint callbacks are also defined here.
