# LLM-Classification-Finetuning
This repository contains a starter notebook for fine-tuning a Large Language Model (LLM) for a classification task. The notebook leverages the KerasNLP and Keras libraries to predict which LLM response a human judge will prefer in a head-to-head battle.

The approach outlined is a "Shared Weight" strategy, similar to how Multiple Choice Question (MCQ) models are trained. We use the DebertaV3 model and implement mixed precision for faster training and inference.
