# Kaggle_Comptetiton
Train an AI system to effectively detect AI generated essays from human written essays to help educational institutions combat cheating during submissions


AI-Generated Text Detection System
Overview
This project implements a machine learning system designed to differentiate between human-written and AI-generated essays. Utilizing advanced models such as LLAMA and BERT, the system trains on a diverse dataset, leveraging techniques like Parameter-Efficient Fine-Tuning (PEFT) for efficient adaptation to the task. The primary goal is to accurately classify essays into two categories: AI-generated and human-written, addressing the growing need for discerning AI influence in text content.

Features
Advanced Language Models: Utilizes LLAMA and BERT models for robust text understanding.
Parameter-Efficient Fine-Tuning: Adapts large language models efficiently to the specific task without extensive retraining.
Stratified K-Fold Cross-Validation: Ensures reliable evaluation and generalization of the model.
Parallel Text Preprocessing: Speeds up the preprocessing stage by handling multiple texts concurrently.
Automated Typo Correction: Improves text quality using language_tool_python.
GPU Optimization: Configures models for efficient GPU usage, including 8-bit quantization.
Debug Mode: Facilitates quick testing and debugging of the system.
Comprehensive Logging: Tracks and reports the system's performance and issues.
Scalability: Structured to handle large datasets and extensive models.
System Requirements
Python 3.8 or later
Pandas, NumPy, PyTorch, Transformers, PEFT, and other dependencies (see requirements.txt for a complete list)
GPU environment (preferably with CUDA support) for efficient model training and inference
Installation
Clone the repository to your local machine or cloud environment.
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Ensure access to appropriate hardware (GPUs) for training and inference.
Usage
Data Preparation
Load your dataset of essays, which should be labeled as 'AI-generated' or 'human-written'. The dataset should be in CSV format and contain at least two columns: text (the essay) and label (0 for human-written, 1 for AI-generated).

Training the Model
Run the training script with the following command:

**python train_model.py**
This script will perform the following steps:

Preprocess the text data, including typo correction.
Load and configure the LLAMA/BERT models.
Train the models using Stratified K-Fold Cross-Validation.
Save the trained models for later inference.

License
This project is open source and available under the MIT License.
