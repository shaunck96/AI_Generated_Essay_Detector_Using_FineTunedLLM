AI-Generated Text Detection System
Overview
The AI-Generated Text Detection System is a machine learning solution aimed at distinguishing AI-generated essays from those written by humans. It leverages advanced models like LLAMA and BERT, trained on diverse datasets. The primary objective is to help educational institutions combat cheating by accurately classifying essays. The system is built to address the increasing challenge of identifying AI's role in text generation.

Features
Advanced Language Models: Uses LLAMA and BERT for deep text analysis.
Parameter-Efficient Fine-Tuning (PEFT): Adapts large models efficiently to specific tasks.
Stratified K-Fold Cross-Validation: Guarantees reliable model evaluation and generalization.
Parallel Text Preprocessing: Enhances preprocessing efficiency by handling multiple texts simultaneously.
Automated Typo Correction: Utilizes language_tool_python to enhance text quality.
GPU Optimization: Optimizes for GPU usage, including 8-bit quantization, for performance.
Debug Mode: Simplifies system testing and debugging.
Comprehensive Logging: Monitors and logs system performance and issues.
Scalability: Designed to manage large datasets and extensive models efficiently.

System Requirements

**Python 3.8+
Libraries: Pandas, NumPy, PyTorch, Transformers, PEFT, etc. (Refer to requirements.txt)
GPU environment (CUDA support recommended) for model training and inference**

Installation
Clone the repository:

**git clone (https://github.com/shaunck96/AI_Generated_Essay_Detector_Using_FineTunedLLM.git)**

Install dependencies:

**pip install -r requirements.txt**

Usage
Data Preparation
Prepare a CSV dataset with essays labeled as 'AI-generated' (label = 1) or 'human-written' (label = 0).
Columns should include text for the essay content.
Training the Model
Run the training script:

**python train_model.py**

The script will preprocess data, load models, conduct Stratified K-Fold Cross-Validation, and save the trained models.

Contributing
Contributions are welcome. Please follow standard coding practices and test new features or fixes thoroughly.
