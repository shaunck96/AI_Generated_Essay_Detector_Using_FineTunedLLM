!pip install -q peft --no-index --find-links /kaggle/input/llm-detect-pip/peft-0.5.0-py3-none-any.whl
!!pip install -q language-tool-python --no-index --find-links /kaggle/input/daigt-misc/language_tool_python-2.7.1-py3-none-any.whl
!!mkdir -p /root/.cache/language_tool_python/
!!cp -r /content/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7
!pip install -i https://test.pypi.org/simple/ bitsandbytes

from __future__ import annotations
import time, sys, gc, logging, random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import transformers
import peft
from accelerate import Accelerator
import bitsandbytes
from sklearn.metrics import accuracy_score, roc_auc_score
from shutil import rmtree
import language_tool_python
import optuna
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from transformers import BertForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import concurrent
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from scipy.special import expit as sigmoid

print(transformers.__version__)
print(peft.__version__)
print(torch.__version__)

language_tool = language_tool_python.LanguageTool('en-US')
N_FOLD = 5
SEED = 42
DEBUG = True
IS_TRAIN = False

# Seed the same seed to all
def seed_everything(seed=42):
    """
    Seeds the random number generators of Python's `random`, NumPy, and PyTorch to ensure reproducibility.

    Parameters:
    - seed (int): A seed value to be used for all random number generators.

    Returns:
    - None: This function does not return anything but sets the seed for various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything()
# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

log_level = "DEBUG"

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING
)

# set the main code and the modules it uses to the same log-level according to the node
transformers.utils.logging.set_verbosity(log_level)

# Cross validation
def cv_split(train_data):
    """
    Performs stratified K-fold cross-validation splitting on the training dataset.

    Parameters:
    - train_data (DataFrame): A pandas DataFrame containing the training data.

    Returns:
    - DataFrame: The input DataFrame with an additional column 'fold' indicating the fold assignment for each row.
    """
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    X = train_data.loc[:, train_data.columns != "label"]
    y = train_data.loc[:, train_data.columns == "label"]

    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train_data.loc[valid_index, "fold"] = fold

    print(train_data.groupby("fold")["label"].value_counts())
    display(train_data.head())
    return train_data

def pre_processing_text(text):
    """
    Processes and corrects typos in the given text using `language_tool_python`.

    Parameters:
    - text (str): The text string to be processed.

    Returns:
    - str: The processed text with corrections applied.
    """
    text = text.replace('\n', ' ')
    typos = language_tool.check(text) # typo is a list
    # Check how many typos
    #if len(typos) > 0:
    #print(f"The number of typos = {len(typos)}\n {typos}")
    text = language_tool.correct(text)
    return text

# Run pre-processing texts in parallel
def parallel_pre_processing_text(texts):
    """
    Processes a list of texts in parallel, applying typo correction to each text.

    Parameters:
    - texts (list of str): A list of text strings to be processed.

    Returns:
    - list of str: A list of processed texts with corrections applied.
    """
    print(f"Total number of texts {len(texts)}")
    results = []
    # run 'pre_processing' fucntions in the process pool
    with ThreadPoolExecutor(4) as executor:
        # results = list(tqdm(executor.map(pre_processing_text, texts)))
        # send in the tasks
        futures = [executor.submit(pre_processing_text, text) for text in texts]
        # wait for all tasks to complete
        for future in futures:
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"Finished {len(results)} / {len(texts)}\n", end='', flush=True)
    # wait for all tasks to complete
    print("results", len(results))
    return results


def load_train_data():
    """
    Loads and preprocesses the training data from specified CSV files.

    Returns:
    - DataFrame: A pandas DataFrame containing the combined and processed training data.
    """
    train_df = pd.read_csv("/content/ai_generated_train_essays_gpt-4.csv")
    train_prompts_df = pd.read_csv("/content/train_prompts.csv", sep=',')

    # rename column generated to label and remove used 'id' and 'prompt_id' columns
    # Label: 1 indicates generated texts (by LLMs)
    train_df = train_df.rename(columns={'generated': 'label'})
    train_df = train_df.reset_index(drop=True)
    train_df = train_df.drop(['id', 'prompt_id'], axis=1)
#     print("Start processing training data's text")
#     start = time.time()
#     # Clear text in both train and test dataset
#     train_df['text'] = train_df['text'].progress_apply(lambda text: pre_processing_text(text))
#     display(train_df.head())
#     print(f"Correct the training data's texts with {time.time() - start : .1f} seconds")

    # Include external data
    external_df = pd.read_csv("/content/train_v2_drcat_02.csv", sep=',')
    # We only need 'text' and 'label' columns
    external_df = external_df[["text", "label"]]
    external_df["label"] = 1

    xls_file_path = '/content/training_set_rel3.xls'
    external_df_two = pd.read_excel(xls_file_path)
    external_df_two = external_df_two[['essay']]
    external_df_two.rename(columns={'essay':'text'},inplace=True)
    external_df_two["label"] = 0

    external_df = pd.concat([external_df,external_df_two], axis=0)
    print("Start processing external data's texts")
    start = time.time()
    external_df['text'] = parallel_pre_processing_text(external_df['text'].to_list())
    print(f"Correct the external data's texts with {time.time() - start : .1f} seconds")
    #external_df['text'] = external_df['text'].map(lambda text: pre_processing_text(text))
    display(external_df.head())
    external_df.to_csv('train_v2_drcat_02_fixed.csv', index=False)
    # Merge train and external data into train_data
    train_data = pd.concat([train_df, external_df, external_df_two])
    train_data.reset_index(inplace=True, drop=True)
    # print(f"Train data has shape: {train_data.shape}")
    print(f"Train data {train_data.value_counts('label')}") # 1: generated texts 0: human texts
    return train_data

# Load the pretrained model and add an extra layer with PEFT library for fine-tuning
def load_model_mistral(fold):
    """
    Loads the LLAMA model for a specific fold with the PEFT (Parameter-Efficient Fine-Tuning) configuration.

    Parameters:
    - fold (int): The fold number for which the model is to be loaded.

    Returns:
    - tuple: A tuple containing the loaded model and tokenizer.
    """
    TARGET_MODEL = "openlm-research/open_llama_3b"
    # TARGET_MODEL = "/kaggle/input/mistral-7b-v0-1/Mistral-7B-v0.1"
    # LoRA: Low-Rank Adaptation of Large Language Models
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
    )
    # Enable GPU to run the model with 4bit
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_compute_dtype=torch.bfloat16
    #)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        load_in_8bit_fp32_cpu_offload=True,  # Enable CPU offloading for certain layers
        bnb_8bit_quant_type="nf8",  # Type of 8-bit quantization, 'nf8' is one of the options
        bnb_8bit_use_double_quant=True,  # Use double quantization
        bnb_8bit_compute_dtype=torch.bfloat16,  # Data type for computation in 8-bit mode
        bnb_8bit_blocksparse_layout=None,  # Block-sparse layout, use None for dense models
        bnb_8bit_custom_kernel=False,  # Use custom kernel, false by default
        bnb_8bit_cpu_offload=True,  # Enable CPU offloading
        bnb_8bit_cpu_offload_dtype=torch.float32,  # Data type for CPU offloaded tensors
        bnb_8bit_cpu_offload_use_pin_memory=True,  # Use pinned memory for CPU offloading
        bnb_8bit_cpu_offload_use_fast_fp32_to_fp16_conversion=False  # Use fast conversion from FP32 to FP16
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    # Load the model
    base_model = LlamaForSequenceClassification.from_pretrained(TARGET_MODEL,
                                                                num_labels=2, # label is 0 or 1
                                                                quantization_config=bnb_config,
                                                                device_map="auto")
    base_model.config.pretraining_tp = 1 # 1 is 7b
    base_model.config.pad_token_id = tokenizer.pad_token_id

    if IS_TRAIN:
        # Parameter-Efficient Fine-Tuning (PEFT) methods enable efficient adaptation of
        # pre-trained language models (PLMs) to various downstream applications
        # without fine-tuning all the model's parameters.
        # https://github.com/huggingface/peft
        model = get_peft_model(base_model, peft_config)
    else:
        OUTPUT_DIR = f"/content/mistral-7b-v0-for-llm-detecting-competition/mistral_7b_fold{fold}"
        # OUTPUT_DIR = f"/kaggle/working/mistral_7b_fold{fold}"
        # Load the pretrained model with PEFT
        model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))

    model.print_trainable_parameters() # Display the trainable parameters

    return model, tokenizer

# Flag to indicate if the model is in training phase
#IS_TRAIN = True  # Set this to False if not training

def load_model_bert_cpu(fold):
    """
    Loads the BERT model for a specific fold.

    Parameters:
    - fold (int): The fold number for which the model is to be loaded.

    Returns:
    - tuple: A tuple containing the loaded BERT model and tokenizer.
    """
    TARGET_MODEL = "bert-base-uncased"
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=True)

    # Load the model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        TARGET_MODEL,
        num_labels=2  # Assuming binary classification
    )

    # Check if GPU is available and move the model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length=512):
    """
    Tokenizes and processes the text data using the provided tokenizer.

    Parameters:
    - examples (dict): A dictionary containing the text data.
    - tokenizer: The tokenizer to be used for processing.
    - max_length (int): The maximum length of the tokenized sequences.

    Returns:
    - dict: A dictionary containing the processed and tokenized text data.
    """
    examples["text"] = list(map(lambda text: pre_processing_text(text), examples["text"]))
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for the model predictions.

    Parameters:
    - eval_pred (tuple): A tuple containing model predictions and actual labels.

    Returns:
    - dict: A dictionary containing computed metrics like accuracy and ROC-AUC.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy_val = accuracy_score(labels, predictions)
    roc_auc_val = roc_auc_score(labels, predictions)
    r = { "accuracy": accuracy_val,
          "roc_auc": roc_auc_val}
    # logging.debug(f'{r}')
    return r


def train_model_by_fold(fold):
    """
    Trains the model on a specified fold of the dataset.

    Parameters:
    - fold (int): The fold number to train the model on.

    Returns:
    - None: This function does not return anything but trains the model on the specified fold.
    """
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Start training the fold {fold} model")
    # Create train and valid dataset for a fold
    fold_valid_df = train_data[train_data["fold"] == fold]
    fold_train_df = train_data[train_data["fold"] != fold]
    # Train the model with small (for debugging) or large samples
    if DEBUG:
        fold_train_df = fold_train_df.sample(frac =.05, random_state=SEED)
        fold_valid_df = fold_valid_df.sample(frac =.05, random_state=SEED)
    else:
        fold_train_df = fold_train_df.sample(frac =.3, random_state=SEED)
        fold_valid_df = fold_valid_df.sample(frac =.3, random_state=SEED)

    print(f'fold_train_df {fold_train_df.groupby("fold")["label"].value_counts()}')
    print(f'fold_valid_df {fold_valid_df.groupby("fold")["label"].value_counts()}')
    # create the dataset
    train_ds = Dataset.from_pandas(fold_train_df)
    valid_ds = Dataset.from_pandas(fold_valid_df)

    # Load the pretrained model and tokenizer
    #model, tokenizer = load_model_mistral(fold)
    model,tokenizer = load_model_bert_cpu(fold)

    # Tokenize the train and valid dataset and pass tokenizer as function argument
    train_tokenized_ds = train_ds.map(preprocess_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer})
    valid_tokenized_ds = valid_ds.map(preprocess_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer})
    # Create data collator with padding (padding to the longest sequence)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Start training processing
    TMP_DIR = Path(f"/content/tmp/mistral_7b_fold{fold}/")
    TMP_DIR.mkdir(exist_ok=True, parents=True)

    STEPS = 5 if DEBUG else 20
    EPOCHS = 1 if DEBUG else 10
    BATCH_SIZE = 2
    training_args = TrainingArguments(output_dir=TMP_DIR,
                                      learning_rate=5e-5,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=1,
                                      gradient_accumulation_steps=16,
                                      max_grad_norm=0.3,
                                      optim='paged_adamw_32bit',
                                      lr_scheduler_type="cosine",
                                      num_train_epochs=EPOCHS,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      load_best_model_at_end=True,
                                      push_to_hub=False,
                                      warmup_steps=STEPS,
                                      eval_steps=STEPS,
                                      logging_steps=STEPS,
                                      report_to='none', # if DEBUG else 'wandb'
                                      log_level='warning', # 'warning' is default level
                                     )


    # Create the trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_tokenized_ds,
                      eval_dataset=valid_tokenized_ds,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)

    trainer.train()

    OUTPUT_DIR = Path(f"/content/working/mistral_7b_fold{fold}/")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    # Save the fine-tuned model
    trainer.save_model(output_dir=str(OUTPUT_DIR))
    print(f"=== Finish the training for fold {fold} ===")
    del model, trainer, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def language_model_training_trigger(IS_TRAIN):
    """
    Triggers the training of the language model based on the IS_TRAIN flag.

    Parameters:
    - IS_TRAIN (bool): Flag indicating whether to train the model.

    Returns:
    - None: This function does not return anything but triggers the training process.
    """
  if IS_TRAIN:
    start = time.time()
    # Load train data
    train_data = load_train_data()
    # Cross validation with 5 fold
    train_data = cv_split(train_data)
    # Train the model
    fold = 0
    train_model_by_fold(0)
    #     # Add multiple threads to run each fold model concurrently

    #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    #     futures = [executor.submit(train_model_by_fold, fold) for fold in range(2)]
    #     # wait for all tasks to complete
    #    wait(futures)
    #    print('All training tasks are done!')

    #for idx, fold in enumerate(range(N_FOLD)):
    sys.exit(f"Training time of fold {fold} = {time.time() - start: .1f} seconds")

def clear_memory():
    """
    Clears GPU memory cache and runs garbage collection to free memory.

    Returns:
    - None: This function does not return anything but performs memory cleanup.
    """
    torch.cuda.empty_cache()
    gc.collect()

# Sigmoid activation function can map 'x' between 0 and 1
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def predict_result_by_fold(fold):
    """
    Generates predictions using the model trained on a specific fold.

    Parameters:
    - fold (int): The fold number of the model to use for prediction.

    Returns:
    - None: This function does not return anything but populates global predictions.
    """
    clear_memory()
    print(f"=== Start prediction with {fold} ===")
    #model, tokenizer = load_model_mistral(fold)
    model, tokenizer = load_model_bert_cpu(fold)

    # Load the test dataframe as dataset
    test_ds = Dataset.from_pandas(test_df)
    test_tokenized_ds = test_ds.map(preprocess_function, batched=True,
                                    fn_kwargs={"tokenizer": tokenizer})
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            padding="longest")
    # Create the trainer
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      data_collator=data_collator)
    pred_output = trainer.predict(test_tokenized_ds)
    logits = pred_output.predictions
    # Apply sigmoid to
    probs = sigmoid(logits[:, 1])
    print(f"fold = {fold} probs = {probs}")
    global predictions
    for i, prob in enumerate(probs):
        predictions[i].append(prob)
    # Clear memory
    del model, trainer, tokenizer, test_ds, test_tokenized_ds, data_collator
    clear_memory()

def predict_result():
    """
    Aggregates predictions from all model folds and computes the final result.

    Returns:
    - list: A list containing the aggregated predictions for each instance in the test dataset.
    """
    global predictions
    predictions = [[] for i in range(len(test_df))]
    start = time.time()
    print(f"=== Begin prediction  ===")

    #for fold in range(N_FOLD):
    #    predict_result_by_fold(fold)
    fold = 0
    predict_result_by_fold(fold)
    print(f"Finish prediction in {time.time() - start: .1f} seconds")

    return predictions

load_train_data()
# Check if we need to fine-tune the LLM model
IS_TRAIN = True

language_model_training_trigger(IS_TRAIN)

# Load test data
test_df = pd.read_csv("/content/test_essays.csv", sep=',')
test_df = test_df.rename(columns={'generated': 'label'})
test_df['text'] = test_df['text'].progress_apply(lambda text: pre_processing_text(text))
# print(f'test_df.shape: {test_df.shape}')
# test_df.head(3)

predictions = predict_result()
probs = [np.mean(pred) for pred in predictions]
print(probs)

sub = pd.DataFrame()
sub['id'] = test_df['id']
sub['generated'] = probs
sub.to_csv('/content/submission.csv', index=False)
sub.head()

