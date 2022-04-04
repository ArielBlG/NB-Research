import argparse

import numpy as np
import random
import torch
import os
import psutil
import humanize
import GPUtil as GPU

from transformers import (BertTokenizerFast,
                          BertConfig,
                          BertForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments)

from datasets import load_dataset

emoji_list = ['#question', '#lightbulb-moment', '#real-world-application', '#learning-goal',
              '#important', '#i-think', '#lets-discuss', '#lost', '#just-curious', '#surprised', '#interesting-topic']

dir_path = '/home/arielblo/Datasets/full_datasets/'
parser = argparse.ArgumentParser()
parser.add_argument("--epochs")
parser.add_argument("--seed")
parser.add_argument("--emojis")
parser.add_argument("--block_size")
args = parser.parse_args()
epochs = float(args.epochs)
random_seed = int(args.seed) if args.seed else 42
emojis_flag = True if args.emojis else False
block_size = int(args.block_size) if args.block_size else 200
print("epochs: " + str(epochs))
print("random seed: " + str(random_seed))
print("block size: " + str(block_size))
print("emojis flag: " + str(emojis_flag))

device = torch.device("cuda")
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

GPU.showUtilization()
GPUs = GPU.getGPUs()
print(GPUs[0])

GPUs = GPU.getGPUs()
gpu = GPUs[0]
process = psutil.Process(os.getpid())
print("***********************")
print("Memory Stats - " + "Program Startup")
print("***********************")
print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
      " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree,
                                                                                                gpu.memoryUsed,
                                                                                                gpu.memoryUtil * 100,
                                                                                                gpu.memoryTotal))
print("***********************")

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
if emojis_flag:
    _tokenizer.add_tokens(emoji_list)
    num_tokens = len(emoji_list)
    model.resize_token_embeddings(len(_tokenizer))
    print(f"Tokenizer after re-sizing: {len(_tokenizer)}")

model.to(device)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_function(examples):
    return _tokenizer(examples["text"])


print(dir_path)
data_files = {'train': f"{dir_path}full_sent_train.csv", 'validation': f"{dir_path}full_sent_test.csv"}
dataset = load_dataset('csv', data_files=data_files)

tokenized_datasets = dataset.map(tokenize_function, batched=False, num_proc=1,
                                 remove_columns=["text", 'Unnamed: 0'])
print(tokenized_datasets)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)
print(lm_datasets)
tokenized_datasets = lm_datasets
training_args = TrainingArguments(
    output_dir=dir_path + 'models',
    overwrite_output_dir=True,
    num_train_epochs=epochs, # for debugging=0.000
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=_tokenizer, mlm=True, mlm_probability=0.15
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],

)
log = trainer.train()
print(log)
trainer.save_model('/home/arielblo/Models/BetterModel')
print("Done!")