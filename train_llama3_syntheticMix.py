import os

import regex as re
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import wandb
from accelerate import Accelerator

# Preprocess tweets -> tokens
def preprocess(examples):
    texts = [re.sub(r'(@|https?)\S+|#', '', text) for text in examples['text']]
    result = tokenizer(texts)  # also remove mentions, urls, and hashtags?
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

chunk_size = 512

# Group tokens into batches
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    if total_length >= chunk_size:
        total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_alpha=128, # default: 8
    r=128, # default: 8
    lora_dropout=0, # default: 0
    )

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Get the TweetEval dataset with labeled emotions
# columns : text, label
# labels  : 0 – anger (2118), 1 – joy (1163), 2 – optimisim (445), 3 – sadness (1326)
shards = {'train': 'emotion/train-00000-of-00001.parquet',
          'test': 'emotion/test-00000-of-00001.parquet',
          'validation': 'emotion/validation-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/cardiffnlp/tweet_eval/" + shards["train"])
df_test = pd.read_parquet("hf://datasets/cardiffnlp/tweet_eval/" + shards["test"])
df_val = pd.read_parquet("hf://datasets/cardiffnlp/tweet_eval/" + shards["validation"])
tweets_df = pd.concat([df_train, df_test, df_val]).reset_index(drop=True)

# General training parameters
epochs = 30
seeds = [99, 13, 53, 8, 78, 14, 59, 38, 64, 37]
splits = np.arange(0, 1.05, 0.1) # encodes the percentage of 'joy' examples to be selected

for seed in seeds:

    # Train with multiple, reproducible seeds
    set_seed(seed)

    for split in splits: # repeat the splits to train one epoch at a time

        # Prepare the data
        # there is at most 1163 joyfull tweets, so we cap the total number to 1163
        if split > 0:
            joy_df = tweets_df[tweets_df.label == 1].sample(int(1163*split), random_state=seed)
        else: joy_df = pd.DataFrame()
        if split < 1:
            sad_df = tweets_df[tweets_df.label == 3].sample(int(1163*(1-split)), random_state=seed)
        else: sad_df = pd.DataFrame()
        ds = Dataset.from_pandas(pd.concat([joy_df, sad_df]).drop(columns='label').reset_index(drop=True))
        
        tokenized_tweets = ds.map(preprocess, batched=True, remove_columns=['text'], num_proc=1)
        lm_tweets = tokenized_tweets.map(group_texts, batched=True, num_proc=1)
        lm_tweets.save_to_disk(f'./data/chunked_tweets/chunked_tweets_llama3_synth_split{split:.2f}_seed{seed}')

        # Set the directory where checkpoints will be stored
        training_directory = f'./data/llama3_synth/lora_split{split:.2f}_seed{seed}'

        # Create model with LoRA adapter
        # all PEFT models must have at least one adapter
        adapter_name = f'{split:.2f}_{seed}'
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B") # re-load the model
        # LoRA needs to be called "default" because of a PEFT bug, see https://github.com/huggingface/transformers/issues/24252
        model = get_peft_model(base_model, lora_config)

        os.environ["WANDB_PROJECT"] = "llama3_synth" # requires login via CLI first
        os.environ["WANDB_RESUME"] = "allow" # allow training to resume after it previously failed
        os.environ["WANDB_LOG_MODEL"] = "false" # do not upload artifacts

        # Re-init to change run name and ID
        if accelerator.is_main_process:
            wandb.finish(quiet=True)
            wandb.init(name=f"synth_split{split:.2f}_seed{seed}", id=f"lora_split{split:.2f}_seed{seed}",
                    dir='./data/wandb',
                    reinit=True)

        train_args = TrainingArguments(
            output_dir=training_directory,
            logging_dir=f'{training_directory}/logs',
            seed=seed,

            # training
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=5e-6,
            num_train_epochs=epochs,
            #weight_decay=0.9,
            gradient_accumulation_steps=4,

            # training optimizations
            bf16=True, # bfloat16 training
            tf32=True,
            optim="adamw_torch_fused", # improved optimizer

            # evaluation
            evaluation_strategy="no",
            save_strategy="epoch", # dataset is quite small, so we save every epoch
            push_to_hub=False,
            report_to="wandb",
            logging_steps=5
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=lm_tweets,
            #eval_dataset=split_dataset['test'], # we don't evaluate test performance because our overall goal is not to produce text
            data_collator=data_collator,
        )

        trainer = accelerator.prepare(trainer)

        if accelerator.is_main_process:
            print(f'\n--- now training {adapter_name} ---\n')

        trainer.train()

        model.save_pretrained(f'{training_directory}/checkpoint-final', selected_adapters=['default'])