import os

import regex as re
import pandas as pd

from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerControl, trainer_utils, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import wandb
from accelerate import Accelerator


# Callback for stopping the training after each epoch
# this allows us to train one epoch at a time, i.e. all weeks for 1 epoch first, than 2 epochs, etc.
class StopCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control:TrainerControl, logs=None, **kwargs):
        control.should_training_stop = True

# Preprocess tweets -> tokens
def preprocess(examples):
    try:
        texts = [re.sub(r'(@|https?)\S+|#', '', text) for text in examples['text']]
    except Exception as e:
        print(e)
        texts = ['']
    result = tokenizer(texts)  # also remove mentions, urls, and hashtags?
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

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


#torch._dynamo.config.cache_size_limit = 16

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

chunk_size = 512

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dates = pd.to_datetime(pd.read_csv('./data/yougov_weeks/index.csv', names=['date'])['date'], format='%Y-%m-%d')

# General training parameters
epochs = 1
current_epoch = 1
seeds = [42, 99, 13]

for seed in seeds:

    # set a global seed for reproducibility
    set_seed(seed)

    for survey_date in dates:

        # Load the Twitter data
        ds = Dataset.from_csv(f'./data/yougov_weeks/{survey_date:%Y-%m-%d}', header=None, names=['text'])

        # Prepare the training data
        tokenized_tweets = ds.map(preprocess, batched=True, remove_columns=['text'], num_proc=32)
        lm_tweets = tokenized_tweets.map(group_texts, batched=True, num_proc=32)
        lm_tweets.save_to_disk(f'./data/chunked_tweets/chunked_tweets_llama3_yougov-{survey_date:%Y-%m-%d}')

         # Set the directory where checkpoints will be stored
        training_directory = f'./data/llama3_empirical/llama3_empirical_lr5-6/lora_{survey_date:%Y_%m_%d}'
        
        # If the training_directory already exists, resume training for another epoch
        if os.path.isdir(training_directory):
            last_checkpoint = trainer_utils.get_last_checkpoint(training_directory)
        else:
            last_checkpoint = None

        # create model with LoRA adapter
        # all PEFT models must have at least one adapter
        adapter_name = f'{survey_date:%Y-%m-%d}_seed{seed}'
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B") # re-load the model
        # LoRA needs to be called "default" because of a PEFT bug, see https://github.com/huggingface/transformers/issues/24252
        model = get_peft_model(base_model, lora_config)

        os.environ["WANDB_PROJECT"] = "llama3_synth" # requires login via CLI first
        os.environ["WANDB_RESUME"] = "allow" # allow training to resume after it previously failed
        os.environ["WANDB_LOG_MODEL"] = "false" # do not upload artifacts

        # re-init to change run name and ID
        if accelerator.is_main_process:
            wandb.finish(quiet=True)
            wandb.init(name=f"{survey_date:%Y_%m_%d}_seed{seed}", id=f"lora_{survey_date:%Y_%m_%d}_seed{seed}",
                    dir='./data/wandb',
                    reinit=True)

        train_args = TrainingArguments(
            output_dir=training_directory,
            logging_dir=f'{training_directory}/logs',
            seed=seed,

            # training
            per_device_train_batch_size=6,
            per_device_eval_batch_size=6,
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
            save_strategy="steps", # save a checkpoint every 50 steps
            save_steps=50,
            push_to_hub=False,
            report_to="wandb",
            logging_steps=5
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=lm_tweets,
            #eval_dataset=split_dataset['test'],  # we don't evaluate test performance because our overall goal is not to produce text
            data_collator=data_collator,
            callbacks=[StopCallback()]  # train one epoch at a time
        )

        trainer = accelerator.prepare(trainer)

        if accelerator.is_main_process:
            print(f'\n--- now training {adapter_name}, epoch {current_epoch} ---\n')

        if last_checkpoint is not None:
            print('last checkpoint:', last_checkpoint)
            trainer.train(resume_from_checkpoint=True)
        else:
            try:
                trainer.train()
            except Exception as e:
                print(e)

        # save the final checkpoint after N epochs as well (not just every 50 steps)
        model.save_pretrained(f'{training_directory}/checkpoint-final', selected_adapters=['default'])