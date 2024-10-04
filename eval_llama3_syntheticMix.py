import os
import json
from itertools import product

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from tqdm.auto import tqdm

from extract_answers import extract_answers


# Prepare the LLM and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
lora_config = LoraConfig()
model = get_peft_model(model, lora_config)
model = model.to(torch.device('cuda'))
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Training parameters
splits = np.arange(0, 1.05, 0.1) # encodes the percentage of 'joy' examples to be selected
seeds = [99, 13, 53, 8, 78, 14, 59, 38, 64, 37] # random seeds used in training

# Question from YouGov's "Britain's Mood, Measured Weekly"
# see https://yougov.co.uk/topics/politics/trackers/britains-mood-measured-weekly
question = 'Broadly speaking, which of the following best describe your mood and/or how you have felt in the past week?'
emotions = ['Happy', 'Sad'] # answer options

results = pd.DataFrame()

with tqdm(product(seeds, splits), total=len(seeds)*len(splits)) as splits_pbar:
    splits_pbar.set_description('Evaluating')

    for seed, split in splits_pbar:

        # Set the directory where checkpoints are stored, depending on data split and seed
        training_directory = f'./data/llama3_synth/lora_split{split:.2f}_seed{seed}'
        assert os.path.isdir(training_directory)
        
        # Get all the checkpoints saved for this Temporal Adapter, including the final checkpoint
        checknums = [int(folder.split('-')[1]) for folder in os.listdir(training_directory) if folder.split('-')[1]!='final']
        checkpoints = ['checkpoint-' + str(num) for num in sorted(checknums)] + ['checkpoint-final']

        # Evaluate each checkpoint separately
        for checkpoint in checkpoints:
            if checkpoint == 'checkpoint-final':
                epoch = 30 # TODO: add trainer_state.json to the final checkpoint
                steps = np.NaN
                step_epoch = 'ep30_sNaN'
            else:
                f = open(f'{training_directory}/{checkpoint}/trainer_state.json')
                trainer_state = json.load(f)
                epoch = round(trainer_state['epoch'], ndigits=2)
                steps = trainer_state['global_step']
                step_epoch = f'ep{epoch}_s{steps}'
            
            # Progress bar
            splits_pbar.set_postfix({'adapter': f'{split:.2f}', 'epoch': epoch, 'seed': seed})

            # Load the current adapter
            model.load_adapter(f'{training_directory}/{checkpoint}', adapter_name=f'{int(split*100)}')
            model.set_adapter(f'{int(split*100)}')

            # Test both lowercase and uppercase first characters for the answer option
            for emotion_type in ['uppercase', 'lowercase']:

                if emotion_type == 'lowercase':
                    _emotions = [emotion.lower() for emotion in emotions]
                else: _emotions = emotions

                # Optionally add an answer prefix between question and answer options                
                for question_type in ['question only', 'answer prefix']:

                    if question_type == 'answer prefix':
                        _question = question + ' I felt'
                    else: _question = question

                    # Extract answers from the selected model with the currently active Temporal Adapter
                    _df = extract_answers(model, tokenizer, _question, answers=_emotions, temps=[0.25,0.5,1,2,4], variants=[1,2])
                    _df['epoch'] = epoch
                    _df['steps'] = steps
                    _df['step_epoch'] = step_epoch # easier post-processing of heterogenous step counts
                    _df['split'] = split
                    _df['seed'] = seed
                    _df['question_type'] = question_type
                    _df['emotion_type'] = emotion_type

                    results = pd.concat([results, _df], ignore_index=True)

            # Reset the currently active Temporal Adapter
            model.set_adapter('default')
            model.delete_adapter(f'{int(split*100)}')

results['data_source'] = 'Llama 3 8B'
results.to_csv('./data/llama3_synth/llama3_synth_lr5-6_seeds.csv', index=False)