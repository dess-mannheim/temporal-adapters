import os
from itertools import product
import json

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from tqdm import tqdm

from extract_answers import extract_answers

# Prepare the LLM and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)
lora_config = LoraConfig()
model = get_peft_model(model, lora_config)
model = model.to(torch.device('cuda'))
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Get the survey dates for evaluation
dates = pd.to_datetime(pd.read_csv('../data/yougov_weeks/index.csv', names=['date'])['date'], format='%Y-%m-%d')

# Question from YouGov's "Britain's Mood, Measured Weekly"
# see https://yougov.co.uk/topics/politics/trackers/britains-mood-measured-weekly
question = 'Broadly speaking, which of the following best describe your mood and/or how you have felt in the past week?'

# Answer options
emotions = ['Happy', 'Sad', 'Energetic', 'Apathetic', 'Inspired', 'Frustrated', 'Optimistic', 'Stressed', 'Content', 'Bored', 'Lonely', 'Scared']

results = pd.DataFrame()

seeds = [99, 13, 42] # random seeds used in training

with tqdm(product(seeds, dates), total=len(seeds)*len(dates)) as dates_pbar:
    dates_pbar.set_description('Evaluating')

    for seed, survey_date in dates_pbar:

        # Set the directory where checkpoints are stored, depending on survey date and seed
        training_directory = f'./data/llama3_empirical_lr5-6/lora_{survey_date:%Y_%m_%d}_seed{seed}'
        if not os.path.isdir(training_directory): # skip folders that don't exist
            continue
        
         # Get all the checkpoints saved for this Temporal Adapter, including the final checkpoint
        checknums = [int(folder.split('-')[1]) for folder in os.listdir(training_directory) if folder.split('-')[1]!='final']
        checkpoints = ['checkpoint-' + str(num) for num in sorted(checknums)] + ['checkpoint-final']

        for checkpoint in checkpoints:

            if checkpoint == 'checkpoint-final':
                epoch = 1.0
                steps = np.NaN
                step_epoch = 'ep1_sNaN'
            else:
                f = open(f'{training_directory}/{checkpoint}/trainer_state.json')
                trainer_state = json.load(f)
                epoch = round(trainer_state['epoch'], ndigits=2)
                steps = trainer_state['global_step']
                step_epoch = f'ep{epoch}_s{steps}'
            
            # SHORT EVALUATION – skip all checkpoints except for step 300
            #if steps != 300: continue
            
            # Progress bar
            dates_pbar.set_postfix({'adapter': f'{survey_date:%Y_%m_%d}', 'epoch': epoch, 'seed': seed})

            # Load the current adapter
            model.load_adapter(f'{training_directory}/{checkpoint}', adapter_name=f'{survey_date:%Y_%m_%d}')
            model.set_adapter(f'{survey_date:%Y_%m_%d}')

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
                    _df = extract_answers(model, tokenizer, _question, answers=_emotions, temps=[0.25,0.5,1,2,4], variants=[1,2,3])
                    _df['epoch'] = epoch
                    _df['steps'] = steps
                    _df['step_epoch'] = step_epoch # easier post-processing of heterogenous step counts
                    _df['date'] = survey_date
                    _df['seed'] = seed
                    _df['question_type'] = question_type
                    _df['emotion_type'] = emotion_type

                    results = pd.concat([results, _df]).reset_index(drop=True)

            # Reset the currently active Temporal Adapter
            model.set_adapter('default')
            model.delete_adapter(f'{survey_date:%Y_%m_%d}')

results['data_source'] = 'Llama 3 8B'
results.to_csv('./data/llama3_empirical/llama3_empirical_lr5-6_seeds.csv', index=False)