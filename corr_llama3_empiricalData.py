from itertools import product
import multiprocessing

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import pearsonr, PermutationMethod

# Load time series data extracted from Llama 3 AFTER normalization
llama_df = pd.read_csv('./data/llama3_empirical/llama3_empirical_lr5-6_seeds_norm.csv', index_col=None)
# Load time series data from YouGov AFTER normalization
yougov_df = pd.read_csv('./data/yougov/yougov_norm.csv', index_col=None)

# Get all the possible answer options, temps, question types,… included in the time series data extracted from Llama 3
emotions = list(llama_df.emotion.str.capitalize().unique())
temps = list(llama_df.temp.unique()) #[0.25,0.5,1,2,4]
question_types = list(llama_df.question_type.unique()) #['question only', 'answer prefix']
emotion_types = list(llama_df.emotion_type.unique()) #['lowercase', 'uppercase']
variants = list(llama_df.variant.unique()) #[1,2,3]
seeds = list(llama_df.seed.unique())
steps = list(llama_df.steps.unique()) + ['epoch1'] # add steps to starmap for better progress visualization


# Calculate the cross-correlation between the Temporal Adapter estimate and YouGov survey data for a given set of extraction parameters
def calc_correlation(emotion, temp, question_type, emotion_type, variant, seed, steps):
    
    yougov_selected = yougov_df[yougov_df.emotion.str.lower() == emotion.lower()]
    
    if emotion_type == 'lowercase': _emotion = emotion.lower()
    else: _emotion = emotion

    # Select the relevant dataset
    llama_selected = llama_df[(llama_df.emotion == _emotion)
                            & (llama_df.temp == temp)
                            & (llama_df.question_type == question_type)
                            & (llama_df.emotion_type == emotion_type)
                            & (llama_df.variant == variant)
                            & (llama_df.seed == seed)]
    
    if steps == 'epoch1':
        llama_selected = llama_selected[(llama_selected.epoch == 1) & (llama_selected.steps.isna())]
        steps = np.NaN # for logging
        epoch = 1
    else:
        llama_selected = llama_selected[llama_selected.steps == steps]
        epoch = np.NaN # for logging


    _results = pd.DataFrame()
    # Make sure we have selected something and that there is more than 2 values
    if len(llama_selected) > 0 and len(llama_selected[llama_selected['probability'].isna() == False]) > 2:

        # Combine estimated data and survey data into a single Dataframe
        combined_df = pd.concat([llama_selected, yougov_selected])[['probability', 'date', 'data_source']]
        corr_df = combined_df.pivot(index='date', columns='data_source', values='probability').dropna()
        
        # We need at least two OVERLAPPING dates/weeks to correlate
        if len(corr_df) > 2:

            # Calculate the Pearson cross-correlation and perform a permutation test with 10000 permuations
            pearson_res = pearsonr(corr_df['Llama 3 8B'], corr_df['YouGov'], method=PermutationMethod(n_resamples=10000, random_state=42))

            _results = pd.DataFrame([{'emotion': emotion,
                                    'steps': steps,
                                    'epoch': epoch,
                                    'seed': seed,
                                    'temp': temp, 'variant': variant,
                                    'question_type': question_type, 'emotion_type': emotion_type,
                                    'correlated_with': 'YouGov', 'correlation': pearson_res.statistic, 'pvalue': pearson_res.pvalue},
                                    ])
    return _results


# A custom function to simulate starmap with an iterable
def my_starmap(args):
    return calc_correlation(*args)


# All the possible combinations to test
conditions = list(product(emotions, temps, question_types, emotion_types, variants, seeds, steps))

# Using multi-threading, evaluate all the correlations
with multiprocessing.Pool() as pool:
    pbar = tqdm(pool.imap(my_starmap, conditions), total=len(conditions))
    results = list(pbar) # list() invokes the evaluation
    
corr_results = pd.concat(results, ignore_index=True)

corr_results.to_csv('./data/llama3_empirical/llama3_empirical_lr5-6_seeds_corr_pval.csv', index=False)