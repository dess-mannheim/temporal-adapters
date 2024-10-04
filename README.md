# Extracting Affect Aggregates from Longitudinal Social Media Data with Temporal Adapters for Large Language Models
Authors: Georg Ahnert, Max Pellert, David Garcia, and Markus Strohmaier

Also have a look at our [preprint on arXiv](https://arxiv.org/abs/2409.17990)!

### Abstract

We propose temporally aligned Large Language Models (LLMs) as a tool for longitudinal analysis of social media data. We fine-tune Temporal Adapters for Llama 3 8B on full timelines from a panel of British Twitter users, and extract longitudinal aggregates of emotions and attitudes with established questionnaires. We validate our estimates against representative British survey data and find strong positive, significant correlations for several collective emotions. The obtained estimates are robust across multiple training seeds and prompt formulations, and in line with collective emotions extracted using a traditional classification model trained on labeled data. To the best of our knowledge, this is the first work to extend the analysis of affect in LLMs to a longitudinal setting through Temporal Adapters. Our work enables new approaches towards the longitudinal analysis of social media data.

### Contents

#### Temporal Adapter Training

- **`train_llama3_empiricalData.py`** trains Temporal Adapters from weekly splits of Twitter data and can be run with `accelerate launch` for distributed training.
- **`train_llama3_syntheticMix.py`** trains Temporal Adapters from synthetically mixed (labeled) tweets and can be run with `accelerate launch` for distributed training.