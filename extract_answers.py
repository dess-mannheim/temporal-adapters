import torch
import pandas as pd
import numpy as np

def extract_answers(model, tokenizer, question:str, answers:list, values:list=None, temps:list=[1], variants:list=[1],
                    instr_tag_before:str='', instr_tag_after:str='') -> pd.DataFrame:
    '''
    A method that extracts token probabilities for survey answers from a causal language model.

    Parameters
    ---

    - model: HF transformers causal language model
    - tokenizer: HF transformers tokenizer used for the selected LLM
    - question: stem of the question that will be part of the prompt, but the token probabilities of these "question stem" tokens are not taken into account
    - answers: list of answer options that are tokenized and then the token probabilities are gathered
    - values: optionally, a list of numerical values to encode the answer options (you probably don't need this)
    - temps: optionally, a list of temperatures to apply to the final softmax function during evaluation
    - variants: optionally, a list of answer scoring variants to apply during evaluation
    - instr_tag_before: optionally, a special token for instruction tuned models to add before the question/user input
    - instr_tag_after: optionally, a special token for instruction tuned models to add after the question/user input
    '''

    # Code all answer options as 1 if not otherwise specified
    if values is None:
        values = [1]*len(answers)
    
    # Backwards compatibility with older implementation
    if type(variants) == int:
        variants = [variants]
    
    # OPTIONALLY add an empty answer at the end of batch for comparison when drawing the softmax over the answer options
    #answers = answers + [tokenizer.pad_token]

    # Encode the concatenated questions + answers
    # add a pad_token at the beginning and concatenate question + answer
    # optionally add instruction tokens around the question
    # KEEP the space between question and answer for a clear separation
    input_texts = [f'{tokenizer.pad_token}{instr_tag_before}{question}{instr_tag_after} {answer}' for answer in answers]
    input_ids, attention_mask = tokenizer(input_texts, padding=True, return_tensors="pt").values()
    input_ids = input_ids.to(torch.device('cuda'))
    attention_mask = attention_mask.to(torch.device('cuda'))
    attention_mask[:,0] = 0  # mask the pad_token
    question_length = len(tokenizer(question)['input_ids']) # number of tokens in the question

    # Perform a single forward pass through the model
    outputs = model(input_ids, attention_mask=attention_mask)

    result_df = pd.DataFrame()
    for temp in temps:
        for variant in variants:

            ### VARIANT 1: calculate log_softmax over all tokens, then gather and sum up
            ### this variant requires normalization afterwards
            if variant == 1:
                probs = torch.log_softmax(outputs.logits/temp, dim=-1).detach() # softmax with temperature on ALL tokens
            else:
                probs = outputs.logits.detach()

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            # the token at index 0 is a pad_token
            _ids = input_ids[:, 1:]

            gen_probs = torch.gather(probs, 2, _ids[:, :, None]).squeeze(-1)

            ### VARIANT 2: calculate log_softmax over relevant tokens, then sum up answer probs
            ### this variant also requires normalization if 
            if variant == 2:
                gen_probs = torch.log_softmax(gen_probs/temp, dim=0)

            # Combine tokens, probabilities, the answer options they belong to and (optionally) how they are coded
            batch_df = pd.DataFrame()
            for input_sentence, input_probs, answer, value in zip(_ids, gen_probs, answers, values):
                    
                text_sequence = []
                for token, p in list(zip(input_sentence, input_probs))[question_length:]:
                    # remove padding at the end
                    if token not in tokenizer.all_special_ids:
                        text_sequence.append((tokenizer.decode(token), p.item()))
            
                answer_df = pd.DataFrame(text_sequence, columns=['token', 'prob'])
                answer_df['answer'] = answer
                answer_df['value'] = value
                batch_df = pd.concat([batch_df, answer_df], ignore_index=True)
            
            # Sum up log probs for all the tokens that make up an answer option
            batch_df = batch_df.drop(columns='token').groupby(['answer', 'value']).sum().reset_index()
            
            batch_df['temp'] = temp
            batch_df['variant'] = variant
            batch_df['answer'] = batch_df['answer'].replace('</s>', 'N/A') # deal with "nonresponse"
            result_df = pd.concat([result_df, batch_df], ignore_index=True)
    
    return result_df