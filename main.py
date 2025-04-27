import torch
import re
import json
from decoding import *
import warnings
from utils import *
from tqdm.auto import tqdm
from model_config import *
from train import train_soft_prompt
from evaluate import evaluating_decoding_methods
warnings.simplefilter("ignore")

if __name__ == "__main__":
    # Parse command-line arguments
    args = get_args()

    # Run either demo or full evaluation based on user choice
    if args.choice == 'evaluating':
        evaluating_decoding_methods(args)

    elif args.choice == 'train_prompt':
            # Initialize with PEFT configuration
            generator, cot_decoding, beam_search_decoding, tokenizer = configurate_method(args)
            
            # Load training data
            train_datasets = load_datasets(args.dataset)
            
            # Train for each dataset
            for dataset_name in train_datasets:
                print(f'\nTraining on dataset: {dataset_name}')
                train_soft_prompt(
                    generator=generator,
                    cot_decoding=cot_decoding,
                    dataset=train_datasets[dataset_name],
                    args=args
                )
