import os
import argparse
import json
import datasets
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict, Optional, Union, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
import matplotlib.pyplot as plt
from decoding import *
import re
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # Global configurations
    parser.add_argument('--choice', default='demo', choices=['demo', 'evaluating', 'train_prompt'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=int)
    parser.add_argument('--model', default='microsoft/phi-2', type=str)
    parser.add_argument('--assistant_model', default='/ix/lshangguan/Zhiwei/Llama-3.2-1B-Instruct', type=str)
    parser.add_argument('--beam_search', action='store_true')
    parser.add_argument('--cot_deocding', action='store_true')
    
    # [DEMO]
    parser.add_argument('--demo_prompt', default="I have 3 apples, my dad has 2 more apples than me, how many apples do we have in total?")
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=300, type=int)
    parser.add_argument('--stop_criteria', default=['Q:', '\n\nQ:'], type=list)
    parser.add_argument('--add_shot', default='', type=str)
    #parser.add_argument('--pattern', default=r'-?\d+(?:\.\d+)?(?![.\d])', type=str)
    parser.add_argument('--pattern', default=r'\b-?(?:\d+(?:\.\d*)?|\.\d+)\b(?![\d])', type=str)
    parser.add_argument('--dtype', default='float32', type=str)
    parser.add_argument('--max_model_len', default=2048, type=int)
    parser.add_argument('--quantize', type=str)
    
    # [GENERATE DATASET]
    parser.add_argument('--dataset', nargs='+', choices=['gsm8k', 'multiarith', 'svamp', 'last', 'singleq', 'addsub', 'coin'])
    parser.add_argument('--dataset_path', default='./dataset', type=str)
    parser.add_argument('--log_path', default='/content/log', type=str)
    parser.add_argument('--plot_path', default='/content/plot', type=str)
    parser.add_argument('--field', default='test', type=str)
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_plot', action='store_true')
    parser.add_argument('--dataset_type', default='aritmetic', type=str)
    parser.add_argument('--log_outputs_path', type=str)
    parser.add_argument('--cot_name', type=str)
    parser.add_argument('--max_samples', default=-1, type=int)
    parser.add_argument('--init_samples', default=-1, type=int)

    # New prompt tuning arguments
    parser.add_argument('--soft_prompt_length', type=int, default=20,
                        help='Number of virtual tokens for prompt tuning')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for prompt tuning')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--prompt_save_path', type=str, default='.',
                        help='Path to save trained prompt')
    parser.add_argument('--prompt_load_path', type=str,
                        help='Path to load existing prompt')
    parser.add_argument('--prompt_init_method', type=str, default='TEXT',
                        choices=['RANDOM', 'TEXT'],
                        help='Method to initialize prompt embeddings')
    parser.add_argument('--lambda_', type=float, default=1.0,
                        help='Weight for incorrect answer penalty')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='Weight for correct answer reward')
    parser.add_argument('--prompt_init_text', type=str, 
                       default="Let's think step by step.",
                       help='Text to initialize prompt tuning when using TEXT method')
    
    # New arguments for weighted loss
    parser.add_argument('--position_alpha', type=float, default=0.7,
                       help='Position weight decay factor (higher = slower decay)')
    parser.add_argument('--top_k_penalty', type=int, default=3,
                       help='Number of top positions to apply extra penalty')
    parser.add_argument('--top_k_weight', type=float, default=2.0,
                       help='Weight of penalty for incorrect answers in top k positions')
    
    args = parser.parse_args()
    
    return args

def compute_soft_prompt_loss(cot_paths, ground_truth, args):
    """
    Compute position-weighted loss for soft prompt training.
    
    Loss incorporates:
    1. Position-based weighting (earlier positions have higher weights)
    2. Correctness-based rewards/penalties
    3. Confidence score scaling
    
    Parameters:
        cot_paths: Chain-of-thought paths containing reasoning and answers
        ground_truth: The correct answer
        args: Arguments containing hyperparameters:
            - mu: reward weight for correct answers
            - lambda_: penalty weight for incorrect answers
            - position_alpha: controls position weight decay (default=0.85)
    """
    loss = 0
    num_paths = len(cot_paths.paths)
    
    # Position weight decay factor (can be added as hyperparameter)
    alpha = getattr(args, 'position_alpha', 0.85)
    
    for idx, path in enumerate(cot_paths.paths):
        # Calculate position-based weight (decreasing with position)
        # Using exponential decay: alpha^position
        position_weight = alpha ** idx
        
        # Check if answer is correct
        is_correct = (path.answer_span == ground_truth)
        
        # Get confidence score
        confidence = path.score
        
        # Compute weighted loss term for this path
        if is_correct:
            # Reward correct answers (higher reward for earlier positions)
            loss -= args.mu * position_weight * confidence
        else:
            # Penalize incorrect answers (higher penalty for earlier positions)
            loss += args.lambda_ * position_weight * confidence
            
            # Additional penalty for incorrect answers in top k positions
            if idx < args.top_k_penalty:
                loss += args.top_k_weight * confidence
    loss /= num_paths
    return torch.tensor(loss, requires_grad=True)

def save_soft_prompt(args, model, save_path):
    """Save the PEFT model state."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Soft prompt saved to {save_path}")
        
def convert_to_string(example):
    example['answer'] = str(example['answer'])
    return example

def concatenate_fields(example):
    example['question'] = example['Body'] + '. ' + example['Question']
    return example

def remove_characters(example):
    example['question'] = example['question'][2:]  # Remove o primeiro caractere
    return example

def load_datasets(datasets: List[str]):

    loaded_datasets = {}

    # math tasks
    for dataset_name in datasets:
        if dataset_name == 'gsm8k':
            loaded_datasets['gsm8k'] = load_dataset('gsm8k', 'main')
        elif dataset_name == 'multiarith':
            dataset = load_dataset('ChilleD/MultiArith')
            dataset = dataset.rename_column('final_ans', 'answer')
            dataset = dataset.map(convert_to_string)
            loaded_datasets['multiarith'] = dataset
        elif dataset_name == 'svamp':
            dataset = load_dataset('ChilleD/SVAMP')
            dataset = dataset.rename_column('Answer', 'answer')
            dataset = dataset.map(concatenate_fields)
            loaded_datasets['svamp'] = dataset
        elif dataset_name == 'singleq':
            dataset = load_dataset("allenai/lila", "singleq")
            dataset = dataset.rename_column('input', 'question')
            dataset = dataset.rename_column('output_answer', 'answer')
            loaded_datasets['singleq'] = dataset
        elif dataset_name == 'addsub':
            dataset = load_dataset("allenai/lila", "addsub")
            dataset = dataset.rename_column('input', 'question')
            dataset = dataset.rename_column('output_answer', 'answer')
            loaded_datasets['addsub'] = dataset
        elif dataset_name == 'coin':
            dataset = load_dataset("skrishna/coin_flip")
            dataset = dataset.rename_column('inputs', 'question')
            dataset = dataset.rename_column('targets', 'answer')
            dataset = dataset.map(remove_characters)
            loaded_datasets['coin'] = {split: dataset.select(range(100)) for split, dataset in dataset.items()}
        elif dataset_name == 'last':
            loaded_datasets['last'] = load_dataset("ChilleD/LastLetterConcat")

    return loaded_datasets

def save_logs(args, log_outputs: Dict, dtype: str, model_name: str, dataset_name: str, log_path: str):

    # Create log directory and save outputs
    log_dir = os.path.join('log', str(args.topk), f'{model_name}_{args.add_shot}')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{dataset_name}_{model_name}_{dtype}_cot_s{args.soft_prompt_length}.json')
    with open(log_file, 'w') as f:
        json.dump(log_outputs, f, indent=4)
    
    print(f"Log saved to: {log_file}")

def extract_cot_paths_from_dataset(dataset: DatasetDict,
                                      dataset_name: str,
                                      generator: GeneratePaths,
                                      cot_decoding: CoTDecoding,
                                      beam_search_decoding: BeamSearchDecoding,
                                      max_samples: int,
                                      init_samples: int,
                                      prompt_key: str,
                                      field: Optional[Literal['train', 'val', 'test']],
                                      beam_search_flag: bool):

    if field is not None:
        dataset = dataset[field]
    else:
        raise ValueError("Field must be one of 'train', 'val', or 'test'.")
        
    prompts = dataset[prompt_key][init_samples:max_samples]
    
    log_outputs = {'cot_decoding': {}, 'beam_search_decoding': {}}
    
    for i, prompt in enumerate(tqdm(prompts, desc=dataset_name, total=len(prompts))):
        
        topk_tokens, outputs = generator.search_cots(prompt)
        cot_paths = cot_decoding.calculate_score(prompt, topk_tokens, outputs)
        log_outputs['cot_decoding'][f'idx={i}:{prompt}'] = asdict(cot_paths)['paths']

        #TODO: beam search decoding
        if beam_search_flag:
            beam_search_outputs = generator.beam_search_decoding(prompt)
            beam_search_paths = beam_search_decoding.calculate_score(prompt, None, beam_search_outputs)        
            log_outputs['beam_search_decoding'][f'idx={i}:{prompt}'] = asdict(beam_search_paths)['paths']

    return log_outputs

def greedy_decoding(log_outputs: Dict) -> List:
    
    return [paths[0]['answer_span'] for key, paths in log_outputs.items()]

def choose_specific_depth(log_outputs: Dict, k: int) -> List:
    
    return [paths[k]['answer_span'] for key, paths in log_outputs.items()]

def best_score(log_outputs: Dict, k: int) -> List:
    
    return [max(paths[:k], key=lambda x: x['score'])['answer_span'] for key, paths in log_outputs.items()]

def self_consistency(log_outputs: Dict, beam_search: bool = False, k: int = 10) -> List:
    
    consistency_answer_span = []
    for key, paths in log_outputs.items():
        consistency = {}
        for path in paths[:k]:
            if len(path['answer_span']) > 0:
                if path['answer_span'] not in consistency:
                    consistency[path['answer_span']] = 0
                if beam_search:
                    consistency[path['answer_span']] += numpy.exp(path['score'])
                else:
                    consistency[path['answer_span']] += path['score']
            
        if len(consistency) > 0:
            major_answer_span = max(consistency, key=consistency.get)
            consistency_answer_span.append(max([item for item in paths if item['answer_span'] == major_answer_span], key=lambda x: x['score'])['answer_span'])
        else:
            consistency_answer_span.append(paths[0]['answer_span'])
            
    return consistency_answer_span

def exact_match(predicted, ground_truth):
    
    return sum([pred == gt for pred, gt in zip(predicted, ground_truth)]) / len(ground_truth)

def extract_exact_match(log_outputs, ground_truth = None, pattern=r'\b-?(?:\d+(?:\.\d*)?|\.\d+)\b(?![\d])'):
    for method in log_outputs.keys():
        for i, (key, paths) in enumerate(log_outputs[method].items()):
            gt = ground_truth[i] if ground_truth is not None else None
            for path in paths:
                reasoning_text = str(path.get('reasoning_text', ''))
                answer_span = re.findall(pattern, reasoning_text)
                if answer_span:
                    # Extract only the numerical part
                    numerical_part = re.search(r'-?\d+(?:\.\d+)?(?![.\d])', answer_span[-1])
                    if numerical_part:
                        path['answer_span'] = numerical_part.group()
                    else:
                        path['answer_span'] = ''
                else:
                    path['answer_span'] = ''
                # Add ground truth and match status if ground truth is provided
                if gt is not None:
                    path['ground_truth'] = gt
                    path['matches_ground_truth'] = (path['answer_span'] == gt)
    return log_outputs

                
def evaluating(beam_search_flag,log_outputs, ground_truth, pattern: str, num_paths: int, tokenizer: AutoTokenizer, max_new_tokens: int):
    
    evaluations = []
    
    for k in range(0, num_paths):
        
        log_outputs = extract_exact_match(log_outputs, ground_truth = ground_truth, pattern=pattern)
        
        greedy_decoding_acc = exact_match(greedy_decoding(log_outputs['cot_decoding']), ground_truth)                       
        cot_decoding_max_acc = exact_match(best_score(log_outputs['cot_decoding'], k=k+1), ground_truth)
        cot_decoding_agg_acc = exact_match(self_consistency(log_outputs['cot_decoding'], k=k+1), ground_truth)
        if beam_search_flag:            
            beam_search_greedy_acc = exact_match(greedy_decoding(log_outputs['beam_search_decoding']), ground_truth)
            beam_search_max_acc = exact_match(best_score(log_outputs['beam_search_decoding'], k=k+1), ground_truth)
            beam_search_agg_acc = exact_match(self_consistency(log_outputs['beam_search_decoding'], beam_search=True, k=k+1), ground_truth)
            evaluations.append({
                'Greedy Decoding': greedy_decoding_acc,
                'CoT-Decoding (max)': cot_decoding_max_acc,
                'CoT-Decoding (agg)': cot_decoding_agg_acc,
                'Beam Search Greedy': beam_search_greedy_acc,
                'Beam Search (max)': beam_search_max_acc,
                'Beam Search (agg)': beam_search_agg_acc
            })
        else:
            evaluations.append({
                'Greedy Decoding': greedy_decoding_acc,
                'CoT-Decoding (max)': cot_decoding_max_acc,
                'CoT-Decoding (agg)': cot_decoding_agg_acc
            })

    return evaluations

def print_evaluations(beam_search_flag: bool, evaluations: List, dataset_name: str):
    
    print(f'--- Evaluation using Exact Match for {dataset_name} ---')
    print(f'\tGreedy Decoding: {evaluations[-1]["Greedy Decoding"]:.4f}')
    print(f'\tCoT-Decoding (max): {evaluations[-1]["CoT-Decoding (max)"]:.4f}')
    print(f'\tCoT-Decoding (agg): {evaluations[-1]["CoT-Decoding (agg)"]:.4f}')
    if beam_search_flag:
        print(f'\tBeam Search Greedy: {evaluations[-1]["Beam Search Greedy"]:.4f}')
        print(f'\tBeam Search (max): {evaluations[-1]["Beam Search (max)"]:.4f}')
        print(f'\tBeam Search (agg): {evaluations[-1]["Beam Search (agg)"]:.4f}')

def load_cot_dataset(args):

    log_outputs = {'cot_decoding': {}}

    with open(f'{args.log_path}/{args.dataset[0]}_cot_decoding.json', 'r') as f:
        log_outputs['cot_decoding'] = json.load(f)

    return log_outputs

def save_plot(args, evaluations: Dict, title: str, plot_path: str):
    """Save evaluation plots and data to files.
    
    Args:
        args: Argument namespace containing topk, add_shot, and beam_search
        evaluations (Dict): Evaluation results
        title (str): Plot title
        plot_path (str): Base path for saving plot
    """
    # Setup directories
    fig_dir = os.path.join('fig', str(args.topk))
    os.makedirs(fig_dir, exist_ok=True)
    
    # Generate file paths
    base_filename = f"{plot_path}_{args.add_shot}"
    plot_file = os.path.join(fig_dir, f"{base_filename}_s{args.soft_prompt_length}.png")
    data_file = os.path.join(fig_dir, f"{base_filename}_s{args.soft_prompt_length}.json")
    
    # Ensure plot directory exists
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    
    # Define plot components
    components = ['Greedy Decoding', 'CoT-Decoding (max)', 'CoT-Decoding (agg)']
    symbols = ['o', 's', 'd']
    colors = [(0.4, 0.8, 0.4, 0.8), (0.4, 0.4, 0.8, 0.8), (0.8, 0.4, 0.4, 0.8)]
    
    # Add beam search components if enabled
    if args.beam_search:
        components.extend(['Beam Search Greedy', 'Beam Search (max)', 'Beam Search (agg)'])
        symbols.extend(['^', 'v', 'p'])
        colors.extend([(0.8, 0.8, 0.4, 0.8), (0.4, 0.8, 0.8, 0.8), (0.8, 0.4, 0.8, 0.8)])
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plot_data = {}
    
    # Plot each component
    for idx, comp in enumerate(components):
        values = [eval_data[comp] for eval_data in evaluations]
        plot_data[comp] = values
        plt.plot(range(1, len(evaluations) + 1), values, 
                marker=symbols[idx], color=colors[idx], label=comp)
    
    # Configure plot
    plt.title(title)
    plt.xlabel('Paths (K)')
    plt.ylabel('Exact Match Acc')
    plt.xticks(range(1, len(evaluations) + 1))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot and data
    plt.savefig(plot_file)
    with open(data_file, 'w') as f:
        json.dump(plot_data, f, indent=4)
    
    print(f"Plot saved to: {plot_file}")
    print(f"Plot data saved to: {data_file}")

def compute_combined_loss(cot_paths, ground_truth, args):
    """
    Compute combined pointwise and pairwise loss.
    """
    # Pointwise loss to increase confidence of correct paths
    pointwise_loss = 0
    for path in cot_paths.paths:
        is_correct = (path.answer_span == ground_truth)
        if is_correct:
            pointwise_loss += -args.mu * path.score  # Reward correct answers
        else:
            pointwise_loss += args.lambda_ * path.score  # Penalize incorrect answers

    # Pairwise ranking loss
    margin = args.margin
    pairwise_loss = 0
    correct_paths = [path for path in cot_paths.paths if path.answer_span == ground_truth]
    incorrect_paths = [path for path in cot_paths.paths if path.answer_span != ground_truth]

    for correct_path in correct_paths:
        for incorrect_path in incorrect_paths:
            pairwise_loss += torch.clamp(margin - (correct_path.score - incorrect_path.score), min=0)

    total_loss = pointwise_loss + args.beta * pairwise_loss  # beta balances the two losses

    return total_loss

def compute_cot_accuracy(cot_paths, ground_truth):
    """
    Compute CoT accuracy by checking if the path with highest confidence has the correct answer.
    
    Args:
        cot_paths: Chain-of-thought paths containing reasoning and answers
        ground_truth: The correct answer
    
    Returns:
        is_correct: Boolean indicating if the highest confidence path has correct answer
        best_confidence: The confidence score of the highest confidence path
    """
    # Group paths by answer and aggregate their scores
    answer_scores = {}
    for path in cot_paths.paths:
        answer = path.answer_span
        if answer not in answer_scores:
            answer_scores[answer] = path.score
        else:
            answer_scores[answer] = max(answer_scores[answer], path.score)
    
    # Find answer with highest aggregated score
    best_answer = max(answer_scores.items(), key=lambda x: x[1])
    
    return best_answer[0] == ground_truth, best_answer[1]