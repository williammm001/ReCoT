import re
import torch
import numpy as np

from typing import List, Dict, Optional, Tuple
from vllm import LLM, SamplingParams
from scipy.stats import entropy
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@dataclass
class Path:
    """
    Represents a single reasoning path in the Chain-of-Thought decoding process.
    
    Attributes:
        reasoning_text (str): The generated reasoning text for this path.
        score (float): The calculated score for this path.
        answer_span (str): The extracted answer span from the reasoning.
        num_path (int): The index of this path in the list of generated paths.
        numeric_score (List[Tuple[str, float]]): The calculated numeric score for this path.
        attention_scores (Dict[str, List[Tuple[str, float]]]): Attention scores for this path.
    """
    reasoning_text: str
    score: float
    answer_span: str
    num_path: int
    numeric_score: List[Tuple[str, float]]
    attention_scores: Dict[str, List[Tuple[str, float]]]
    hit_max_token_numbers: bool = False

@dataclass
class DecodingInfo:
    """
    Contains information about the decoding process for a given question.
    
    Attributes:
        question (str): The input question.
        paths (List[Path]): A list of Path objects representing different reasoning paths.
    """
    question: str
    paths: List[Path]

class GeneratePaths():
    """
    Handles the generation of multiple reasoning paths for a given input prompt.
    """

    def __init__(self, model: AutoModelForCausalLM, assistant_model: AutoModelForCausalLM=None,
                       tokenizer: AutoTokenizer=None,
                       max_new_tokens: int=300, 
                       topk: int=10, 
                       stop: List[str]=['Q:', '\n\nQ:', '\n\nExercise'],
                       prompt: str=''):
        """
        Initialize the GeneratePaths class.

        Args:
            model (AutoModelForCausalLM): The language model from HuggingFace transformers
            tokenizer (AutoTokenizer): The tokenizer from HuggingFace transformers
            max_new_tokens (int): Maximum number of tokens to generate for each response
            topk (int): Maximum number of paths to explore for the first token of the response
            stop (List[str]): List of stopping criteria for text generation
            prompt (str): Additional prompt text to be added before the main input
        """
        self.model = model
        self.assistant_model = assistant_model
        self.tokenizer = tokenizer
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.stop = stop
        self.topk = topk
        self.prompt = prompt

    def search_cots(self, raw_prompt: str) -> List[str]:
        """
        Generate multiple Chain-of-Thought paths for a given prompt.

        Args:
            raw_prompt (str): The input prompt to generate paths for.

        Returns:
            tuple: Contains the top-k tokens and their probabilities, and the generated outputs.
        """
        # Format the raw prompt into a predefined template format.
        formatted_prompt = self.format_prompt(raw_prompt)
        # Explore the first K paths of the response using greedy decoding.
        topk_tokens = self.get_first_topk_tokens(formatted_prompt)
        prompts = [formatted_prompt + token for token in topk_tokens['decoded']] # K questions.
        # Continue generating the K paths for the remaining M - 1 tokens.
        outputs = self.generate_paths(prompts)
        
        return topk_tokens, outputs
    
    @torch.inference_mode(False)
    def get_first_topk_tokens(self, prompt: str) -> Dict[str, List]:
        """
        Get the top-k tokens for the first step of generation using Transformers API.

        Args:
            prompt (str): The formatted input prompt.

        Returns:
            Dict[str, List]: Contains decoded tokens, probabilities, token IDs, and log probabilities.
        """
        # Encode the prompt
        encoding = self.tokenizer(prompt, return_tensors="pt", return_length=True)
        input_length = encoding.length[0]
        
        # Remove length from inputs before passing to model
        inputs = {
            'input_ids': encoding['input_ids'].to(self.model.device),
            'attention_mask': encoding['attention_mask'].to(self.model.device),
            'output_hidden_states': False,
            'output_attentions': False,  # Changed to False since we don't need it
            'return_dict': True
        }
        
        # Get the logits for just the next token, DO NOT USE ASSISTANT MODEL
        outputs = self.model.forward(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Get top-k logits and indices
        topk_logits, topk_indices = torch.topk(next_token_logits, k=self.topk)
        topk_probs = torch.softmax(topk_logits, dim=0)
        topk_log_probs = torch.log(topk_probs)

        topk_tokens = {
            'decoded': [self.tokenizer.decode(idx) for idx in topk_indices],
            'probs': topk_probs.tolist(),
            'token_id': topk_indices.tolist(),
            'logprobs': [{int(idx.item()): type('LogProb', (), {
                'logprob': log_prob.item(),
                'decoded_token': self.tokenizer.decode(idx)
            })} for idx, log_prob in zip(topk_indices, topk_log_probs)]
        }
        
        return topk_tokens
    
    @torch.inference_mode(False)  # Allow gradient tracking during training
    def generate_paths(self, prompts: List[str]) -> Dict[int, Dict]:
        """
        Generate multiple paths for the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate paths for.

        Returns:
            Dict[int, Dict]: Generated outputs for each prompt.
        """
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Convert prompts to input_ids
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        # Configure generation parameters
        gen_kwargs = {
            'num_return_sequences': 1,
            'max_new_tokens': self.max_new_tokens,
            'temperature': 0.0,
            'top_p': 1.0,
            'do_sample': False,
            'return_dict_in_generate': True,
            'output_attentions': True,
            'output_scores': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Add stopping criteria if needed
        if self.stop:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopOnTokens(StoppingCriteria):
                def __init__(self, stop_token_ids, device):
                    self.stop_token_ids = [ids.to(device) for ids in stop_token_ids]  # Move to correct device
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_ids in self.stop_token_ids:
                        if torch.all((input_ids[0][-len(stop_ids):] == stop_ids)).item():
                            return True
                    return False
            
            stop_token_ids = [self.tokenizer(stop_text, return_tensors='pt')['input_ids'][0]
                             for stop_text in self.stop]
            gen_kwargs['stopping_criteria'] = StoppingCriteriaList([StopOnTokens(stop_token_ids, self.model.device)])
        
        # Generate
        outputs = self.model.generate(**inputs, **gen_kwargs, assistant_model=self.assistant_model)
        #print("Test: ",outputs.attentions[0][-1].shape)
        # Convert to a format similar to vLLM's output
        results = []
        for i, output_ids in enumerate(outputs.sequences):
            # Get generated tokens without special tokens
            generated_ids = output_ids[inputs['input_ids'].shape[1]:]
            # Filter out special tokens (like EOS, PAD tokens)
            non_special_tokens = [token_id for token_id in generated_ids 
                                if token_id not in [self.tokenizer.eos_token_id, 
                                                  self.tokenizer.pad_token_id]]
            hit_max_token_numbers = len(non_special_tokens) >= self.max_new_tokens
            generated_text = self.tokenizer.decode(output_ids[inputs['input_ids'].shape[1]:], 
                                                 skip_special_tokens=True)
            sequence_scores = [score[i] for score in outputs.scores]
            
            # Create attention dictionary for each generated token
            attention_dict = {}
            
            # Safely handle attention scores
            if outputs.attentions is not None and len(outputs.attentions) > 0:
                # Get all tokens (input + generated)
                all_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)
                input_length = inputs['input_ids'].shape[1]
                
                # For each generation step
                for step, step_attentions in enumerate(outputs.attentions):
                    # Get the last layer's attention weights for this step
                    last_layer_attention = step_attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
                    # Average across heads for the current sequence
                    attention_scores = last_layer_attention[i].mean(dim=0)  # Shape: [seq_len, seq_len]
                    
                    # The token generated at this step
                    current_token = all_tokens[input_length + step]
                    # Get attention scores for the current token (last row of attention matrix)
                    curr_attention = attention_scores[-1]  # Shape: [seq_len]
                    
                    # Store attention to all previous tokens
                    attention_dict[current_token] = [
                        (prev_token, float(curr_attention[j].item()))
                        for j, prev_token in enumerate(all_tokens[:input_length + step])
                    ]
            
            result = type('GenerationOutput', (), {
                'outputs': [type('Output', (), {
                    'text': generated_text,
                    'logprobs': [self._convert_scores_to_logprobs(score) for score in sequence_scores],
                    'attention_scores': attention_dict,
                    'hit_max_token_numbers': hit_max_token_numbers
                })]
            })
            results.append(result)
        
        return results
    
    def _convert_scores_to_logprobs(self, scores):
        """Helper method to convert scores to logprobs format similar to vLLM."""
        probs = torch.softmax(scores, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=2)
        log_probs = torch.log(top_probs)
        
        return {int(idx.item()): type('LogProb', (), {
            'logprob': log_prob.item(),
            'decoded_token': self.tokenizer.decode(idx)
        }) for idx, log_prob in zip(top_indices, log_probs)}
    
    def format_prompt(self, raw_prompt: str) -> str:
        """
        Format the raw prompt to match the expected question-answer template.

        Args:
            raw_prompt (str): The original input prompt.

        Returns:
            str: Formatted prompt with question and answer parts.
        """
        return f'Q:{raw_prompt}\nA:{self.prompt}'
    
    def beam_search_decoding(self, prompt: str) -> Dict[int, Dict]:
        """
        Perform beam search decoding for the given prompt.

        Args:
            prompt (str): The input prompt for beam search decoding.

        Returns:
            Dict[int, Dict]: Generated outputs using beam search decoding.
        """
        # Set the environment variable to allow deprecated beam search
        import os
        os.environ['VLLM_ALLOW_DEPRECATED_BEAM_SEARCH'] = '1'
        
        sampling_params = SamplingParams(n=self.topk, best_of=self.topk, use_beam_search=True, 
                                         temperature=0, max_tokens=self.max_new_tokens, 
                                         stop=self.stop, logprobs=1)
        
        # Use the dedicated beam search method
        import json
        outputs = self.model.generate(prompt, sampling_params, use_tqdm=False)
        # Assume 'outputs' is a dictionary or a structured object
        output_data = [output.to_dict() if hasattr(output, 'to_dict') else str(output) for output in outputs]

        return outputs

class CoTDecoding():
    """
    Handles the Chain-of-Thought decoding process, including score calculation and answer span extraction.
    """
    
    def __init__(self, pattern: str=r'\b-?(?:\d+(?:\.\d*)?|\.\d+)\b(?![\d])',
                       tokenizer: AutoTokenizer=None,
                       prompt: str='',
                       dataset_type: str = 'aritmetic'):
        """
        Initialize the CoTDecoding class.

        Args:
            pattern (str): Regex pattern for extracting numerical answer spans.
            tokenizer (AutoTokenizer): Tokenizer for encoding and decoding text.
            prompt (str): Additional prompt text to be used in formatting.
            dataset_type (str): Type of dataset being processed (e.g., 'arithmetic', 'symbolic', 'commonsense').
        """
        self.pattern = pattern
        self.tokenizer = tokenizer
        # Set padding side to left for decoder-only models
        if self.tokenizer is not None:  # Add check for None
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_type = dataset_type

    def calculate_score(self, prompt, topk_tokens, outputs):
        """
        Calculate the score for each path based on answer span probabilities.

        Args:
            prompt (str): The input prompt.
            topk_tokens (Dict): Dictionary containing the top-k tokens and their log probabilities.
            outputs (Dict): Dictionary containing the outputs from the model.

        Returns:
            DecodingInfo: An object containing the question and the scored paths.
        """
        paths = []
        
        for k, output in enumerate(outputs):
            reasoning = topk_tokens['decoded'][k] + output.outputs[0].text
            hit_max_token_numbers = output.outputs[0].hit_max_token_numbers
            # Encode the reasoning text and obtain offset mappings.
            encode = self.tokenizer(reasoning, return_offsets_mapping=True)

            # Extract answer span based on dataset type
            if self.dataset_type == 'aritmetic':
                answer_span = re.findall(self.pattern, reasoning)
            elif self.dataset_type == 'symbolic':
                answer_span = ''
            elif self.dataset_type == 'commonsense':
                answer_span = ''
            
            score = 0
            score_list = []
            answer_attention_scores = {}  # Store attention scores only for answer spans

            if len(answer_span):
                output.outputs[0].logprobs.insert(0, topk_tokens['logprobs'][k])
                positions = self.find_all_positions(reasoning, answer_span)
                
                # Only process numbers that have corresponding positions
                for idx, a_s in enumerate(answer_span):
                    # Find matching position for current answer span
                    matching_position = next((pos for pos in positions if pos['number'] == a_s), None)
                    if matching_position is None:
                        continue  # Skip this answer span if no position found
                        
                    d_t = ""
                    # Use the matched position's span
                    last_pattern_span = matching_position['span']
                    idx_answer = []
                    for i, span in enumerate(encode.offset_mapping):
                        if span[0] >= last_pattern_span[0] and span[1] <= last_pattern_span[1]:
                            # Token span is completely within the answer span
                            tid = encode.input_ids[i]
                            d_t += self.tokenizer.decode(tid)
                            idx_answer.append(i)
                            # print(f"Token {i} is completely within the answer span")
                        elif span[0] <= last_pattern_span[0] and span[1] >= last_pattern_span[1]:
                            idx_answer.append(i)
                            tid = encode.input_ids[i]
                            d_t += self.tokenizer.decode(tid)
                        elif span[0] <= last_pattern_span[0] and span[1] > last_pattern_span[0]:
                            idx_answer.append(i)
                            tid = encode.input_ids[i]
                            d_t += self.tokenizer.decode(tid)
                        if a_s in d_t:
                            break
                    token_id = [encode.input_ids[idx] for idx in idx_answer]
                    
                    # Filter log probabilities for tokens in the answer span.
                    adjusted_indices = [i-1 for i in idx_answer]
                    filtered_answer = [output for i, output in enumerate(output.outputs[0].logprobs) 
                                    if i in adjusted_indices and i >= 0]  # ensure we don't get negative indices
                    sum_answer_span_probs = 0

                    for logprob_dict in filtered_answer:
                        logprob_list = list(logprob_dict.items())
                        # print("\n\n____")
                        # print("Text:\n", reasoning)
                        # print("Current token:", a_s, "|Current subtoken: ", logprob_list[0][1].decoded_token)
                        # print("____\n\n")
                        if len(logprob_list) == 2:
                            prob_diff = (torch.exp(torch.tensor([logprob_list[0][1].logprob])) - torch.exp(torch.tensor([logprob_list[1][1].logprob]))).item()
                        else:
                            prob_diff = torch.exp(torch.tensor([logprob_list[0][1].logprob])).item()
                        sum_answer_span_probs += prob_diff

                    # Calculate the score as the average probability difference.
                    score = 0 if len(filtered_answer) == 0 else sum_answer_span_probs / len(filtered_answer)
                    a_s = self.tokenizer.decode(token_id).strip()
                    numeric_score = (a_s, score)
                    score_list.append(numeric_score)

                    # Get attention scores for this answer span
                    if hasattr(output.outputs[0], 'attention_scores'):
                        # Get all tokens that make up this answer span
                        answer_tokens = self.tokenizer.tokenize(a_s)
                        for token in answer_tokens:
                            if token in output.outputs[0].attention_scores:
                                answer_attention_scores[token] = output.outputs[0].attention_scores[token]
                    
                    score_list.append((a_s, score))
            else:
                a_s = '|<NotFounded>|'
            
            paths.append(Path(
                reasoning_text=reasoning, 
                score=score,
                answer_span=a_s,
                num_path=k,
                numeric_score=score_list,
                hit_max_token_numbers=hit_max_token_numbers,
                attention_scores=answer_attention_scores  # Add the filtered attention scores
            ))
        
        # Create the output object with the prompt and paths.
        output = DecodingInfo(
            question=prompt,
            paths=paths
        )
        
        return output
    def find_all_positions(self,reasoning_text: str, numbers: list) -> list:
        positions = []
        seen_positions = set()  # To track positions we've already found
        
        for number in numbers:
            # Find all positions of this number
            start = 0
            while True:
                # Find next occurrence of the number starting from 'start'
                pos = reasoning_text.find(number, start)
                if pos == -1:  # No more occurrences found
                    break
                    
                # Check if this is a whole number (not part of another number)
                is_whole_number = True
                if pos > 0 and reasoning_text[pos-1].isdigit():
                    is_whole_number = False
                if pos + len(number) < len(reasoning_text) and reasoning_text[pos + len(number)].isdigit():
                    is_whole_number = False
                    
                # Only add if we haven't seen this position and it's a whole number
                if pos not in seen_positions and is_whole_number:
                    positions.append({
                        'number': number,
                        'span': (pos, pos + len(number))
                    })
                    seen_positions.add(pos)
                
                start = pos + 1  # Move start to look for next occurrence
                
        # Sort positions by their appearance in text
        positions.sort(key=lambda x: x['span'][0])
        return positions

class BeamSearchDecoding():
    """
    Handles the Beam Search decoding process, including score calculation and answer span extraction.
    """

    def __init__(self, pattern: str=r'\b-?(?:\d+(?:\.\d*)?|\.\d+)\b(?![\d])',
                       tokenizer: AutoTokenizer=None,
                       prompt: str='',
                       dataset_type: str = 'aritmetic'):
        """
        Initialize the BeamSearchDecoding class.

        Args:
            pattern (str): Regex pattern for extracting numerical answer spans.
            tokenizer (AutoTokenizer): Tokenizer for encoding and decoding text.
            prompt (str): Additional prompt text to be used in formatting.
            dataset_type (str): Type of dataset being processed (e.g., 'arithmetic', 'symbolic', 'commonsense').
        """
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type

    def calculate_score(self, prompt, topk_tokens, outputs):
        paths = []

        for k, output in enumerate(outputs[0].outputs):

            reasoning = output.text
            
            # Extract answer span based on dataset type
            if self.dataset_type == 'aritmetic':
                answer_span = re.findall(self.pattern, reasoning)
            elif self.dataset_type == 'symbolic':
                answer_span = ''  # Implement symbolic extraction if needed
            elif self.dataset_type == 'commonsense':
                answer_span = ''  # Implement commonsense extraction if needed
            
            score = output.cumulative_logprob  # Use the cumulative log probability as the score
            
            if len(answer_span):
                answer_span = answer_span[-1]  # Use the last found answer span
            else:
                answer_span = '|<NotFound>|'
            
            paths.append(Path(reasoning_text=reasoning, 
                              score=score,
                              answer_span=answer_span,
                              num_path=k))
        
        output = DecodingInfo(
            question=prompt,
            paths=paths
        )
        
        return output

