from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from peft import PromptTuningConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from decoding import *

def configurate_method(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Model Configuration
    tokenizer = setup_tokenizer(args.model)
    model = load_base_model(args)
    model = setup_prompt_tuning(model, args)

    assistant_model = load_assistant_model(args)
    # Check if using zero-shot or few-shot learning
    if args.add_shot == '':
        print('Zero Shot Learning, Not providing any extra prompt.')
    else:
        print(f'ZERO-SHOT-COT: "{args.add_shot}"')
    
    # Initialize the path generator
    generator = GeneratePaths(model=model,
                            assistant_model=assistant_model,
                            tokenizer=tokenizer,
                            topk=args.topk, 
                            max_new_tokens=args.max_new_tokens, 
                            stop=args.stop_criteria,
                            prompt=args.add_shot)

    # Initialize the Chain-of-Thought decoding
    cot_decoding = CoTDecoding(pattern=args.pattern,
                               tokenizer=tokenizer,
                               prompt=args.add_shot)

    # Initialize the Beam Search decoding
    beam_search_decoding = BeamSearchDecoding(pattern=args.pattern,
                                              tokenizer=tokenizer,
                                              prompt=args.add_shot)

    return generator, cot_decoding, beam_search_decoding, tokenizer

def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_quantization_config(quantize_type):
    return BitsAndBytesConfig(
        load_in_4bit=quantize_type == '4bit',
        load_in_8bit=quantize_type == '8bit',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
def load_base_model(args):
    if args.quantize:
        quantization_config = get_quantization_config(args.quantize)
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    print("Loading model without quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=getattr(torch, args.dtype)
    )
    print("Model loaded successfully.")
    return model

def load_assistant_model(args):
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return assistant_model

def setup_prompt_tuning(model, args):
    # Load existing prompt if specified, no matter evaluating or training
    if args.prompt_load_path:
        config = PeftConfig.from_pretrained(args.prompt_load_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=getattr(torch, args.dtype)
        )
        model = PeftModel.from_pretrained(model, args.prompt_load_path)
        
        # Add these debugging lines:
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Force model to GPU if needed
        if torch.cuda.is_available():
            model = model.cuda()
        model.print_trainable_parameters()
        return model
    if args.choice == 'train_prompt':
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.soft_prompt_length,
            prompt_tuning_init=args.prompt_init_method,
            tokenizer_name_or_path=args.model,
            prompt_tuning_init_text=args.prompt_init_text
        )
        return get_peft_model(model, peft_config)
    return model