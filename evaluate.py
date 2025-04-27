from model_config import *
from utils import *

def evaluating_decoding_methods(args):
    # Set up the evaluation environment
    generator, cot_decoding, beam_search_decoding, tokenizer = configurate_method(args)
    datasets = load_datasets(args.dataset)
    if args.beam_search:
        beam_search_flag = True
    else:
        beam_search_flag = False
    for dataset in datasets:
        # Load existing outputs if specified, otherwise generate new ones
        if args.log_outputs_path:
            log_outputs = {'cot_decoding': {}}
            log_outputs['cot_decoding'] = json.load(args.log_outputs_path + '/' + args.cot_name)
        else:
            # Determine the number of samples to process, -1 means all samples
            if args.max_samples == -1:
                max_samples = datasets[dataset][args.field].num_rows
            else:
                max_samples = args.max_samples
            
            if args.init_samples == -1:
                init_samples = 0
            else:
                init_samples = args.init_samples
            
            # Extract CoT paths from the dataset
            log_outputs = extract_cot_paths_from_dataset(dataset=datasets[dataset],
                                            dataset_name=dataset,
                                            max_samples=max_samples,
                                            init_samples=init_samples,
                                            field=args.field,
                                            prompt_key='question',
                                            generator=generator,
                                            cot_decoding=cot_decoding,
                                            beam_search_decoding=beam_search_decoding,
                                            beam_search_flag = beam_search_flag      
            )

        # Save logs if specified
        if args.save_log:
            model_name = args.model.split("/")[-1]
            save_logs(args,log_outputs, args.dtype, model_name, dataset, args.log_path)
        
        # Extract ground truth answers
        ground_truth = [re.findall(args.pattern, string)[-1] for string in datasets[dataset][args.field]['answer']][init_samples:max_samples]

        # Evaluate the model's performance
        evaluations = evaluating(beam_search_flag,log_outputs=log_outputs,
                                 ground_truth=ground_truth,
                                 pattern=args.pattern,
                                 num_paths=args.topk, 
                                 tokenizer=tokenizer,
                                 max_new_tokens=args.max_new_tokens)
        
        # Save evaluation plot if specified
        if args.save_plot:
            model_name = args.model.split("/")[-1]
            save_plot(args,evaluations, f'Exatch Match Evaluation - {dataset} - {model_name}', f'{model_name}/{dataset}-{model_name}-{args.dtype}')
        
        # Print evaluation results
        print_evaluations(beam_search_flag, evaluations, dataset)