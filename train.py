import torch
from tqdm import tqdm
from utils import *

def print_training_settings(args):
    print(f"\n{'='*50}")
    print("Starting Soft Prompt Training")
    print(f"{'='*50}")
    print("Settings:")
    print(f"- Learning Rate: {args.learning_rate}")
    print(f"- Number of Epochs: {args.num_epochs}")
    print(f"- Soft Prompt Length: {args.soft_prompt_length}")
    print(f"- Lambda: {args.lambda_}")
    print(f"- Mu: {args.mu}")
    print(f"{'='*50}\n")

def setup_optimizer(generator, args):
    return torch.optim.AdamW(
        generator.model.parameters(),
        lr=args.learning_rate
    )

def get_max_samples(dataset, args):
    return dataset[args.field].num_rows if args.max_samples == -1 else args.max_samples

def train_epoch(generator, cot_decoding, dataset, optimizer, args, max_samples):
    total_loss = 0
    path_correct_predictions = 0
    cot_correct_predictions = 0
    total_samples = 0
    cot_paths = None
    pbar = tqdm(range(max_samples), desc="Training")

    generator.model.train()
    for idx in pbar:
        prompt = dataset[args.field]['question'][idx]
        ground_truth = dataset[args.field]['answer'][idx]

        topk_tokens, outputs = generator.search_cots(prompt)
        cot_paths = cot_decoding.calculate_score(prompt, topk_tokens, outputs)

        loss = compute_soft_prompt_loss(cot_paths=cot_paths, ground_truth=ground_truth, args=args)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        for path in cot_paths.paths:
            if path.answer_span == ground_truth:
                path_correct_predictions += 1

        is_cot_correct, _ = compute_cot_accuracy(cot_paths, ground_truth)
        if is_cot_correct:
            cot_correct_predictions += 1

        total_samples = idx + 1
        path_acc = path_correct_predictions / (total_samples * len(cot_paths.paths))
        cot_acc = cot_correct_predictions / total_samples
        current_loss = total_loss / total_samples

        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'path_acc': f'{path_acc:.4f}', 'cot_acc': f'{cot_acc:.4f}'})

    return total_loss, path_correct_predictions, cot_correct_predictions, cot_paths

def train_soft_prompt(generator, cot_decoding, dataset, args):
    print_training_settings(args)
    optimizer = setup_optimizer(generator, args)
    max_samples = get_max_samples(dataset, args)
    print(f"Training on {max_samples} samples")

    best_val_score = 0
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 30)

        total_loss, path_correct_predictions, cot_correct_predictions, cot_paths = train_epoch(
            generator, cot_decoding, dataset, optimizer, args, max_samples
        )

        avg_loss = total_loss / max_samples
        path_accuracy = path_correct_predictions / (max_samples * len(cot_paths.paths))
        cot_accuracy = cot_correct_predictions / max_samples

        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Path-level Accuracy: {path_accuracy:.4f}')
        print(f'CoT Accuracy: {cot_accuracy:.4f}')

        if cot_accuracy > best_val_score and args.prompt_save_path:
            best_val_score = cot_accuracy
            save_soft_prompt(args, generator.model, args.prompt_save_path)
            print(f'Saved new best model with CoT accuracy: {cot_accuracy:.4f}')

    print(f"\n{'='*50}")
    print("Training Completed!")
    print(f"Final Best CoT Accuracy: {best_val_score:.4f}")
    print(f"Model saved to: {args.prompt_save_path}")
    print(f"{'='*50}\n") 