# evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import ViltProcessor, ViltForQuestionAnswering
from tqdm import tqdm
import config
from vilt_dataset import VQADataset

def evaluate(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            labels = batch['labels']

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.2%}")

def main():
    # Initialize processor and model
    processor = ViltProcessor.from_pretrained(config.OUTPUT_DIR)
    model = ViltForQuestionAnswering.from_pretrained(config.OUTPUT_DIR).to(config.DEVICE)

    # Get evaluation dataset
    dataset = VQADataset(config.VAL_PATH, processor)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Evaluate the model
    evaluate(model, data_loader)

if __name__ == "__main__":
    main()
