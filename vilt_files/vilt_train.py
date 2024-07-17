# train.py

import torch
from transformers import ViltProcessor, ViltForQuestionAnswering, AdamW, get_scheduler
from tqdm import tqdm
import config
from vilt_dataset import get_dataset

def train(model, data_loader, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(config.DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            tqdm.write(f"Average Loss: {avg_loss:.4f}")

    return model

def main():
    # Initialize processor and model
    processor = ViltProcessor.from_pretrained(config.MODEL_ID)
    model = ViltForQuestionAnswering.from_pretrained(config.MODEL_ID).to(config.DEVICE)

    # Get dataset
    dataset_length, total_iterations, data_loader = get_dataset(config.TRAIN_PATH, processor)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LR)
    scheduler = get_scheduler(
        config.LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=len(data_loader) * config.EPOCHS
    )

    # Start training
    model = train(model, data_loader, optimizer, scheduler, config.EPOCHS)

    # Save model
    model.save_pretrained(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
