import torch
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import get_dataset
from utils import write_to_csv, get_csv, calculate_metrics
from transformers import ViltProcessor, ViltForQuestionAnswering

test_set_length, total_iterations, test_loader = get_dataset(config.JSON_FILE)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(config.DEVICE)

# Initialize a dictionary to hold all data
data_collection = {"image_path": [], "question": [], "real_answer": [], "formatted_pred": [], "prediction": []}

# Get the maximum sequence length for the model
max_length = processor.model_max_length if hasattr(processor, 'model_max_length') else 16

def check_device(tensors, device):
    for name, tensor in tensors.items():
        if tensor.device != torch.device(device):
            print(f"Tensor '{name}' is on device '{tensor.device}', expected '{device}'")

def preprocess_images(images):
    # Convert images to torch tensors if not already
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
    # Normalize images to the range [0, 1] if they aren't already
    if images.max() > 1.0:
        images = images / 255.0
    return images

with torch.no_grad():
    for batch in tqdm(test_loader, total=total_iterations):
        questions, imgs, answers, image_paths = batch['question'], batch['image'], batch['answer'], batch['image_path']

        # Preprocess images
        imgs = preprocess_images(imgs)

        # Create processor inputs without using do_rescale argument
        inputs = processor(images=imgs, text=questions, padding=True, return_tensors="pt").to(config.DEVICE)
        
        # Move inputs to the correct device
        inputs = {key: value.to(config.DEVICE) for key, value in inputs.items()}
        
        # Check devices of all input tensors
        check_device(inputs, config.DEVICE)

        # Check the length of input_ids
        input_ids_length = inputs["input_ids"].shape[-1]
        if input_ids_length > max_length:
            print(f"Warning: input_ids length {input_ids_length} exceeds max_length {max_length}. Truncating manually.")
            inputs = {key: value[:, :max_length] if key in ["input_ids", "attention_mask", "token_type_ids"] else value for key, value in inputs.items()}

        outputs = model(**inputs)

        # Debugging: print the type and shape of outputs
        print(f"Outputs type: {type(outputs)}, shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")

        # Process logits to obtain predictions
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            # Convert logits to predictions
            predictions = logits.argmax(dim=-1)
        else:
            predictions = outputs.argmax(dim=-1)
        
        # Decode predictions to text
        decoded_predictions = processor.batch_decode(predictions, skip_special_tokens=True)

        formatted_predictions = [pred if pred.isdigit() else config.NUM_DICT.get(pred, "-1") for pred in decoded_predictions]

        data_collection['image_path'].extend(image_paths)
        data_collection['question'].extend(questions)
        data_collection['real_answer'].extend(answers)
        data_collection['formatted_pred'].extend(formatted_predictions)
        data_collection['prediction'].extend(decoded_predictions)

results_df = pd.DataFrame(data_collection)
results_df['real_answer'] = results_df['real_answer'].astype(int)
results_df['formatted_pred'] = results_df['formatted_pred'].astype(int)
write_to_csv(results_df, "results")
bin_df = calculate_metrics(results_df)
write_to_csv(bin_df, "bin")
