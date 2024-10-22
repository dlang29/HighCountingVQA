import torch
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import get_dataset
from utils import write_to_csv, get_csv, calculate_metrics
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoModel, PaliGemmaForConditionalGeneration, Blip2ForConditionalGeneration
from peft import get_peft_model, PeftModel

test_set_length, total_iterations, test_loader = get_dataset(config.JSON_FILE)
print(test_set_length, total_iterations)

# Blip specific loading
# processor instead of tokenizer -> contains BERT tokenizer + BLIP image processor
processor = AutoProcessor.from_pretrained(config.MODEL_ID, do_rescale=False, token = config.HF_TOKEN) #do_rescale false, because ToTensor() already does that

# differentiate between model types and pre-trained/fine-tuned
if config.TRAINED:
    if 'pali' in config.MODEL_ID:
        model = PaliGemmaForConditionalGeneration.from_pretrained(config.BEST_CHECKPOINT, token = config.HF_TOKEN, torch_dtype=torch.bfloat16).to(config.DEVICE)
        model = PeftModel.from_pretrained(model, config.BEST_CHECKPOINT)
    else:
        model = AutoModelForVisualQuestionAnswering.from_pretrained(config.BEST_CHECKPOINT, token = config.HF_TOKEN, torch_dtype=torch.bfloat16).to(config.DEVICE)
        
else:
    if 'pali' in config.MODEL_ID:
        model = PaliGemmaForConditionalGeneration.from_pretrained(config.MODEL_ID, token = config.HF_TOKEN, torch_dtype=torch.bfloat16).to(config.DEVICE)
    else:
         model = AutoModelForVisualQuestionAnswering.from_pretrained(config.MODEL_ID, token = config.HF_TOKEN, torch_dtype=torch.float16).to(config.DEVICE)


# Initialize a dictionary to hold all data
data_collection = {"image_path": []
,"question": [],"real_answer": [],"formatted_pred": [],"prediction": []}


with torch.no_grad():
    for batch in tqdm(test_loader, total=total_iterations):
        questions, imgs, answers, image_paths = batch['question'], batch['image'], batch['answer'], batch['image_path']

        inputs = processor(images=imgs, text=questions, padding=True, return_tensors="pt").to(config.DEVICE)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(**inputs, max_new_tokens=32)
        # dependent on the used model it may also output the input
        if config.SKIP_INPUT:
            outputs = outputs[:, input_len:]
        predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        formatted_predictions = [pred if pred.isdigit() else config.NUM_DICT.get(pred, "-1") for pred in predictions]

        data_collection['image_path'].extend(image_paths)
        data_collection['question'].extend(questions)
        data_collection['real_answer'].extend(answers)
        data_collection['formatted_pred'].extend(formatted_predictions)
        data_collection['prediction'].extend(predictions)

# convert accumulated data to a datadrame and save for later processing
results_df = pd.DataFrame(data_collection)
results_df['real_answer'] = results_df['real_answer'].astype(int)
results_df['formatted_pred'] = results_df['formatted_pred'].astype(int)

write_to_csv(results_df, "results")
