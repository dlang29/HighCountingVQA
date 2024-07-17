import torch
import transformers
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoModel, PaliGemmaForConditionalGeneration
from transformers import TrainingArguments, Trainer

from peft import get_peft_model, LoraConfig

from dataset import get_dataset

from tqdm import tqdm
import config
import os

# NOTE: not needed here -> if we want to use custom metrics during evaluation (instead of just the loss) we can use this template
# -> for generative models Seq2SeqTrainer with "predict_with_generate=True" is needed, because we need the generate function during evaluation and not just the current logits (of 1 token)
def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def main():
    # initialize all components
    processor = AutoProcessor.from_pretrained(config.MODEL_ID, do_rescale=False, token = config.HF_TOKEN)
    if "pali" in config.MODEL_ID:
        model = PaliGemmaForConditionalGeneration.from_pretrained(config.MODEL_ID, token = config.HF_TOKEN, torch_dtype=torch.bfloat16).to(config.DEVICE) # Training will happen automatically on device of the model
        # to big for full fine-tuning -> freezing image encoder still to big -> use Lora Fine-tuning
        lora_config = LoraConfig(
                        r=8, # this was used in an example (also often r=16, lora_alpha=16)
                        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM",
                    )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        model = AutoModelForVisualQuestionAnswering.from_pretrained(config.MODEL_ID, token = config.HF_TOKEN, torch_dtype=torch.bfloat16).to(config.DEVICE)

    _, _, train_ds = get_dataset(config.TRAIN_PATH, processor)
    _, _, val_ds = get_dataset(config.VAL_PATH, processor)
    _, _, test_ds = get_dataset(config.TEST_PATH, processor)

    # # metric function for evaluation => not needed now
    # acc_metric = evaluate.load("accuracy")
    # acc_compute_metrics = lambda eval_pred: compute_metrics(eval_pred, metric=acc_metric)

    # define all training arguments
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    no_bf_support = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE, # may use a smaller batchsize for training because of higher memory usage
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION, # "virtual" batchsize -> accumuluate gradients over multiple steps
        dataloader_num_workers=config.NUM_WORKERS,
        # bf16 or fp16 based on which is supported for the used GPU (half precision used here because of bigger models like Pali)
        bf16=not no_bf_support,
        fp16=no_bf_support,
        learning_rate=config.LR,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=5 # not more checkpoints should be saved
    )

    # define callbacks
    callbacks = []
    if config.EARLY_STOPPING:
        callbacks.append(transformers.EarlyStoppingCallback(early_stopping_patience=config.PATIENCE))
    
    # define trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks
    )

    transformers.logging.set_verbosity_info()
    trainer.train()

    # save best checkpoint again separately
    trainer.save_model(config.BEST_CHECKPOINT)

    

    # ToDo: add autoamtic call of evaluation script based on the testset after training? -> or rather manual evaluation.py call on test set


main()
