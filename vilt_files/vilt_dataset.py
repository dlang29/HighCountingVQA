# dataset.py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import config

class VQADataset(Dataset):
    def __init__(self, json_file, processor=None):
        self.json_file = json_file
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor()
        ])

        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(config.DATA_ROOT, item['image'])
        print(f"Loading image from: {image_path}")  # Add this line for debugging
        image = self.transform(Image.open(image_path).convert('RGB'))

        if self.processor is not None:
            inputs = self.processor(images=image, text=item['question'], padding="max_length", max_length=config.MAX_LENGTH, truncation=True, return_tensors="pt")
            inputs['labels'] = torch.tensor(item['answer'], dtype=torch.long)
            return inputs
        else:
            return {
                'image': image,
                'question': item['question'],
                'answer': item['answer']
            }

def get_dataset(json_file, processor=None):
    dataset = VQADataset(json_file, processor)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    
    dataset_length = len(dataset)
    total_iterations = len(data_loader)
    
    return dataset_length, total_iterations, data_loader
