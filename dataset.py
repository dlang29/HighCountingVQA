from torch import DeserializationStorageContext
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import json
import os
import config

def get_dataset(json_file, processor=None):
  # already preprocess to a standardized format
  transform = transforms.Compose([
      transforms.Resize(config.IMG_SIZE),
      transforms.ToTensor()
  ])
  # for evaluation return the dataloader, for training only the dataset
  if processor is None:
    test_set = HighCountVQADataset(data_root=config.DATA_ROOT,
                                  json_file=json_file,
                                  transform=transform)
    test_loader = DataLoader(test_set,
                          batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
  else:
    test_set = HFHighCountVQADataset(data_root=config.DATA_ROOT, json_file=json_file, processor=processor, transform=transform)

  
  test_set_length = len(test_set)
  total_iterations = int(test_set_length / config.BATCH_SIZE)
  
  if processor is None:
    return test_set_length, total_iterations, test_loader
  else:
    return test_set_length, total_iterations, test_set


class HighCountVQADataset(Dataset):
  """
  Standard dataset class used for evaluation.
  Returns raw inputs. 
  """
  def __init__(self, data_root, json_file, transform=transforms.ToTensor(), mode='RGB'):
    with open(json_file, 'r') as file:
      self.data_points = json.load(file)

    self.mode = mode
    self.data_root = data_root
    self.transform = transform
    # map image paths based on keywords inside the tallyqa.json file
    self.source_map = {'generate': 'coco', 'imported_vqa': 'coco',
                       'tdiuc_templates': 'coco', 'amt': 'visual_genome',
                       'imported_genome': 'visual_genome'}

  def __len__(self):
    return len(self.data_points)

  def __getitem__(self, idx):
    data_point = self.data_points[idx]
    img_path = os.path.join(self.data_root, self.source_map[data_point['data_source']], data_point['image'])

    img = self.transform(Image.open(img_path).convert(self.mode))

    answer = data_point['answer']
    question = data_point['question']

    return {'question': question, 'image': img, 'answer': answer, 'image_path': data_point['data_source']}


class HFHighCountVQADataset(HighCountVQADataset):
  """
  Dataset class for training.
  Already handles input processing and formatting for batching.
  """
  def __init__(self, data_root, json_file, processor, transform=transforms.ToTensor(), mode='RGB'):
    super().__init__(data_root, json_file, transform, mode)
    self.processor = processor
  
  def __getitem__(self, idx):
    data_point = self.data_points[idx]
    img_path = os.path.join(self.data_root, self.source_map[data_point['data_source']], data_point['image'])
    img = self.transform(Image.open(img_path).convert(self.mode))

    # different processing procedures for the models
    if "pali" in config.MODEL_ID:
      encodings = self.processor(images=img, text=data_point['question'],suffix=str(data_point['answer']),  padding="max_length", max_length=config.MAX_LENGTH, truncation=True, return_tensors="pt")
    else:
      labels = self.processor.tokenizer.encode(str(data_point['answer']), padding="max_length", max_length=8, truncation=True, return_tensors="pt")

      # NOTE: set label padding positions to -100 in order to be ignored for the loss computation (throws error)
      #labels[labels == self.processor.tokenizer.pad_token_id] = -100
    
      encodings["labels"] = labels

    # remove batch dimension
    for k,v in encodings.items():  encodings[k] = v.squeeze()

    return encodings
    
