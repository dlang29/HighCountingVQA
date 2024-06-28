from torch import DeserializationStorageContext
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import json
import os


class HighCountVQADataset(Dataset):
  def __init__(self, data_root, json_file, transform=transforms.ToTensor(), mode='RGB'):
    with open(json_file, 'r') as file:
      self.data_points = json.load(file)

    self.mode = mode
    self.data_root = data_root
    self.transform = transform
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

    return {'question': question, 'image': img, 'answer': answer}