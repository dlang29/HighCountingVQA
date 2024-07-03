from torch import DeserializationStorageContext
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import json
import os
import config

def get_dataset():
  # declare all necessary components (here just for inference)
  transform = transforms.Compose([
      transforms.Resize(config.IMG_SIZE),
      transforms.ToTensor()
  ])
  test_set = HighCountVQADataset(data_root=config.DATA_ROOT,
                                json_file=config.JSON_FILE,
                                transform=transform)
  test_loader = DataLoader(test_set,
                          batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

  test_set_length = len(test_set)
  total_iterations = int(test_set_length / config.BATCH_SIZE)
  return test_set_length, total_iterations, test_loader


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

    return {'question': question, 'image': img, 'answer': answer, 'image_path': data_point['data_source']}