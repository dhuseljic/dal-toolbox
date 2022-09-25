import os
import numpy as np
import torch

from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

def download_and_unzip(URL, root_dir):
  cwd = os.getcwd()
  os.system("wget "+"-P"+root_dir+" "+URL)
  os.chdir(root_dir)
  os.system("unzip tiny-imagenet-200.zip")
  os.system("rm tiny-imagenet-200.zip")
  os.chdir(cwd)

class TinyImageNet(Dataset):
  def __init__(self, root, split, transform=None, target_transform=None):
    super().__init__()
    self.split = split
    self.transform = transform
    self.target_transform = target_transform
    self.wnids_path = os.path.join(root, "tiny-imagenet-200", 'wnids.txt')
    self.words_path = os.path.join(root, "tiny-imagenet-200", 'words.txt')

    if not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root)

    self.nid_to_ints = {}
    self.nid_to_words = {}
    self.init_label_names()

    self.images = []
    self.labels = []

    if split == 'train':
      path = os.path.join(root, "tiny-imagenet-200", 'train')
      self.load_train_data(path)
    elif split == 'test':
      path = os.path.join(root, "tiny-imagenet-200", 'val')
      self.load_val_data(path)
    else:
      assert True, "This split is not available!"


  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image, target = self.images[idx], self.labels[idx]

    image = Image.fromarray(image)

    if self.transform is not None:
      image = self.transform(image)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return image, target

  def init_label_names(self):
    with open(self.wnids_path, 'r') as idf:
      for i, nid in enumerate(idf):
        nid = nid.strip()
        self.nid_to_ints[nid] = i
    self.nid_to_words = defaultdict(list)
    with open(self.words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

  def load_train_data(self, path):
    nids = os.listdir(path)
    for nid in nids:
      anno_path = os.path.join(path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(path, nid, 'images')
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, _, _, _, _ = line.split()
          fname = os.path.join(imgs_path, fname)
          image = Image.open(fname)
          array = np.asarray(image)
          # Workaround for Greyscale pictures
          if array.size == 4096:
            array = array.reshape(1, 64, 64)
            array = np.repeat(array,3,axis=0)
          array = array.reshape(3,64,64).transpose(1,2,0)
          self.images.append(array)
          self.labels.append(self.nid_to_ints[nid])

  def load_val_data(self, path):
    with open(os.path.join(path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, _, _, _, _ = line.split()
        fname = os.path.join(path, 'images', fname)
        image = Image.open(fname)
        array = np.asarray(image)
        if array.size == 4096:
            array = array.reshape(1, 64, 64)
            array = np.repeat(array,3,axis=0)
        array = array.reshape(3,64,64).transpose(1,2,0)
        self.images.append(array)
        self.labels.append(self.nid_to_ints[nid])
