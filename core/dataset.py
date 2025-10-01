import os
import cv2
import numpy as np
import imageio
import random

from petrel_client.client import Client   # fallback to local/http
from torch.utils.data import Dataset

from core.annotation import Monst3RAnno, PexelsAnno
from core.utils import print_attributes

class PointmapDataset(Dataset):
    def __init__(self, datalist, s3_conf_path='~/petreloss.conf', debug=False, max_frames=None, random_shuffle=False, cache_dir='.cache/', skip_invalid=True):
        self.max_frames = max_frames
        self.list_with_caption = False
        if isinstance(datalist, str):
            self.items = []
            if 'vbench' in datalist:
                self.list_with_caption = True
                self.captions = []
                assert random_shuffle == False, "random shuffle is not supported for datalist with caption"
            with open(datalist, 'r') as f:
                for line in f.readlines():
                    if self.list_with_caption:
                        item, caption = line.split('|')
                        self.items.append(item)
                        self.captions.append(caption.strip())
                    else:
                        self.items.append(line.strip())
        else:
            if '|' in datalist[0]:
                self.list_with_caption = True
                self.items = []
                self.captions = []
                assert random_shuffle == False, "random shuffle is not supported for datalist with caption"
                for line in datalist:
                    item, caption = line.split('|')
                    self.items.append(item)
                    self.captions.append(caption)
            else:
                self.items = datalist
        if random_shuffle:
            random.shuffle(self.items)
        
        if s3_conf_path is not None:
            self.client = Client(s3_conf_path)
        else:
            self.client = None

        self.debug = debug
        self.cache_dir = cache_dir
        self.skip_invalid = skip_invalid

    
    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        while True:
            try:
                item = self._get_item(idx)
                break
            except Exception as e:
                print(e)
                if self.skip_invalid:
                    idx = (idx + 1) % self.__len__()
                else:
                    item = None
                    break
        return item

    def _get_item(self, idx):
        if self.list_with_caption:
            monst3r_annotation = Monst3RAnno(anno_dir=self.items[idx], client=self.client, max_frames=self.max_frames, cache_dir=self.cache_dir, caption=self.captions[idx])
        else:
            monst3r_annotation = Monst3RAnno(anno_dir=self.items[idx], client=self.client, max_frames=self.max_frames, cache_dir=self.cache_dir)
        if self.debug:
            print_attributes(monst3r_annotation)
        return monst3r_annotation
        
        