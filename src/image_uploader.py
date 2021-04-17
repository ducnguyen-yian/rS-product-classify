import json
import os
import sys
import cv2
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed

dirpath = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dirpath, "..", "..", "data-science-core"))

from util import s3, network, file

def bucket_path(s3_path):
    split = s3_path.split("/")
    return (split[2], "/".join(split[3:]))

def upload_to_s3(temp_image, s3_path):
    (bucket, path) = bucket_path(s3_path)
    s3.set_with_object_name(bucket, path, temp_image)
    file.remove_if_exists(temp_image)

def download_image(image_url, temp_image):
    network.download_image(image_url, temp_image)

def verify_image(image_path):
    try:
        image = cv2.imread(image_path)
        shape = np.shape(image)
        if len(shape) == 0 or shape[0] != 256 or shape[1] != 256:
            # can't trust this because the product's not cached, just skip it
            return False
    except:
        return False
    return True

class ImageUploader:
  def __init__(self, manifest):
    self.manifest = manifest
  
  def upload_all(self):
    self.count = 0
    self.executor = ThreadPoolExecutor(max_workers=30)
    for entry in self.manifest:
      f = self.executor.submit(self.upload_image, entry)
      f.add_done_callback(self.handle_future)
  
  def upload_image(self, entry):
    temp_image = entry['product_id']
    download_image(entry['cached_image_url'], temp_image)
    if verify_image(temp_image):
      upload_to_s3(temp_image, entry['target_image_url'])
  
  def handle_future(self, future):
    try:
      res = future.result()
      if (self.count % 100) == 0:
        print(self.count)
    except:
      pass
    self.count += 1
    if len(self.manifest) - self.count <= 0:
      print("Complete.")