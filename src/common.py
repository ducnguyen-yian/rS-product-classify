import json
from urllib.parse import urlparse
import os
import sys

from skimage import io
import pandas as pd

dirpath = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(dirpath, "..", "..", "data-science-core"))

from util import s3, file

def load_taxonomy(taxonomy_path="../files/taxonomy.json"):
  with open(taxonomy_path) as taxonomy_file:
    taxonomy = json.load(taxonomy_file)
  return taxonomy

def get_job_name_from_entry(entry):
  for key in entry:
    if "-metadata" in key:
      return entry[key]["job-name"]
  return None

def load_manifest(s3_manifest_path):
    local_path = "temp.manifest"
    job_name = None
    s3.get_with_s3_path(s3_manifest_path, local_path)
    products = []
    with open(local_path, "r") as json_file:
        for line in json_file:
            product = json.loads(line)
            products.append(product)
            if job_name is None:
              job_name = get_job_name_from_entry(product)
    file.remove_if_exists(local_path)
    return { "name": job_name, "products": products }

def trim_manifest(manifest, taxonomy, hierarchy, s3_image_base_path=None):
  name = manifest["name"]
  out = []
  for prod in manifest["products"]:
    if name not in prod:
      # in case there was an error with this product in labeling
      continue
    confidence = prod[name]['confidence']
    if confidence < 0.4:
       continue
    label_index = int(prod[name]['result']['product-category'])
    # note that this is a set
    worker_labels = {int(worker['label']) for worker in prod[name]['workers']}
    category = taxonomy.get_from_label_index(label_index, worker_labels)
    if category is None:
      # sorry label, not included in the pruning :(
      continue
    if not category[hierarchy.leaf_key]:
      continue
    x = {key: prod[key] for key in ['product_id', 'cached_image_url', 'target_image_url', 'title', 'description']}
    x['original_label'] = label_index
    x['confidence'] = confidence
    x['category_id'] = category["id"]
    x['category_name'] = category['name']
    x['one_hot_list'], _ = hierarchy.encode(x['category_id'])
    x['image'] = x['product_id'] + '.jpg'
    if s3_image_base_path is not None:
      x['target_image_url'] = s3_image_base_path + "/" + x['image']
    for idx in range(hierarchy.num_leaves):
      x[str(idx)] = int(x['one_hot_list'][idx])
    out.append(x)
  return out

def add_s3_image_rendering_to_dataframes(image_url_column="target_image_url"):
  def render_image(self, idx):
    s3_image_path = self[image_url_column][idx]
    local_image_path = str(idx) + ".jpg"
    s3.get_with_s3_path(s3_image_path, local_image_path)
    io.imshow(local_image_path)
    os.remove(local_image_path)
  pd.DataFrame.render_image = render_image
  