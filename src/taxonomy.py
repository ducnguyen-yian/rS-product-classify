import copy
import csv
import json
from src.common import load_taxonomy

class MigratedTaxonomy:
  def __init__(self):
    self.disburse_map = {}
    self.taxonomy = []
  
  def __getitem__(self, key):
    return self.taxonomy[key]
  
  def get_from_label_index(self, label_index, worker_labels=None):
    if label_index in self.remap:
      return self.taxonomy[self.remap[label_index]]
    if worker_labels is not None:
      if label_index in self.disburse_map:
        disburse_to = self.disburse_map[label_index]
        target = worker_labels & disburse_to
        if len(target) > 0:
          target_label, *_ = target
          if target_label in self.remap:
            return self.taxonomy[self.remap[target_label]]
    return None
    
  def add(self, category):
    self.taxonomy.append(category)
  
  def build_remap(self):
    self.remap = {}
    for cat in self.taxonomy:
      if "remap" in cat:
        for idx in cat["remap"]:
          self.remap[idx] = cat["id"]
        del cat["remap"]
  
  def add_disbursement(self, idx, disburse_to):
    self.disburse_map[idx] = set(disburse_to)
  
  def get_disbursed_category(self, label_index, worker_labels):
    category = self.get_from_label_index(label_index)
    if category is None:
      if label_index in self.disburse_map:
        disburse_to = self.disburse_map[label_index]
  
  def save(self, filepath):
    with open(filepath, 'w') as f:
      f.write(json.dumps(self.taxonomy))
        
  
  @property
  def tree(self):
    return self.taxonomy

class Migration:
  
  def __init__(self, label_taxonomy_file_path, migration_file_path):
    self.label_taxonomy = load_taxonomy(label_taxonomy_file_path)
    with open(migration_file_path) as csv_file:
      reader = csv.reader(csv_file)
      # drop header
      next(reader)
      self.migration_file = [row for row in reader]
  
  def run(self, id_col=1, rec_id_col=9, rename_col=10, disburse_col=11):
    migrated = MigratedTaxonomy()
    category_map = {}
    
    for row in self.migration_file:
      
      rec_id = row[rec_id_col]
      if len(rec_id) == 0:
        # this category has been excluded
        continue
      
      rec_id = int(rec_id)
      category_id = int(row[id_col])
      
      disburse_to = row[disburse_col]
      if len(disburse_to) > 0:
        # we only collect this into another category
        migrated.add_disbursement(category_id, [int(target) for target in disburse_to.split(",")])
        continue
      
      if rec_id not in category_map:
        category = copy.copy(self.label_taxonomy[rec_id])
        category["subcategories"] = [] # these are all leaf nodes now!
        category["remap"] = []
        category_map[rec_id] = category
      else:
        category = category_map[rec_id]
      
      rename = row[rename_col]
      if len(rename) > 0:
        category["name"] = rename
        
      category["remap"].append(category_id)
      
    new_tax = []
    for _, v in category_map.items():
      migrated.add(v)

    parents = {}
    for cat in migrated.taxonomy:
      node = cat
      while 'parent' in node:
        parent = copy.copy(self.label_taxonomy[node['parent']])
        if parent['id'] not in parents:
          migrated.add(parent)
          parents[parent['id']] = True
        node = parent

    node_map = {}
    for idx, node in enumerate(migrated.taxonomy):
      node_map[node["id"]] = idx
      node["id"] = idx

    for node in migrated.taxonomy:
      if "parent" in node:
        node["parent"] = node_map[node["parent"]]
      # we may have excluded some children nodes, so have to check if they're here
      node["subcategories"] = [node_map[sub] for sub in node["subcategories"] if sub in node_map]

    migrated.build_remap()
    return migrated
      
      
      
      