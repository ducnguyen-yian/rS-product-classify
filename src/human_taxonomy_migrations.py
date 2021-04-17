import copy
import csv
import json
from src.hierarchy import *
from src.common import *
import pandas as pd
import boto3
from botocore.errorfactory import ClientError

# load in manifest -- call function for this
# load in labeling taxonomy , create Hierarchy to get taxonomy in training form
# if this is an old labeling job, use the mapping to go between labeling and new training taxonomy *double hop 
# pass in manifest, returns training dataframe 

def create_dataset_from_manifest(manifest_s3_path, taxonomy_file_path, mapping_df_file_path, label_to_training_category_csv_path, labeling_job_name, human_group_taxonomy_file_path):
    #manifest s3 url, mapping dataframe, key of labeling job -- grabs correct json from each product
    manifest = load_manifest(manifest_s3_path)
    
    label_to_training_mapping = pd.read_csv(label_to_training_category_csv_path)
    labeling_taxonomy = load_taxonomy(taxonomy_file_path)
    hierarchy = Hierarchy(labeling_taxonomy[:], children_key="subcategories")

    keys_to_keep = ['product_id', 'source_image_url',
       'target_image_url', 'source-ref', 'title', 'description',
         labeling_job_name]
    
#     products = [{key:p[key] for key in keys_to_keep} for p in manifest['products']]
    products = [{key:p[key] for key in keys_to_keep} for p in manifest['products'] if labeling_job_name in p.keys()]
    
    for product in products:
        for key in product[labeling_job_name].keys():
            product[key]=product[labeling_job_name][key]
        del product[labeling_job_name]
    
    products_df = pd.json_normalize(products)
    products_df['result.product-category'] = products_df['result.product-category'].astype(int)
        
    if mapping_df_file_path is not None: 
        mapping_df = pd.read_json(mapping_df_file_path)
        mapping_df = mapping_df[['id','human_category','remap_id']]
        
        products_df = pd.merge(products_df, mapping_df, how='left', left_on='result.product-category',right_on='id') #labeling #1 to labeling id #2

        products_df['result.product-category'] =  products_df['remap_id']
        products_df['result.human-group'] =  products_df['human_category']
        
        products_df = products_df.drop(['remap_id', 'id','human_category'], axis=1)
    
    products_df=products_df[products_df['result.product-category']>=0]
    products_df['category_id']=products_df['result.product-category'].apply(hierarchy.new_id) #labeling id #2 to training id
    
    label_to_final_cat = {}
    label_to_final_cat_name = {}
    h_tax = hierarchy.taxonomy
    for dic in h_tax:
        label_category_id = dic['id']
        training_category_id = int(label_to_training_mapping[label_to_training_mapping['label_category_id']==label_category_id]['training_category_id'].values[0])
        training_category_name = str(label_to_training_mapping[label_to_training_mapping['label_category_id']==label_category_id]['training_category_name'].values[0])
        dic['id'] = training_category_id
        dic['name'] = training_category_name
        label_to_final_cat[label_category_id] = training_category_id
        label_to_final_cat_name[label_category_id] = training_category_name

        if 'parent' in dic.keys():
            label_parent_category_id = dic['parent']
            training_parent_category_id = int(label_to_training_mapping[label_to_training_mapping['label_category_id']==label_parent_category_id]['training_category_id'].values[0])
            dic['parent'] = training_parent_category_id

        new_subcategories_list = []
        for subcat in dic['subcategories']:
            new_subcat = int(label_to_training_mapping[label_to_training_mapping['label_category_id']==subcat]['training_category_id'].values[0])
            if new_subcat!=training_category_id:
                new_subcategories_list.append(new_subcat)
        dic['subcategories'] = list(set(new_subcategories_list))  

        if dic['subcategories']==[]:
            dic['is_leaf']=True
        else:
            dic['is_leaf']=False
    h_tax = [i for n, i in enumerate(h_tax) if i not in h_tax[n + 1:]]
    h_tax = [i for i in h_tax if (i['name']=='Main') or (i['parent']!=i['id'])]  
    hierarchy.taxonomy = sorted(h_tax, key = lambda i: i['id'])
    
    products_df['category_name']= products_df['category_id'].apply(lambda x: label_to_final_cat_name[x])
    products_df['category_id']= products_df['category_id'].apply(lambda x: label_to_final_cat[x])
    
    #remove non-leaf nodes and remove low confidence records. category_id>=0
    products_df=products_df[products_df['category_id']<hierarchy.num_leaves]
    products_df=products_df[products_df['confidence']>0.5]
    
    products_df['one_hot_list']=products_df['category_id'].apply(hierarchy.encode)
    products_df['one_hot_list']=products_df['one_hot_list'].apply(lambda x: x[0])
    
    #products_df['category_name'] = products_df['category_id'].apply(lambda x: hierarchy.taxonomy[x]['name'])
    
    products_df = products_df[['product_id', 'target_image_url', 'title', 'description',
       'confidence', 'result.product-category', 'result.human-group', 'category_id', 'category_name', 'one_hot_list']]
    products_df = products_df.rename(columns={"result.product-category":"original_label",'result.human-group':'human_group'})
    
    products_df['human_group'] = products_df['human_group'].str.replace('N/A','none')
    products_df['human_group'] = products_df['human_group'].str.capitalize()
    
    human_group_taxonomy = pd.read_json(human_group_taxonomy_file_path)

    products_df = pd.merge(products_df, human_group_taxonomy, how='left', left_on='human_group', right_on='human_group_name')
    products_df = products_df.drop('human_group_name', axis=1)
    products_df = products_df.rename(columns={'id':'human_group_id'})
    
    #add one-hot encoding for human-group to dataframe
    products_df['human_group_one_hot_list'] = products_df['human_group_id'].apply(encode_human_group)
    
    products_df['image'] = products_df['target_image_url'].apply(lambda x: "/".join(x.split("/")[6:]))

    return products_df.drop_duplicates(subset=['product_id'])

    
def encode_human_group(s):
    zero_list = np.zeros(5)
    zero_list[s] = 1
    return zero_list


def final_dataset_updates(new_manifest, s3_bucket, s3_prefix, min_product_count):
    '''takes in dataset after run through create_dataset_from_manifest() function'''
    new_manifest['target_image_url']='s3://'+s3_bucket+'/'+s3_prefix+'/'+new_manifest['image']
    
    #convert to json
    new_manifest_json = new_manifest.to_json(orient='records')
    new_manifest_json = json.loads(new_manifest_json)
    
    s3 = boto3.resource('s3')

    cache = {}
    for product in new_manifest_json:
        cache[product['target_image_url']] = product

    final_manifest = []
    bucket = s3.Bucket(s3_bucket)

    #check in image is in bucket
    for image in bucket.objects.filter(Prefix=s3_prefix):
        key = 's3://'+s3_bucket+'/' + image.key
        if key in cache:
            final_manifest.append(cache[key])
    
    final_manifest_df = pd.DataFrame(final_manifest)
    
    #remove categories with less than 100 images 
    category_counts = pd.DataFrame(final_manifest_df.groupby('category_id').count()['product_id']).reset_index()
    categories_with_enough_products = list(category_counts[category_counts['product_id']>=min_product_count].category_id)
    final_manifest_df['enough_products_flag'] = final_manifest_df['category_id'].isin(categories_with_enough_products)
    final_manifest_df = final_manifest_df[final_manifest_df['enough_products_flag']==True]
    final_manifest_df = final_manifest_df.drop(columns='enough_products_flag',axis=1)
    
    return final_manifest_df

    
    

# def original_output_manifest_to_human_category(manifest,
#                                                taxonomy,
#                                                mapping_df,
#                                                s3_image_base_path=None):
#   name = manifest["name"]
#   out = []
#   id_map = mapping_df["remap_id"]
#   for prod in manifest["products"]:
#     if name not in prod:
#       # in case there was an error with this product in labeling
#       continue
#     confidence = prod[name]['confidence']
#     if confidence < 0.4:
#        continue
#     label_index = int(prod[name]['result']['product-category'])
#     # map from old category to new category
#     category = taxonomy[id_map[label_index]]
#     if category is None:
#       # sorry label, not included in the pruning :(
#       continue
# #     if not category[hierarchy.leaf_key]:
# #       continue
#     x = {key: prod[key] for key in ['product_id', 'cached_image_url', 'target_image_url', 'title', 'description']}
#     x['original_label'] = label_index
#     x['confidence'] = confidence
#     x['category_id'] = category["id"]
#     x['category_name'] = category['name']
# #     x['one_hot_list'], _ = hierarchy.encode(x['category_id'])
#     x['image'] = x['product_id'] + '.jpg'
#     if s3_image_base_path is not None:
#       x['target_image_url'] = s3_image_base_path + "/" + x['image']
# #     for idx in range(hierarchy.num_leaves):
# #       x[str(idx)] = int(x['one_hot_list'][idx])
#     out.append(x)
#   return out
  

# def trim_manifest(manifest, taxonomy, hierarchy, s3_image_base_path=None):
#   name = manifest["name"]
#   out = []
#   for prod in manifest["products"]:
#     if name not in prod:
#       # in case there was an error with this product in labeling
#       continue
#     confidence = prod[name]['confidence']
#     if confidence < 0.4:
#        continue
#     label_index = int(prod[name]['result']['product-category'])
#     # note that this is a set
#     worker_labels = {int(worker['label']) for worker in prod[name]['workers']}
#     category = taxonomy.get_from_label_index(label_index, worker_labels)
#     if category is None:
#       # sorry label, not included in the pruning :(
#       continue
#     if not category[hierarchy.leaf_key]:
#       continue
#     x = {key: prod[key] for key in ['product_id', 'cached_image_url', 'target_image_url', 'title', 'description']}
#     x['original_label'] = label_index
#     x['confidence'] = confidence
#     x['category_id'] = category["id"]
#     x['category_name'] = category['name']
#     x['one_hot_list'], _ = hierarchy.encode(x['category_id'])
#     x['image'] = x['product_id'] + '.jpg'
#     if s3_image_base_path is not None:
#       x['target_image_url'] = s3_image_base_path + "/" + x['image']
#     for idx in range(hierarchy.num_leaves):
#       x[str(idx)] = int(x['one_hot_list'][idx])
#     out.append(x)
#   return out