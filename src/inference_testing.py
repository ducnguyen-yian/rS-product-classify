import os
import logging
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from pathlib import Path
import bz2
import tarfile

import argparse
import sys
import numpy as np
import pandas as pd
import json
import datetime
import boto3
from pathlib import Path
from urllib.parse import urlparse

from src.base_model_trainer import BaseModelTrainer
from src.resnet_model_trainer import ResnetHierarchicalModel
from src.metrics import *
from src.losses import *
from src.hierarchy import *
from src.class_imbalance import *
from src.data_generators import *
from src.inference_data_generators import *
from src.taxonomy import *
from src.image_uploader import *

from sklearn.feature_extraction.text import CountVectorizer

from math import floor

from util import s3, file

def load_from_s3(s3_path, local_path):
    s3.get_with_s3_path(s3_path, local_path)
    
def download_model_locally(taxonomy_s3_path, s3_model_artifact):
    #download taxonomy locally 
    load_from_s3(taxonomy_s3_path, '../files/taxonomy.json')
    taxonomy = load_taxonomy('../files/taxonomy.json')
    H = Hierarchy(taxonomy, children_key='subcategories')
    
    #download model locally
    if not os.path.isfile('../files/training_job_model.tar.gz'):
        load_from_s3(s3_model_artifact,'../files/training_job_model.tar.gz')
    if not os.path.isdir('../files/model'):
        file.tar_gz_decompress('../files/training_job_model.tar.gz', '../files/model')
    
    model = tf.keras.models.load_model('../files/model/000000001', compile=False)
    
    hierarchy_metric = HierarchyMetric(H)
    fallback_metric=FallbackAccuracy(H, 0.5)
    metrics=[hierarchy_metric, fallback_metric]
    
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)
    
    return model
        
def create_vectorizer(combination_type, text_cols):
    comb_type=combination_type
    text_cols = text_cols.split(",")

    vectorizer=CountVectorizer()

    training_dataframe = pd.read_json('../files/model/training_dataframe.json')
    df = training_dataframe
    
    text_cols_var = text_cols + []
    all_text=''
    while len(text_cols_var) > 0:
        all_text += ' ' + df[text_cols_var.pop()]

    vectorizer.fit(all_text)

    encoding = vectorizer.transform([df[text_cols[0]].iloc[0]]).toarray()
    text_model_input_size = encoding.shape[1]

    if comb_type is CombinationType.CONCAT:
        # if this is the case, we need vectorizer_size * num_text_cols
        text_model_input_size *= len(text_cols)
    
    return vectorizer

# def download_s3_folder(s3_uri, local_dir=None):
#     """
#     Download the contents of a folder directory
#     Args:
#         s3_uri: the s3 uri to the top level of the files you wish to download
#         local_dir: a relative or absolute directory path in the local file system
#     """
#     s3 = boto3.resource("s3")
#     bucket = s3.Bucket(urlparse(s3_uri).hostname)
#     s3_path = urlparse(s3_uri).path.lstrip('/')
#     if local_dir is not None:
#         local_dir = Path(local_dir)
#     for obj in tqdm(bucket.objects.filter(Prefix=s3_path)):
#         target = obj.key if local_dir is None else local_dir / Path(obj.key).relative_to(s3_path)
#         target.parent.mkdir(parents=True, exist_ok=True)
#         if obj.key[-1] == '/':
#             continue
#         bucket.download_file(obj.key, str(target))
        
def format_new_data(data_to_predict_s3_path, data_to_predict_s3_file_name, local_json_file_name, upload_images_flag=False):
    '''read in new data from s3 and upload images to s3'''
    load_from_s3(data_to_predict_s3_path+data_to_predict_s3_file_name, '../files/'+local_json_file_name+".json")

    new_data_to_predict = pd.read_json('../files/'+local_json_file_name+".json")
    new_data_to_predict['source_image_url'] = new_data_to_predict.agg(lambda x: f"https://images.rewardstyle.com/img?v=1&width=256&height=256&crop&p={x['product_id']}", axis=1)
    new_data_to_predict['target_image_url'] = new_data_to_predict.agg(lambda x: "{}images/{}.jpg".format(data_to_predict_s3_path,x['product_id']), axis=1)
    new_data_to_predict = new_data_to_predict[['product_id', 'source_image_url', 'target_image_url', 'title', 'description']]

    new_data_to_predict['image'] = new_data_to_predict['product_id'].astype(str)+'.jpg'
    new_data_to_predict['image_path']='../files/new_data_to_predict/'+local_json_file_name+'/'+new_data_to_predict['image']
    new_data_to_predict['one_hot_list'] = pd.Series()

    new_data_to_predict_dict = new_data_to_predict.to_dict('records')
    if upload_images_flag==True:
        image_uploader = ImageUploader(new_data_to_predict_dict)
        image_uploader.upload_all(True)
        
    new_data_to_predict['title'] = new_data_to_predict['title'].fillna(' ')
    new_data_to_predict['description'] = new_data_to_predict['description'].fillna(' ')
    return new_data_to_predict.to_json('../files/'+local_json_file_name+".json")
        
            
def predict(vectorizer, local_json_file_name, combination_type, model, taxonomy_s3_path):
    dg = InferenceDataGenerator(vectorizer, preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    new_data_to_predict = pd.read_json('../files/'+local_json_file_name+".json")
    new_data_to_predict['file_exist'] = new_data_to_predict['image_path'].apply(lambda x: os.path.isfile(x))

    new_data_to_predict = new_data_to_predict[new_data_to_predict['file_exist']==True]
    
    new_data = dg.flow_from_dataframe(
    new_data_to_predict, text_cols=['title','description'], combination_type=combination_type, batch_size=32, target_size=(256, 256))
    
    prediction = model.predict_generator(new_data, verbose=1)
    
    confidence = np.amax(prediction, axis=1)
    
    predicted_category = np.argmax(prediction, axis=1) 
    
    new_data_to_predict['predicted_category'] = predicted_category
    new_data_to_predict['confidence'] = confidence
    new_data_to_predict.drop(['one_hot_list','file_exist'],axis=1)
    
    taxonomy_df = clean_taxonomy_for_analysis(taxonomy_s3_path)
    
    final_data = pd.merge(new_data_to_predict, taxonomy_df, how='left', left_on='predicted_category',right_on='id')
    final_data = final_data[['product_id', 'title', 'description','confidence', 'category1', 'category2',
       'category3','full_category']]
    
    return final_data
    
def clean_taxonomy_for_analysis(taxonomy_s3_path):
    load_from_s3(taxonomy_s3_path, 'taxonomy.json')
    taxonomy = load_taxonomy('taxonomy.json')
    H = Hierarchy(taxonomy, children_key='subcategories')
    
    taxonomy_df = pd.DataFrame(H.taxonomy)
    taxonomy_df['parent_name'] = taxonomy_df['parent'].apply(lambda x: str(taxonomy_df[taxonomy_df['id']==x]['name'].values)).apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace("'",''))
    taxonomy_df['grandparent_id'] = taxonomy_df['parent'].apply(lambda x: str(taxonomy_df[taxonomy_df['id']==x]['parent'].values)).apply(lambda x: x.replace('[','').replace(']','').replace('.',''))
    taxonomy_df['grandparent_name'] = taxonomy_df['grandparent_id'].apply(lambda x: str(taxonomy_df[taxonomy_df['id'].astype(str)==x]['name'].values)).apply(lambda x: x.replace('[','').replace(']','').replace("'",'').replace("'",''))

    taxonomy_df['category1'] = np.where(np.logical_and(taxonomy_df["grandparent_name"]!='',taxonomy_df["grandparent_name"]!='Main')
                                        , taxonomy_df["grandparent_name"] 
                                        , np.where(
                                            np.logical_and(taxonomy_df["parent_name"]!='',taxonomy_df["parent_name"]!='Main'), 
                                            taxonomy_df["parent_name"],
                                            taxonomy_df["name"]))
    taxonomy_df['category2'] = np.where(np.logical_or(taxonomy_df["grandparent_name"]=='',taxonomy_df["grandparent_name"]=='Main')
                                        , taxonomy_df["name"]
                                        , taxonomy_df["parent_name"])
    taxonomy_df['category3'] = np.where(taxonomy_df["category2"]!=taxonomy_df['name']
                                        , taxonomy_df["name"]
                                        , '')
    taxonomy_df['full_category'] = np.where(taxonomy_df['category3']!='',
                                           taxonomy_df['category1']+'->'+taxonomy_df['category2']+'->'+taxonomy_df['category3'],
                                           np.where(taxonomy_df['category2']!='',
                                                   taxonomy_df['category1']+'->'+taxonomy_df['category2'],
                                                   taxonomy_df['category1']))
    taxonomy_df = taxonomy_df[['id', 'category1', 'category2','category3','full_category']]
    
    return taxonomy_df