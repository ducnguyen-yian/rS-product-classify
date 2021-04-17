import pandas as pd
import numpy as np

from scipy import sparse
#from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

def balance_classes(df, max_count, vectorizer, num_leaves):
  df_new_title=SMOTE(df, max_count, vectorizer, 'title', num_leaves)
  df_new_description=SMOTE(df, max_count, vectorizer, 'description', num_leaves)
  
  df_new=df_new_title
  df_new.description=df_new_description.description
#   df_new['description_product_id'] = df_new_description.product_id
  
  oversampled_df=pd.DataFrame()
  
  undersampled_df=random_sampling(df, max_count)
  
  old_category_groups=df.groupby('category_id')
  new_category_groups=df_new.groupby('category_id')
  
  for category in df_new.category_id.unique():
    old_category_df=old_category_groups.get_group(category)
    new_category_df=new_category_groups.get_group(category)
    
    frames=[old_category_df, new_category_df]
    oversampled_category_df=pd.concat(frames, ignore_index=True)
    oversampled_df=pd.concat([oversampled_category_df, oversampled_df], ignore_index=True)
  
  result_df=pd.concat([undersampled_df, oversampled_df], ignore_index=True)
  
  return result_df

def random_sampling(df, max_count):
  df2=pd.DataFrame()
    
  category_groups=df.groupby('category_id')

  for category in df.category_id.unique():
    category_df=category_groups.get_group(category)
    
    if len(category_df)>max_count:
      sample_df=category_df.sample(max_count, random_state=0)
      frames=[sample_df, df2]
      df2=pd.concat(frames)
  return df2
  
def create_Ts(df, max_count):
    Ts=[]
    
    category_groups=df.groupby('category_id')
  
    for category in df.category_id.unique():
        category_df=category_groups.get_group(category)
        if len(category_df)<max_count:
            T=max_count-len(category_df)
            Ts.append(T)
  
    Ts=np.array(Ts)
  
    return Ts

def SMOTE(df, max_count, vectorizer, text_col, num_leaves):
    np.random.seed(0)
    #create number of instances to be generated per category
    Ts=create_Ts(df, max_count)
  
    category_ids=[]
  
    category_groups=df.groupby('category_id')

    for category in df.category_id.unique():
        category_df=category_groups.get_group(category)
    
        if len(category_df)<max_count:
            category_ids.append(category_df.category_id.values[0])
      
    result_df=pd.DataFrame(index=range(len(df)+sum(Ts)), columns=df.columns)
  
    df_new=pd.DataFrame(columns=df.columns)
  
    X=vectorizer.fit_transform(np.array(df[text_col]))
    y=df['one_hot_list']
  
    KNN=NearestNeighbors(n_neighbors=6)
    KNN.fit(X, y)
  
    X_new=[]

    y_new=np.zeros((sum(Ts), num_leaves))

    new_count=0
  
    for i in range(len(Ts)):
        '''loop through categories '''
        new_text = []
        category_id=category_ids[i]
        category_indices=np.where(df.category_id==category_id)
        category_indices=np.array(category_indices)
        category_indices=np.squeeze(category_indices, axis=0)
        category_indices_list = list(category_indices)
        duplicated_category_df = df.iloc[category_indices_list,:]
        randomly_sample_indices = []
    
        for j in range(Ts[i]):
            '''randomly sample enough products per category'''
            idx=np.random.randint(low=0, high=category_indices.shape[0])
            randomly_sample_indices.append(idx)
            '''find nearest neighbor for both title and description columns '''
            old=X[category_indices[idx]].toarray()
            nns=KNN.kneighbors(old, return_distance=False)
            nn_selection=np.random.randint(low=1, high=nns.shape[-1])
            nn_idx=nns[0][nn_selection]
            nn=X[nn_idx]
            diff=old-nn
            steps=np.random.uniform(size=old.shape[0])
            new=old+np.multiply(steps, diff)
            X_new.append(new)
            y_new[new_count][category_id]=1

            new_instance_array=np.squeeze(np.array(vectorizer.inverse_transform(X_new[new_count])))

            separator=' '

            new_instance_string=''

            if new_instance_array.size>1:
                new_instance=list(new_instance_array)
                new_instance_string=separator.join(new_instance)

            new_text.append(new_instance_string)
            new_count +=1

        randomly_sampled_duplicated_category_df = duplicated_category_df.copy().iloc[randomly_sample_indices,:]
        randomly_sampled_duplicated_category_df.loc[:,text_col] = new_text

        df_new = df_new.append(randomly_sampled_duplicated_category_df, ignore_index=True)

    return df_new

