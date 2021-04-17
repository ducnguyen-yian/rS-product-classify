# data-science-product-image

This is the repository for the Product Insights Project at rewardStyle. This README will explain in detail all the relevant pieces in this repo for the project.

## Objective

The objective of this project is to build an ML model that can effectively categorize rS products. This information is something that can be deployed across the board to many of the business units at rewardStyle.

## The Pipeline

### The Dataset

Our current dataset is a selection of around 50k rS product images from a selection of ~400 categories from Google's product taxonomy.

### Labeling

What are we using as the "ground truth" for our model? Each of the images we use to train our model has been labeled by a set of 3 independently selected manual labelers from Amazon Mechanical Turk. At one point, we tried to use rS employees as manual labelers, but limitations with Amazon Ground Truth made this quite difficult. In addition, the images which remain in our dataset are those on which the labelers achieved consenus, or where at least two of the three labelers selected agreed on a category for a particular image.

### Preprocessing

It is important to note that in order to increase model performance and make training easier, we made modifications to the original product taxonomy of ~400 categories and reduced it down to 73 categories with 57 leaves by combining specific redundant categories into more general categories and eliminating barely represented categories (categories containing 100 images or less). Our product taxonomy is hierarchical, with the 57 leaf nodes being the product categories the model is trained to predict.

To avoid a class imbalance issue with accuracy, we must first balance our trainin dataset so that each class is evenly represented. We accomplish this by setting a max_count value for each class. Categories that are over-represented will be randomly sampled down to the max_count value. Categories that are under-represented will have their images duplicated and put through image augmentation, while we use SMOTE to generate synthetic text inputs for synthetic products that we create during training.

There are two inputs to our model: the product image and a vectorized version of the product title and description. It is necessary for the product images used to be of the size (256, 256, 3), as this is what the model expects. 

For the text piece of the model, we first take the title and description of each product and run it through a count vectorizer, which builds a vocabulary of words by keeping track of the number of times it sees a word. The vectorizer transforms each piece of product text it sees into a vector of 0s and 1s where each 1 represents the occurence of a word. The model expects its text input to be in this form.

### Model Definition

For training, we use a multi-input model implemented in Keras. 

For the image piece of the model, we use ResNet50 as the base model archictecture. We remove the top layers to accomodate the size of our product images, and we add a Dense layer at the end of our image model. This layer has 1024 nodes.

Assuming one is already passing in a vectorized text input, the text piece of the model takes in this transformed vector and then runs it through two Dense layers of 256 nodes each.

At the end of the model, the image and text information is combined into one Dense output, which is then passed to an output layer with 57 nodes, representing each of the 57 leaf nodes left in the product taxonomy after pruning. From this, we get our final product category prediction.

### Training

Our training process goes as follows:

1. Train for 10 epochs using categorical cross-entropy as the loss and accuracy and hierarchy win as the metrics
2. Fine-tune the training for 5 epochs using the hierarchical loss function and accuracy and hierarchy win as the metrics
3. Save the model to S3

The hierarchy loss function helps the model keeep track of the hierarchy within the product taxonomy, and the win metric gives the model credit for guessing a correct parent category even when it guesses the incorrect leaf node. We ran several training jobs via a hyperparameter tuning job using this process, trying to maximize the model accuracy. Examples of hyperparameters we tuned include:

1. Dropout rate
2. Batch size
3. Learning Rate
4. Rate of weight decay
5. Maximum amount of images in a category required for over/undersampling
6. Number of degrees to rotate our images during training
7. Amount at which to modify the brightness of our images during training
8. Amount of shear to apply to our images during training
9. Amount of zoom to apply to our images during training
10. Channel shifting
11. Number of layers to freeze in the original ResNet50 model 

### Results

The above training process produces a Top-1 accuracy of 91-92%. We also have a fallback metric which checks to see if the model's second most confidnet category prediction shares a parent with the true value. If it does, we count this as correct as well. This metric brings us slightly higher, to 92.8%. Many of the model's mistakes are between similar categories (ex. Tees vs Shortsleeve Tops, Shoes vs. Flats). These results are very promising. 

### Inference

Once we trained the model to a promising accuracy level, it became necessary to figure out the best way for the model to perform inferences on new data. To this end, we investigated SageMaker's Batch Transform funcitonality, but when we struggled to get it working, we created our own framework for large-scale inference instead. We used the built in functionality of Keras to make predictions on an already trained model.

Our inference framework has the following steps:

1. We download the model locally. As part of this, we load in the taxonomy used in training. Our model has a custom 'hierarchical loss' metric, and the taxonomy is required for this.

2. We re-create the vectorizer using the exact dataframe (vocabulary) that was used in this specific training run.

3. We format the new/unseen data that we want to predict. This includes the costly uploading of images to s3, and downloading them locally to our instance. With millions of products, this will be a big pain point.

4. We predict the categories for the new data using the predict_generator method on the loaded in model. As part of this, we load in a custom InferenceDataGenerator (code found in inference_data_generators.py file). We use the flow_from_dataframe method, which transforms the images and text in the same way done in training. Once we have the output from predict_generator, we grab the predicted categories and confidences and add these data elements to the original dataframe. The final step is adding the hierarchical categories from the clean_taxonomy_for_analysis() function. 

"The desired output of this model for data analysis is a dataset with rewardstyle product_id, title, description, confidence, and category names -- including all levels of the categorical hierarchy. An example can be seen below.

![Screen Shot 2021-03-18 at 12 35 12 PM](https://user-images.githubusercontent.com/71672837/112006891-88696e00-8afa-11eb-99c7-5ddba37c8b84.png)


## Next Steps

We want to be able to use SageMakers Batch Transformation -- https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipeline-batch.html -- to be able to use a SageMaker model and calculate predictions for an entire dataset.

We believe an inference.py file is required for this, with specific input_handler and output_handler functions -- to preprocess the new data and format the output data for analysis. https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker_batch_transform/tensorflow_cifar-10_with_inference_script/code/inference.py

Perhaps some of the code in the inference-testing.ipynb may be able to help in the creation of this code. This will be something looked at in Q2.

## 'config' Folder

This folder only contains the ['test_config.py'](config/test_config.py) file, a file that establishes relevant SageMaker constants for our work, such as the name of the S3 bucket where our dataset is contained, the extension of our manifests, the size of our manifests, and where our product data comes from.

## 'files' Folder

This folder contains all the files that are neither source files nor Jupyter notebooks that we regularly use. It contains various versions of the product taxonomy that we use as labels for our data in JSON format. The latest version of the taxonomy that we have been using is the ['label-taxonomy-final.json'](files/label-taxonomy-final.json) file. There are also several images in this folder called 'specify-job-details-xx.png' which explain the steps to set up an MTurk labeling job in image form. The 'tickets' subdirectory contains some dummy images from our dataset which we used to test out image augmentation options for training.

## 'notebooks' Folder

This folder contains all the relevant Jupyter notebooks wihich have performed various tasks for us throughout the course of this project. They are as follows:

['create-dataset-from-manifests.ipynb'](notebooks/create-dataset-from-manifests.ipynb) is a notebook that reads in the labeled manifests and transforms them into a dataset ready for training. This step includes taxonomy migrations due to the differences in taxonomy between labeling jobs.

['create-manifest-from-datalake.ipynb'](notebooks/create-manifest-from-datalake.ipynb) is a notebook that allows us to pull product data from the datalake into an input manifest. This involves remapping the product taxonomy, and this taxonomy is what the MTurk workers see during a labeling job.

['deploy-model.ipynb'](notebooks/deploy-model.ipynb) takes a trained model and deploys it to a SageMaker endpoint.

['hyperparameter-tuning.ipynb'](notebooks/hyperparameter-tuning.ipynb) is the notebook from which we launch our hyperparameter tuning jobs.

['inference-testing.ipynb'](notebooks/inference-testing.ipynb) is the notebook that downloads a trained model locally from S3, performs preprocessing on new data, and makes inferences on new, unseen product data using the trained model.

['removing-bad-images-from-manifest.ipynb'](notebooks/removing-bad-images-from-manifest.ipynb) is a notebook in which we check to see if there are images in an input manifest that do not exist in S3, and remve them if necessary.

['render-confusion-matrix.ipynb'](notebooks/render-confusion-matrix.ipynb) is a notebook that allows us to view the confusion matrix from a completed training job, which will show us where the model made mistakes.

['run-mock-training.ipynb'](notebooks/run-mock-training.ipynb) is a notebook where we run a mock training job in order to test out any changees we have made to our training scripts.

['train-model.ipynb'](notebooks/train-model.ipynb) is where we run our larger training jobs to train the model.

## 'src' Folder

This folder contains all the relevant Python source files wihich have performed various tasks for us throughout the course of this project. They are as follows:

['base_model_trainer.py'](src/base_model_trainer.py) contains the base class that we use for model training. All other model training scripts inherit from this class.

['class_imbalance.py'](src/class_imbalance.py) is a script that modifies an existing training dataframe by correcting class imbalance. That is, it randomly samples frrom overepresented classes and duplicates instances in underepresented classes. Thw values which determine whether a class is over or under-represented are user-defined.

['common.py'](src/common.py) contains helper methods which are repeatedly referenced by other scripts and notebooks.

['concatenated_model_trainer.py'](src/concatenated_model_traner.py) is a script containing a class that subclasses resnet_model_trainer. It loads in the training dataframe, balances it, defines the comibned image and text model, trains it, evaluate its performance, and then saves it to S3.

['data_generators.py'](src/data_generators.py) contains a subclass of Keras' ImageDataGenerator class, designed to return both images and text as model inputs in batches.

['hierarchy.py'](src/hierarchy.py) contains the hierarchy class, which incorporates the concept of a hierarchy into our product taxonomy.

['human_taxonomy_migrations.py'](src/human_taxonomy_migrations.py) is a script that takes an output manifest form a labeling job and turns it into a training dataframe that we can use for training. All the taxonomy migration neccessary is built in here as well.

['image_uploader.py'](src/image_uploader.py) is a script that can upload an individual image to S3.

['inference_data_generators.py'](src/inference_data_generators.py) contains a subclass of the data generators found in data_generators.py. It is meant to return the same things as our reular data generator without needing a label column.

['inference_testing.py'](src/inference_testing.py) is a script that contains our inference framework. Here, we can load in a trained model from S3, and perform all the steps needed to use it to make predictions on an entirely new dataset.

['iterative_model_trainer.py'](src/iterative_model_trainer.py) contains a subclass that represents an old attempt to load in a model that has been already trained and resume training it from its current state.

['losses.py'](src/losses.py) contains all the relevant loss functions that we use for model training.

['metrics.py'](src/metrics.py) contains all the relevant metrics that we use for model training. 

['resnet_model_trainer.py'](src/resnet_model_trainer.py) contains a class that is a subclass of base_model_trainer which defines the Resnet image model, trains it, evaluates its performance, and then saves the trained model to S3.

['taxonomy.py'](src/taxonomy.py) is a script that performs tasks on our product taxonomy. It mainly contains the Migration class, which allows us to migrate between two versions of our taxonomy.
