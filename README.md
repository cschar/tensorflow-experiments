### note
install notes for tensorflow-gpu windows 10
https://github.com/tensorflow/tensorflow/issues/6698#issuecomment-366494831

### using conda (python --> python3)

Notes from sentdex object recognition tutorial:
https://pythonprogramming.net/custom-objects-tracking-tensorflow-object-detection-api-tutorial/

# find images on google, 100 - 500

(or use pre-labelled data)
https://pythonprogramming.net/static/downloads/machine-learning-data/object-detection-macaroni.zip

label images with 
https://github.com/tzutalin/labelImg

save in images folder.
seperate into train/test folders.


# 
https://github.com/datitran/raccoon_dataset
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9



# MAL trial ===

```

ffmpeg -i myfile.avi -r 1000 -f image2 image-%07d.png

python export_dir_to_xml.py --image_dir=~/mdev/mal/images/test/ 

python ~/mdev/tensorflow-experiments/generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --images_input=images/test

python ~/mdev/tensorflow-experiments/generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --images_input=images/train

### move .record data into object_detection/data
## setup .config files and directories

cp data/train.record ~/mdev/tensorflow-experiments/object_detection/data/mal_train.record
cp data/test.record ~/mdev/tensorflow-experiments/object_detection/data/mal_test.record



python train.py --logtostderr --train_dir=training_mal/ --pipeline_config_path=training_mal/ssd_mobilenet_v1_pets_mal.config

### optional visualize loss graphs
tensorboard --logdir='training_mal'


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training_mal/ssd_mobilenet_v1_pets_mal.config \
    --trained_checkpoint_prefix training_mal/model.ckpt-7359  \
    --output_directory graph_mal2



```

# DAISY trial ===
```

cd ~
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

mkdir images
mkdir images/train
mkdir images/test
mkdir data
# copy 10 into test
# copy rest into train

cd <this_repo>

python export_dir_to_xml.py --image_dir=~/flower_photos/daisy/images/train/

python export_dir_to_xml.py --image_dir=~/flower_photos/daisy/images/test/


# wants train/test/data directoies in local path

cd ~/flower_photos/daisy
mkdir data

python xml_to_csv.py --base_dir=~/flower_photos/daisy


# generated labels 
ls data


# Now use CSV labels to generate a tfrecord 
# (Tensorflow's recommended input data format for disk data)
# .record files will be consumed in transfer learning step

python C:\Users\codywin\mdev\tensorflow-experiments\generate_tfrecord.py  --csv_input=data/test_labels.csv  --output_path=data/test.record   --images_input=~\flower_photos\daisy\images\test\

python C:\Users\codywin\mdev\tensorflow-experiments\generate_tfrecord.py  --csv_input=data/train_labels.csv  --output_path=data/train.record  --images_input=~\flower_photos\daisy\images\train\



# copy from template
# copy model
# cp ssd_mobilenet_v1_coco_11_06_2017.tar.1.gz ssd_mobilenet_v1_coco_11_06_2017_daisy.tar.1.gz 
# tar xvf ssd_mobilenet_v1_coco_11_06_2017_daisy.tar.gz


# in data directory in object_detection folder..............

# setup training/ssd_mobilenet_v1_pets_daisy.config (copy from template)


cat > data/daisy-object-detection.pbtxt
item{
    id: 1
    name: 'daisy'
}

cp ~/flower_photos/daisy/data/test.record ./data/daisy_test.record
cp ~/flower_photos/daisy/data/train.record ./data/daisy_train.record

# make sure unzipped ssd_mobilenet_v1_coco_11_06_2017 folder is in object_detection
## this is referenced in .config file 
## if its ../object_detection, wont work!


python train.py --logtostderr --train_dir=training_daisy/ --pipeline_config_path=training_daisy/ssd_mobilenet_v1_pets_daisy.config


INFO:tensorflow:global step 1: loss = 13.5551 (7.474 sec/step)
INFO:tensorflow:global step 2: loss = 12.0193 (0.576 sec/step)
INFO:tensorflow:global step 3: loss = 10.0618 (0.611 sec/step)
INFO:tensorflow:global step 4: loss = 8.7930 (0.624 sec/step)
INFO:tensorflow:global step 7: loss = 6.7923 (0.622 sec/step)
INFO:tensorflow:global step 8: loss = 5.8830 (0.607 sec/step)
INFO:tensorflow:global step 9: loss = 5.1327 (0.613 sec/step)
INFO:tensorflow:global step 10: loss = 4.4913 (0.649 sec/step)
INFO:tensorflow:global step 18: loss = 1.9779 (0.638 sec/step)
INFO:tensorflow:global step 19: loss = 1.6576 (0.511 sec/step)
INFO:tensorflow:global step 20: loss = 1.6131 (0.632 sec/step)


INFO:tensorflow:Saving checkpoint to path training_daisy/model.ckpt
### at this point we get a model.ckpt-1121.meta file there

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training_daisy/ssd_mobilenet_v1_pets_daisy.config \
    --trained_checkpoint_prefix training_daisy/model.ckpt-1121  \
    --output_directory graph_daisy


jupyter notebook

# open up object_detection_tutorial_daisy to use the inference graphh
# located in graph_daisy

### DONE


```
# ===========



## make sure tensorflow/models is installed 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# install models/research from tensorflow 
```
git clone https://github.com/tensorflow/models
cd models/research 
pip install -e .
#or 
python setup.py install

cd slim
pip install -e . 
```


## Train Model:
### Generate model.ckpt-<step checkpoint> index and meta files:
```
cd object_detection
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

## CPU vs GPU while training
```
with cpu
INFO:tensorflow:global step 13: loss = 8.2746 (2.709 sec/step)
INFO:tensorflow:global step 14: loss = 6.5976 (2.698 sec/step)
INFO:tensorflow:global step 15: loss = 7.1465 (2.721 sec/step)
INFO:tensorflow:global step 16: loss = 7.0370 (2.741 sec/step)
INFO:tensorflow:global step 17: loss = 6.6211 (2.880 sec/step)
INFO:tensorflow:global step 18: loss = 6.1563 (2.789 sec/step)
INFO:tensorflow:global step 19: loss = 7.5296 (2.736 sec/step)

with gpu (source activate tensorflow-gpu)
INFO:tensorflow:global step 10: loss = 7.9591 (0.549 sec/step)
INFO:tensorflow:global step 11: loss = 6.9812 (0.553 sec/step)
INFO:tensorflow:global step 12: loss = 7.4003 (0.580 sec/step)
INFO:tensorflow:global step 13: loss = 7.6535 (0.548 sec/step)
INFO:tensorflow:global step 14: loss = 7.5797 (0.546 sec/step)
INFO:tensorflow:global step 15: loss = 7.8302 (0.537 sec/step)
INFO:tensorflow:global step 16: loss = 7.3381 (0.557 sec/step)
INFO:tensorflow:global step 17: loss = 5.9450 (0.619 sec/step)
INFO:tensorflow:global step 18: loss = 5.7952 (0.557 sec/step)
INFO:tensorflow:global step 19: loss = 5.8427 (0.537 sec/step)

```


Then  when loss < 1.

## exporting inference graph from training .ckpt files


```
# --trained_checkpoint_prefix corresponds to whatever number you have reached in the training folder

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-16333  \
    --output_directory mac_n_cheese_graph
```
## exporting conda env
 conda env export -n <name of env> -f myenv.yml

####TOCheckout
https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite


## Great intro article
https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
