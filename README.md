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


####TOCheckout
https://github.com/OluwoleOyetoke/Computer_Vision_Using_TensorFlowLite

