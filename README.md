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

```
# use the xml files generated by labelimg to get our csv files
# for train and test labels.
python xml_to_csv.py

#convert the csv files into .record files so tensorflow object detection  (transfer learning)
#api can use it
```


## make sure tensorflow/models is installed 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

# install models/research from tensorflow 
```
git clone https://github.com/tensorflow/models
cd models/research 
pip install -e .
#or 
python3 setup.py install

cd slim
pip install -e . 
```
