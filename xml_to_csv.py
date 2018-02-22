import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
      '--base_dir',
      type=str,
      default='',
      help= 'where you should have following directories: [images/test, images/train, data]'
  )
FLAGS, unparsed = parser.parse_known_args()




def xml_to_csv(path):
    xml_list = []
    for idx, xml_file in enumerate(glob.glob(path + '/*.xml')):
        if(idx > 0 and idx % 100 == 0):
            print(idx, " xml files processed")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



if FLAGS.base_dir:
  BASE_DIR=os.path.expanduser(FLAGS.base_dir)
else:
  raise Exception("provide base_dir arg")


def main():
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

def main_daisy():
    print("Setting up in BASE_DIR {}".format(BASE_DIR))
    print("images/test, images/train ---> data/test_labels.csv, data/train_labels.csv")
    for directory in ['train','test']:
        image_path = os.path.join(BASE_DIR, 'images/{}'.format(directory))
        print("converting xml in path: ", image_path)
        xml_df = xml_to_csv(image_path)
        
        xml_df.to_csv(
            os.path.join(BASE_DIR, 'data/{}_labels.csv'.format(directory)),
            index=None)
        print('Successfully converted xml to csv.')

# main()
main_daisy()