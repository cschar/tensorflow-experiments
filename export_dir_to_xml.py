

XML_TEMPLATE="""
<annotation>
	<folder>frames</folder>
	<filename>{filename}</filename>
	<path>{full_file_path}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{img_width}</width>
		<height>{img_height}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>{label_name}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
</annotation>
"""


import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help= 'path to all .jpg --> .xml '
  )
FLAGS, unparsed = parser.parse_known_args()


if FLAGS.image_dir:
  IMG_DIR=os.path.expanduser(FLAGS.image_dir)
else:
  IMG_DIR=os.path.expanduser("~/flower_photos/daisy/")


for idx, img_path in enumerate(os.listdir(IMG_DIR)):
  if img_path[-3:] != 'jpg':
    continue
  if idx % 50 == 0:
    print(idx)
  # if idx > 10:
  #   break
  # if(os.path.isfile(img_path)):
  width, height = -1, -1
  img_full_path = IMG_DIR+img_path
  if img_path[-3:] == 'jpg':
    im = Image.open(img_full_path)
    width,height = im.size
    im.close()
  # print(width, height)

  f = open(IMG_DIR+img_path+".xml", 'w')
  f.write(XML_TEMPLATE.format(
    filename=img_path,
    full_file_path=img_full_path,
    img_width=width,
    img_height=height,
    label_name="daisy",
    xmin=0,
    ymin=0,
    xmax=width-10,
    ymax=height-10

  ))

  f.close()


