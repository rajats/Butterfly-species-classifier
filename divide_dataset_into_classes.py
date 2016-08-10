import re
import os
import zipfile
from os import walk
from shutil import copy

filepath = os.path.abspath("leedsbutterfly_dataset_v1.0.zip")
zip_ref = zipfile.ZipFile(filepath, 'r')
zip_ref.extractall()
zip_ref.close()

dataset_path = os.path.abspath("leedsbutterfly")
os.chdir(dataset_path)
images_path = os.path.abspath("images")
os.mkdir('classes')
class_path = os.path.abspath("classes")
os.chdir(class_path)
for iclass in range(10):
    os.mkdir(str(iclass+1))
os.chdir(images_path)

all_images = []
for (dirpath, dirnames, filenames) in walk(images_path):
    all_images.extend(filenames)
    break

for iclass in range(10):
    for img in all_images:
        if iclass < 9:
            rgx = r"^00"+re.escape(str(iclass+1))+r".+[.jpg]"
        else:
            rgx = r"^0"+re.escape(str(iclass+1))+r".+[.jpg]"
        if re.match(rgx, img):
            copy(images_path+"/"+img, class_path+"/"+str(iclass+1))
