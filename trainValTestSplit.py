import os
import random
import shutil
from shutil import copyfile

img_path ='/Users/maisycook/Documents/Plant Disease Detection/PlantVillage_for_object_detection/Dataset/images'
label_path ='/Users/maisycook/Documents/Plant Disease Detection/PlantVillage_for_object_detection/Dataset/labels'

ipaths=[]
types=[]
for dirname, _, filenames in os.walk(img_path):
    for filename in filenames:
        ipaths+=[os.path.join(dirname, filename)]
        types+=[filename.split('.')[-1]]
tpaths=[]
for dirname, _, filenames in os.walk(label_path):
    for filename in filenames:
        tpaths+=[os.path.join(dirname, filename)]    
ipaths=sorted(ipaths)
tpaths=sorted(tpaths)
paths=[]
for ip,tp in zip(ipaths,tpaths):
    paths+=[(ip,tp)]
random.shuffle(paths)
ipaths=[]
tpaths=[]
for p in paths[0:1000]:
    ipaths+=[p[0]]
    tpaths+=[p[1]]
    
    print(set(types))
for typei in list(set(types)):
    print(typei,types.count(typei))
    
    

os.makedirs('datasets', exist_ok=True)
os.makedirs('datasets/train', exist_ok=True)
os.makedirs('datasets/valid', exist_ok=True)
os.makedirs('datasets/test', exist_ok=True)

for i in range(len(ipaths)):
    ipath=ipaths[i]
    ifile=ipath.split('/')[-1]
    tpath=tpaths[i]
    tfile=tpath.split('/')[-1]

    if i%3==0:
        copyfile(ipath, f'datasets/train/{ifile}')
        copyfile(tpath, f'datasets/train/{tfile}')
    elif i%3==1:
        copyfile(ipath, f'datasets/valid/{ifile}')
        copyfile(tpath, f'datasets/valid/{tfile}')
    else:
        copyfile(ipath, f'datasets/test/{ifile}')
        

        
        