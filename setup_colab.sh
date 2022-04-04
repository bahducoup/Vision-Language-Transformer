#!/bin/bash

pip install --upgrade pip
pip install -r requirement.txt
python -m spacy download en_core_web_lg
mkdir /content/Vision-Language-Transformer/data/images
cd /content/Vision-Language-Transformer/data/images
apt-get install aria2

aria2c -c -x 10 -s 10 http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
aria2c -c -x 10 -s 10 http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

cd ..
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
unzip refcoco.zip
python data_process_v2.py --data_root . --output_dir . --dataset refcoco --split unc --generate_mask

