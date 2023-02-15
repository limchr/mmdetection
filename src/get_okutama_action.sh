#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash ./scripts/get_coco.sh

url=https://okutama-action.s3.eu-central-1.amazonaws.com/


# Download/unzip labels
d='./okutama_action/TrainSetFrames/' # unzip directory
mkdir -p $d
f='TrainSetFrames.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background


d='./okutama_action/TestSetFrames/' # unzip directory
mkdir -p $d
f='TestSetFrames.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
