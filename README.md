
# Enhanced Lightweight ViT for FGVC Tasks with SAA & EIP

### Introduction

A repository for the code used to create and train the model defined in: 

"Enhanced Fine-Grained Visual Classification through Lightweight Transformer Integration and Auxiliary Information Fusion".

### Quick Start

* install requirements
```
pip install -r requirements.txt
```

* The default dataset is [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). You should download the dataset and put it in respective folders (\<root\>/datasets/) and Unzip file.
The folder sturture as follow:
```
datasets
  |———cub-200
  |     └—————images/
  |     └—————attributes/
  |     └—————image_class_labels.txt
  |     └—————images.txt
  |     └—————train_test_split.txt
  |     └—————classes.txt
  |———deepfashion
  |———grainset
```
* Try to run the following comand to start training.
```
python3 train.py --epoch 100 --num-classes 200 --mask-type linear --data-dir "./datasets/cub-200"
```

### Acknowledgement
Many thanks for [DCL](https://github.com/JDAI-CV/DCL) and [EfficientFormer](https://github.com/snap-research/EfficientFormer). A part of the code is borrowed from them.
