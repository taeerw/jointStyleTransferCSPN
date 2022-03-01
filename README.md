# jointStyleTransferCSPN
Repository for the paper "Joint learning of linear transformations and Spatial Propagation Networks for fast image style transfer"

# Prerequisites
- Pytorch
- torchvision


# Data preparation
- MSCOCO:
  wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
- WikiArt:
  Download from: https://www.kaggle.com/c/painter-by-numbers
  
  
 # Training
 python Train_end_to_end.py --contentPath MSCOCO_PATH --stylePath WIKIART_PATH --outf OUTPUT_DIR
 
 # Testing
 python TestPhotoReal.py
  
