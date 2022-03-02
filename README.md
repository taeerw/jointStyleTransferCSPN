# jointStyleTransferCSPN
![image](https://github.com/taeerw/jointStyleTransferCSPN/blob/main/images/Slide1.PNG)

Repository for the paper "Joint learning of linear transformations and Spatial Propagation Networks for fast image style transfer"
This repository is based on https://github.com/sunshineatnoon/LinearStyleTransfer and https://github.com/XinJCheng/CSPN
- Li, Xueting, et al. "Learning linear transformations for fast image and video style transfer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.‏
- Cheng, Xinjing, Peng Wang, and Ruigang Yang. "Learning depth with convolutional spatial propagation network." IEEE transactions on pattern analysis and machine intelligence 42.10 (2019): 2361-2379.‏

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
  
