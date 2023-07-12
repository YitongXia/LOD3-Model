# add openings to 3D BAG LOD2.2 buidlding model

### Introduction
This project is to reconstruct LOD3 building model utilizing oblique aerial imagery and 3D BAG LOD2.2 building model.

![..\\figures\\]
### file structure:
- stage 1: 
  - data registration model creation
  - data registration, image extraction and rectification
- stage 2:
  - Mask R-CNN detection (in colab)
- stage 3:
  - final integration

### Folder structures


### input data and infomation:
- imagery dataset
- original LOD2.2 building model (after region growing algorithm)
- LOD2.2 building model after surface merging (saved as a new file in surface_merge.py)
- calibrated camera parameters of single imagery:
  - fileName imageWidth imageHeight 
  - camera matrix K [3x3]
  - radial distortion [3x1]
  - tangential distortion [2x1]
  - camera position t [3x1]
  - camera rotation R [3x3]
  - camera model m = K [R|-Rt] X
- external camera parameters
  - imageName, X, Y, Z, Omega, Phi, Kappa
- offset value:
  - X, Y, Z
- a LOD3 CityJSON template file
- openings detection result (in coco format, automatically generated)