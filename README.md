# StereoBVS
This is the C++ code for our paper Automatic Streaming Segmentation of Stereo Video Using Bilateral Space. 
This implemetation is based on Visual Studio 2015 and OpenCV 2.4.

### Introduction
Give a sequence of left-view frames from a stereo video and the corrponding disparity images, this project can generate the foreground segmentation results for all frames.

### Quick start
1. Put your left-view image sequences in `data\left_image_folder\`, and disparity image seqences in `data\disparity_image_folder`.
The filenames of images should be in the form of 0.png, 1.png, 2.png, ...., etc.
2. Configure your visual studio project
  * configure opencv
  * add `code\inc\`, `graphcut\inc` and `graphcut\src` in include paths
3. run the project and you'll find the results in `data\result_image_folder`
