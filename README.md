# Cup class detection project

## Main idea
The main concept is to create a system that would be used to check the quality of packaging of food products such as yoghurts or other food products on production lines. The project aims to automate the quality control process through an intelligent vision system based on image processing and Convolutional Neural Networks. The main task of the network would be to detect dirty, damaged and unmarketable items. Placing the camera next to the conveyor belt would allow the elimination of individual pieces of the product already at the stage of exiting the machine, before reaching the carton. 


See on YouTube: [Cup class detection](https://www.youtube.com/watch?v=bSiHZTzwNEM&t=2s)

[![Cup class detection](assets/yt_video_speed.gif)](https://www.youtube.com/watch?v=bSiHZTzwNEM&t=2s)


The system control will be based on a Raspberry PI microcontroller with an additional camera. Further, a database will be created containing product photos that will be used to train the neural network model. In the first phase, the project will be implemented in a simulation test environment, and soon on the packing machine.
The considerations would focus on finding the most optimal conditions for the operation of the system, such as the use of additional light sources, appropriate structures of the neural network model and applications in an industrial environment. 

## Example Factor - light.
Before applying light :

<img src="assets/unclean_dark.gif" width="450" height="300">

After applying light :

<img src="assets/unclean_light.png" width="450" height="300">

# tensorflow-yolov4-tflite
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0. 
Convert YOLO v4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

### Prerequisites
* Tensorflow 2.3.0rc0

### Performance
<p align="center"><img src="assets/yt_video_speed.gif" width="640"\></p>

#### Output

##### Yolov4 original weight
<p align="center"><img src="result.png" width="240"\></p>

### Convert to tflite

```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416.tflite

# yolov4 quantize float16
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-fp16.tflite --quantize_mode float16

# yolov4 quantize int8
python convert_tflite.py --weights ./checkpoints/yolov4-416 --output ./checkpoints/yolov4-416-int8.tflite --quantize_mode int8 --dataset ./coco_dataset/coco/val207.txt

# Run demo tflite model
python detect.py --weights ./checkpoints/yolov4-416.tflite --size 416 --model yolov4 --image ./data/kite.jpg --framework tflite
```
Yolov4 and Yolov4-tiny int8 quantization have some issues. I will try to fix that. You can try Yolov3 and Yolov3-tiny int8 quantization 

#### Resolution and FPS

| Detection    | 512x512 | 416x416 | 320x320 |
|--------------|---------|---------|---------|
| YoloV4 Tiny  | 55.43   | 52.32   |         |
| EfficientDet | 61.96   | 57.33   |         |
| SSDMobileNet | 61.96   | 57.33   |         |

### Benchmark
```bash
python benchmarks.py --size 416 --model yolov4 --weights ./data/yolov4.weights
```

### Traning your own model
```bash
# Prepare your dataset
# If you want to train from scratch:
In config.py set FISRT_STAGE_EPOCHS=0 
# Run script:
python train.py

# Transfer learning: 
python train.py --weights ./data/yolov4.weights
```
The training performance is not fully reproduced yet, so I recommended to use Alex's [Darknet](link) to train your own data, then convert the .weights to tensorflow or tflite.


### TODO
#### Database
* [x] Create the first database (1000 photos, 4th class) 
* [x] Use Roboflow to store a database
* [ ] Mark images with VOC labels
* [ ] Complete the photo database - photos of cups with different labels, with and without lighting, photos on different backgrounds and in various configurations 
#### Model training 
* [x] Prepare your first CNN model
* [x] Use pretrained Models and google collab
* [ ] Prepare scripts in Jupyter Notebook and Google Collab(for [Yolov4 Training](link), [SSD Mobilenet](link), [.pb model conversion in tflite](link)
#### Script for camera
* [x] Prepare script for camera in anaconda
* [ ] Launch of the tflite model 
* [ ] Show FPS number on the video stream
#### Testing and maintaining script on Raspberry PI 
* [x] Migrate model to RPI
* [ ] Choose the most efficient solution
* [ ] Deploy an anti-crash solution and automatic script execution after restart
* [ ] Try to execute the script as a service and show video in the sample API 

### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [link](link)
  
   My project is inspired by these previous fantastic YOLO implementations:
  * [Yolov3 tensorflow](link)
  * [Yolov3 tf2](link)
