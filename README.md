# Cup class detection project

## Main idea
The main concept is to create a system that would be used to check the quality of packaging of food products such as yoghurts or other food products on production lines. The project aims to automate the quality control process through an intelligent vision system based on image processing and Convolutional Neural Networks. The main task of the network would be to detect dirty, damaged and unmarketable items. Placing the camera next to the conveyor belt would allow the elimination of individual pieces of the product already at the stage of exiting the machine, before reaching the carton. 

The system control will be based on a Raspberry PI microcontroller with an additional camera. Further, a database will be created containing product photos that will be used to train the neural network model. In the first phase, the project will be implemented in a simulation test environment, and soon on the packing machine.
The considerations would focus on finding the most optimal conditions for the operation of the system, such as the use of additional light sources, appropriate structures of the neural network model and applications in an industrial environment. 

#### Old version - object classification
See on YouTube: [Cup class detection](https://www.youtube.com/watch?v=bSiHZTzwNEM&t=2s)

[<img src="assets/yt_video_speed.gif" width="450" height="300">](https://www.youtube.com/watch?v=bSiHZTzwNEM&t=2s)

#### New version - object detection

[<img src="assets/roboflow_labels.png" width="550" height="300">](https://www.youtube.com/watch?v=bSiHZTzwNEM&t=2s)


### TODO
#### Database
* [x] Create the first database (1000 photos, 4th class) 
* [x] Use Roboflow to store a database
* [x] Mark images with VOC labels
* [ ] Complete the photo database - photos of cups with different labels, with and without lighting, photos on different backgrounds and in various configurations 
#### Model training 
* [x] Prepare your first CNN model
* [x] Use pretrained Models and google collab
* [x] Prepare scripts in Jupyter Notebook and Google Collab(for [Yolov4 Training](link), [SSD Mobilenet](link), [.pb model conversion in tflite](link)
#### Script for camera
* [x] Prepare script for camera in anaconda
* [x] Launch of the tflite model 
* [x] Show FPS number on the video stream
#### Testing and maintaining script on Raspberry PI 
* [x] Migrate model to RPI
* [ ] Choose the most efficient solution
* [ ] Deploy an anti-crash solution and automatic script execution after restart
* [ ] Try to execute the script as a service and show video in the sample API 

### Prerequisites
* Tensorflow 2.5.0
* Object Detection API

#### Convert to tflite

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

### Performance

#### Resolution and FPS

| Detection    | 512x512 | 416x416 | 320x320 |
|--------------|---------|---------|---------|
| YoloV4 Tiny  | 55.43   | 52.32   |         |
| EfficientDet | 61.96   | 57.33   |         |
| SSDMobileNet | 61.96   | 57.33   |         |


## Example Factor - light.
Before applying light :

<img src="assets/unclean_dark.gif" width="450" height="300">

After applying light :

<img src="assets/unclean_light.png" width="450" height="300">

### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [link](link)
