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

## To be continued... 
