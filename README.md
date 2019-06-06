# SSD_Light

# ATTENTION! I'm sorry that this project is not able to used directly yet! You should read the code and modify it before use! I will finish it if hava enough time.

Reference
---------
I refered `balancap's SSD` code. But his code is too heavy for me to understand. So I decided to write my own SSD code.<br>
My code is strongly similar to his.<br>

Envs
----
* `Python 3.6.8`<br>
* `tensorflow-gpu 1.4.0`<br>
* `tensorflow-tensorboard 0.4.0`<br>
* `CUDA Version 8.0.61`<br>
* `CUDNN Version 6.0.21`<br>

Test
----
The test function is not well finished.<br>
`python run.py --test --image_path /TEST_IMAGE_PATH`<br>
`--model_path /YOU_MODEL_PATH --out_path /OUT_RESULT_PATH` and batch_size are also changeable<br>
Look run.py and test_ssd.py for details.<br>

Train
-----
`python run.py --train --image_path /TRAIN_IMAGE_PATH --annotation_path /TRAIN_ANNOTATION_PATH`<br>
`--restore` is an optional that you can continue previous training.<br>
batch_size, model_path, learning_rate, epochs are also caon be changed.<br>
Look run.py and train_ssd.py for details<br>
Your own data must be `VOC2007 like`. To change FDDB dataset to VOC like, you can see [my other project](https://github.com/stpraha/FDDB2VOClike).

Example
-------
![pic1](https://github.com/stpraha/SSD_Light/blob/master/examples/0000.jpg)
![pic2](https://github.com/stpraha/SSD_Light/blob/master/examples/0001.jpg)


Progress
--------
Under development<br>
This project is almost finished at 2019/3/22.<br>

2019/3/5<br>
Fixed a bug: Attempting to use uninitialized value<br>
Refactored loss_function, ssd_net and train_ssd<br>

2019/3/6<br>
Fixed a bug: with training loop goes on, the speed decreases.<br>
Rewrite ground_truth_process.py. It was changed from using Tensorflow to using Numpy.<br>

2019/3/9<br>
Fixed a bug: dropout layer still working when testing.<br>
Rewrite loss_function.py. Rewrite the SmoothL1-Loss function.<br>
Add test_ssd.py.<br>
Add nms: encode_predictions.py.<br>
Add picture saver: draw_pic.py.<br>

2019/3/16<br>
Fixed a bug: the loss doesn's decrease.<br>
Found a bug in balancap's code. Submitted an Issue.<br>
Build a new path.<br>
Single class location and classification is abled.<br>

2019/3/22<br>
A runnable version is published.<br>
run.py was added.<br>
Some functions wer rewrited.<br>
Try to adjust parameters, but it doesnt work well.<br>

Next task:<br>
Make 


Sadness
-------
Due to my fault. Old commit log was missiing.
