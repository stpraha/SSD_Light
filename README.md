# SSD_Light

# ATTENTION! I'm sorry that this project is NOT able to be used directly! If you turly want to make it runnable, read the code and modify it! I will finish it when I hava enough time.

Reference
---------
I refered `balancap's SSD` code. However, his code is too heavy for me to understand. So I decided to write my own SSD code.<br>
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
Check run.py and test_ssd.py for details.<br>

Train
-----
`python run.py --train --image_path /TRAIN_IMAGE_PATH --annotation_path /TRAIN_ANNOTATION_PATH`<br>
`--restore` is optional which make you able to continue previous training.<br>
batch_size, model_path, learning_rate, epochs are also changeable.<br>
Check run.py and train_ssd.py for details<br>
Training data must be `VOC2007 like`. To change the format of data set from FDDB to VOC, see [my other project](https://github.com/stpraha/FDDB2VOClike).

Example
-------
![pic1](https://github.com/stpraha/SSD_Light/blob/master/examples/0000.jpg)
![pic2](https://github.com/stpraha/SSD_Light/blob/master/examples/0001.jpg)


Progress
--------
Under development<br>
This project is almost terminated on 2019/3/22.<br>

2019/3/5<br>
Fix a bug: Attempting to use uninitialized value<br>
Refactored loss_function, ssd_net and train_ssd<br>

2019/3/6<br>
Fix a bug: with training loop goes on, the speed decreases.<br>
Rewrite ground_truth_process.py. It was changed from using Tensorflow to using Numpy.<br>

2019/3/9<br>
Fix a bug: dropout layer still working when testing.<br>
Rewrite loss_function.py. Rewrite the SmoothL1-Loss function.<br>
Add test_ssd.py.<br>
Add nms: encode_predictions.py.<br>
Add picture saver: draw_pic.py.<br>

2019/3/16<br>
Fix a bug: the loss doesn's decrease.<br>
Found a bug in balancap's code. Submitted an Issue.<br>
Build a new path.<br>
Single class location and classification is abled.<br>

2019/3/22<br>
A runnable version is published.<br>
run.py was added.<br>
Some functions were rewrited.<br>
Try to adjust parameters, but it doesnt work well.<br>

