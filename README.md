# SSD_Light
Reference
---------
I refered `balancap's SSD` code. But his code is too heavy for me to understand. So I decided to write my own SSD code.<br>
My code is strongly similar to his.

Progress
--------
Under development<br>
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

Next task:<br>
Train, test, adjust parameters<br>
Then write a better picture saver.<br>
Finally write run.py, config.conf<br>

Sadness
-------
Due to my fault. Old commit log was missiing.
