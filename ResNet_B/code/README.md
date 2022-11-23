======= **Environment** =======

Pytorch 1.7.0, Python 3.8.13, CUDA 11.6

======= **Folders** =======

* cifar-10-batches-py: contains public training, testing, as well as the private testing images. Note the public training and testing files of CIFAR-10 is removed due to exceed uploading size.
* code: all the code files.
* result: prediction on the private testing images.
* saved models: all the trained models, including the best model (checkpint 20).

======= **Training on Public Training Set** =======

Please run in main.py with mode `train`

======= **Testing on Public Testing Set** =======

Please run in main.py with mode `test`
The model with best performace is checkpint 20, which already set as `checkpoint_num_list = [20]`
Feel free to check the performance of other models by change the `checkpoint_num_list` in `main.py` under mode `test`

======= **Prediction on Private Testing Set** =======

Please run in main.py with mode `predict`
The checkpoint_name has been set as `model-bottleneck_res-20` in Configure.py, which is the model with the best performance on public testing set. 
Feel free to check other models by change the checkpoint_name in `prediction_configs` in `Configure.py`. 

====== **Note for the Grader** ======

The resnet version has been set to `2`, which is bottleneck residual block. 
Because the code is implemented based on ResNet with standard residual block and bottleneck residual block, the functions for standard residual block was kept
to avoid errors. 
