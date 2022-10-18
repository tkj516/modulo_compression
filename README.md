# modulo_compression


## Open Image Dataset
The Open Images dataset is under data/rc_pytorch/train_oi_r.  The original tar 
files have been moved to data2/Open_Images.

## Setting up Tensorflow
Tensorflow was not using the GPU be default.  Basically, I followed the 
installation steps on the Tensorflow website and things seem to be working.  The
CUDA toolkit version is 11.2 and the CuDNN version is 8.0.1.  These are installed
sepcifically for the ``modulo" Anaconda environment that I am using now.

## Implementation Details
- tf.floormod is giving rise to JIT compilation error so currently manually 
implemented it using tf.floordiv
