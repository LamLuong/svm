# FaceTrainingSVM
Traning and evaluating svm Model

#### Step 1: Prepare data

The image dataset should be split to individual class, each class includes images of a person.
Then split this dataset to 2 sets: train and test data, follow structure below:
```````````````
  dataset
     | ---- train
             |---- class_1
             |---- class_1
     | ---- test
             |---- class_1
             |---- class_2
````````````````
#### Step 2: Build project

Prerequire OpenCV

Clone project
```````````````````
$ git clone 
$ mkdir build
$ cd build
$ cmake ..
$ make -j

assume we have 12 physic cores
$ export OMP_NUM_THREADS=12
```````````````````
#### Step 3: Create feature dataset
Assume using 128 dimensions feature.
```````````````````
$ mkdir -p train_feature/feature_128
$ mkdir -p test_feature/feature_128

$ ./process_data path_to_128.prototxt path_to_128.caffemodel path_to/dataset/train train_feature/feature_128

$ ./process_data path_to_128.prototxt path_to_128.caffemodel path_to/dataset/test test_feature/feature_128
```````````````````

#### Step 4: Training
`````````
$ ./training train_feature/feature_128
`````````

#### Step 5: Evaluating
`````````
$ ./evaluate test_feature/feature_128
`````````
Note: about param of batch_size, learning_rate, regularization and n_epochs, please refer in training.cpp
