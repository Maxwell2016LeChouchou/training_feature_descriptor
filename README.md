# training_feature_descriptor
Building shallow/middle depth CNN models to train a feature descriptor to assitant Deep SORT matching cascade (part my Thesis Project)
Still use softmax/cross_entropy to observe the training processing but I only need to extrac the features from the first a few layers 

The images, tfrecords and ckpt are saved in my local computer due to github only allows passing less 100MB online


# Data Processing:
Dataset from Youtube faces

1. Postive Training dataset consists of all youtube celebrity face groudtruth which extract their bboxs from youtube faces, as a result of being no background from each picture, only faces

2. Negative Training dataset is background obtained from the original pictures of youtube faces, have same size of faces. Details are in code: test_4.py

3. Overall postive training dataset has 8494 images, negative training dataset has 25482 images to satify Postive Training Dataset :  Negative Training Dataset = 1 : 3

4. Likewise the training dataset, the eval dataset also have the same structures of training dataset, and eval postive training dataset(1945 pos images) : eval negtive training dataset(5838 neg images)

# Convert to tfrecord:
Convert all the postive dataset with label (faces = 1), and negative dataset with label(non-faces = 0) to tfrecords. There is two ways to do so:
1. Maintain same channel (RGB) but resize the images all in one standard size: generate_tfrecord.py

2. Convert it to Grayscale and resize the iamges: generate_tfrecord_v2.py

# Training Data:
Simple CNN models procedure: Parse tfrecords -> image/label -> CNN/ResNets --> Softmax/Cross_Entropy --> 1/0 = faces/non-faces

There is two ways to do so:
1. Use a ResNets which consists of 2 CNN, 1 max_pool, 6 ResNets and then dense the layer: main.py
2. Use a simple CNN which consists 4 CNNs, 3 max_pool, and 2 dense layers: main_v2.py


# Result:
Above 99% training (supposed to be, simple classification, unique dataset)

Result.png
