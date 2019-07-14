# 1-Flowers

There is a database of photos of flowers from  [**kaggle**](https://www.kaggle.com/alxmamaev/flowers-recognition#flowers.zip). 

## 1.0

I try to get feature vectors from the database and construct a simple network to identify these flowers, which is the same as the network in `../Demonstrate-1.py`.

I need first read all the files in dataset and get the color information about every pixel. Then store it in a verctor which is the eigenvector for the corresponding photo.  Finally I use these eigenvectors to train our network with `tensorflow.keras`

I devide dataset randomly into 5 pieces. And I will use 1 part to be the test set and outhers to be the training data.

The result seems unsatisfactory. The best accuracy is only about `50%` or less. But needless to say, the first try is succefully. I have constructed a network using the `tensorflow.keras` and succefully run it on a dataset that extracted by myself. 

## 2.0

Used an Alex Net to train data. Because of the change of image size, I made some adjustment. 

I write a program named `SystemCall.py` to make the model automaticly try different parameters and save the output into the corresponding files. The parts of the output files are in `./Output/`.

And after that, we use the `./Output/Original_Output/format.py` to format these output files. Then use `./Output/Graph/plot.py` to generate graphs.

The output graphs are in `./Output/Graph/`.

We can also adjust the numbers of cells in dense layers or in convolution layers.

## 3.0

There are 2 dropout layers in this network. Since most of the cases appears the overfitting is a big problem with the parameters he gives, so I will adjuct the `Dropout` layers to optimize the results.

This version has best results, about 65% accuracy.

## 3.1

Version 2.1 is based on 2.0. I just make some improvement to the `SystemCall.py` file. And I also modified the `KerasNetwork.py` that allows it to accept command parameters as some parameters in the network. So I make it easier to change some small parts of the `CNN` network. 