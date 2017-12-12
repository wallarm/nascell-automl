
if you want to use the following architecture for MNIST:
- input layer : 784 nodes (MNIST images size)
- first convolution layer : 5x32
- first max-pooling layer: 2
- second convolution layer : 5x64
- second max-pooling layer: 2
- output layer : 10 nodes (number of class for MNIST)
you can do it with following command: 
```
python3 train.py --architecture "5, 32, 2,  5, 3, 64, 2, 3"
```
every 4 numbers represent the size of the kernel, count of filters, max-pooling and dropout per layer. 


