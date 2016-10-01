# quantized_net by chainer

This is an experimental code for reproducing [1]'s result using chainer. 
No optimization is used for binary operations. I just binalize weight and activation at computation, and use a straight through estimator for gradient computation. 

- [1] "Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations", Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
https://arxiv.org/abs/1609.07061

Code is almost equivalent to chainer/examples/mnist/ except:

- Use quantized weight, quantized activation, batch_normalization (net.py, qst.py, link_quantized_linear.py function_quantized_linear.py)
- Use weight clip, optimizer.add_hook(weight_clip.WeightClip()) (weight_clip.py)


Usage
```
# cpu
./train_mnist.py

# gpu (use device id=0)
./train_mnist.py --gpu=0
```

Result
```
GPU: 0
# unit: 1000
# Minibatch-size: 100
# epoch: 20

epoch       main/loss   validation/main/loss  main/accuracy  validation/main/accuracy
1           0.498896    0.413862              0.944183       0.9693
2           0.395025    0.38905               0.976715       0.9745
3           0.373157    0.382003              0.984215       0.9738
4           0.357318    0.371697              0.990015       0.9773
5           0.349601    0.368348              0.991598       0.9786
6           0.340391    0.366556              0.994482       0.9794
7           0.337083    0.363402              0.995299       0.9787
8           0.332844    0.35933               0.996549       0.9789
9           0.331479    0.357395              0.996716       0.9801
10          0.327898    0.356452              0.996949       0.9779
11          0.325316    0.356169              0.997532       0.979
12          0.324888    0.353728              0.997899       0.9808
13          0.322971    0.35396               0.998266       0.9805
14          0.324934    0.35352               0.997949       0.9801
15          0.32234     0.354705              0.998449       0.979
16          0.322491    0.350558              0.998349       0.9806
17          0.321024    0.353457              0.998649       0.9803
18          0.320683    0.351302              0.998699       0.9813
19          0.32054     0.35186               0.999116       0.9803
20          0.321184    0.351329              0.998499       0.9806
```
