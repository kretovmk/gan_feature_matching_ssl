## Improved GAN: feature matching

GAN for semi-supervised learning: feature-matching technique. NN constructed according to "later" implementation.

*Later implementation (2018):*

Original code (tf): https://github.com/bruno-31/GAN-manifold-regularization

Paper: https://openreview.net/forum?id=Hy5QRt1wz

*Original implementation (2016):*

Original code (theano): https://github.com/openai/improved-gan

Paper: https://arxiv.org/abs/1606.03498

**Not implemented or different in comparison with original NN: **
* No data-based initialization (from 100 training examples for stats' calculation)
* No learning rate decay (from 1200 to 1400 epochs)
