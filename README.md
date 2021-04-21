# Artificial Neural Networks From Scratch

This is a collection of from scratch implementations of classic neural network
archictures applied to a wide range of problems, e.g. classification, function
approximation, time series prediction, similarity clustering, error correction
and autoencoding. Contains implementations of the following network types:

| Network              | Useful for                                                                       |
| -------------------- | -------------------------------------------------------------------------------- |
| Perceptron           | Classification and regression tasks.                                             |
| RBF networks         | Modelling distributions as mixtures of gaussians.                                |
| Self Organizing Maps | Intelligently clustering data based on similarity in one or multiple dimensions. |
| Hopfield Networks    | Recalling distorted patterns.                                                    |
| Autoencoder          | Learning compression functions.                                                  |

The network implementations can be found under `ann/` and notebooks
demonstrating their use under `notebooks`, to run them, first install the
dependencies with `pip -r requirements.txt` and then start a Jupyter notebook
with `PYTHONPATH` containing the root directory of this repository.
