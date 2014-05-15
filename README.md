ProbabilisticGraphicalModels
==========================

Coding projects from my PGM class @ UMass Amherst. Most projects are vectorized as much as possible and rely heavily on numpy for this vectorization.

### Assignments

####Assignment 1 (BayesNets)

* Build a Bayesian network for heart disease prediction according to the random variables and dependencies in the given illustration
* Calculate the conditional probability tables for each variable given training data
* Perform marginal and MAP queries on the calculated conditional probability distribution of single variables given a configuration of the other random variables 
  in the graph which possibly contain unobserved variables other than the query variable
* Find the mean and standard deviation of MAP query accuracy on the heart disease variable for five pairs of training and testing data sets.

####Assignment 2 (CRFOCR)

* Build a linear-chain conditional random field (CRF) for optical character recognition (OCR) with binary pixel values as features and the ten most commonly used 
  letters of the alphabet as labels
* Implement the message-passing dynamic-programming algorithm for MAP inference on markov networks
* Condition the random variables associated with letters on the random variables associated with pixel values by reducing the markov network
* Implement the objective and gradient functions for learning in a CRF
* Implement learning by optimizing the parameters for OCR on the given data set using an optimizer from SciPy (I used fmin\_l\_bfgs\_b)
* Record training time, accuracy and optimized average conditional log likelihood for trainin sets of different sizes

####Assignment 3 (GibbsCRFDenoise)

* Build a grid-structured conditional random field (CRF) for image denoising with noisy pixel values as features and correct pixel values as labels
* Condition the random variables associated with correct pixel values on the random variables associated with the original, noisy pixel values
* Implement a by-pixel Gibbs sampler for the grid-structured CRF; random numbers are sampled from uniform distribution for binary-valued pixels and from normal 
  distribution for real-valued, grayscale pixels
* For the real-valued pixels, run the Gibbs sampler with both a single pair-wise weight and specific pair-wise weights that penalizes large deviations from 
  original difference between neighboring pixels
* Run the Gibbs sampler on binary-valued pixels both for a fixed number of iterations and until convergence, where convergence is defined as a difference between
  the Mean Absolute Error of time step t and t-1 being less than a constant for three consecutive iterations

####Assignment 4 (RBMOCR)

* Build a restricted Boltzmann machine (RBM) to infer hidden variable values from observed pixel values for MNIST hand-written digit images
* Implement inference for an RBM by conditioning on the observed pixel values of an MNIST instance and using weights from training to calculate the hidden variable 
  values
* Implement training for the RBM's parameters using mini-batch stochastic gradient ascent where each batch consists of multiple MNIST training images
* Use a multi-chain Gibbs sampler that alternates conditioning on the observed pixel values and the hidden variables values to calculate negative gradient terms
