ProbabilisticGraphicalModels
==========================

Coding projects from my PGM class @ UMass Amherst

### Assignments

####Assignment 1 (BayesNets)

* Build a Bayesian network for heart disease prediction according to the random variables and dependencies in the given illustration
* Calculate the conditional probability tables for each variable given training data
* Perform marginal and MAP queries on the calculated conditional probability distribution of single variables given a configuration of the other random variables 
  in the graph which possibly contain unobserved variables other than the query variable
* Find the mean and standard deviation of MAP query accuracy on the heart disease variable for five pairs of training and testing data sets.

####Assignment 2 (CRF OCR)

* Build a linear-chain conditional random field (CRF) for optical character recognition (OCR) with binary pixel values as features and the ten most commonly used 
  letters of the alphabet as labels
* Implement the message-passing dynamic-programming algorithm for MAP inference on markov networks
* Condition the random variables associated with letters on the random variables associated with pixel values by reducing the markov network
* Implement the objective and gradient functions for learning in a CRF
* Implement learning by optimizing the parameters for OCR on the given data set using an optimizer from SciPy (I used fmin\_l\_bfgs\_b)
* Record training time, accuracy and optimized average conditional log likelihood for trainin sets of different sizes
