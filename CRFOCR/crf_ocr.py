import glob, math, operator, time 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp
from collections import defaultdict, OrderedDict

def is_numeral(char):

  try:
    int(char)

    return True
  except ValueError:

    return False

def get_datum_number(file_path):

  numerical_chars = [char
                     for char in file_path
                     if is_numeral(char)]

  return int(reduce(lambda x,y: x + y, numerical_chars))

def list2staggered(original):

  return zip([None]+original,original+[None])[1:-1]

def dir2data(data_path, search_str):
  """Takes a file path and a search string and returns a map of
  each datum file's number to its data matrix"""

  file_paths = glob.glob(data_path + search_str + "_img*.txt")

  number_path_map = {get_datum_number(file_path) :
                     file2matrix(file_path)
                     for file_path in file_paths}

  with open(data_path + search_str + "_words.txt") as words:
    true_words = [word[0:len(word)-1] 
                  for word in words]

  return {datum_number :
          (file_path, true_words[datum_number-1])
          for (datum_number, file_path) in number_path_map.items()}

def file2matrix(file_path):
  """Takes a file path and returns a matrix of the data found in
  the file"""
  
  with open(file_path) as rows:
    row_list = [map(lambda item: float(item), row.split()) 
                for row in rows]

  return np.array(row_list)

def files2model(transition_weights_path, feature_weights_path, ordering):
  """Takes the file paths for the transition and feature weights, a list
  that represents the ordering and returns a tuple of data structures that
  represent the model"""

  letter_index_map = {ordering[index] :
                      index 
                      for index in range(len(ordering))}

  return (file2matrix(transition_weights_path),
          file2matrix(feature_weights_path).T,
          ordering,
          letter_index_map)

def get_nodes(model, img_matrix):
  """Takes the model and an image matrix and returns a vector containing
  the node potentials for the input image matrix"""

  feature_weights = model[1]

  return np.dot(img_matrix, feature_weights)

def forward_factor_product(sender_matrix, parent_matrix):
  """Takes a sender matrix and a parent/recipient matrix and 
  returns the forward factor product"""

  return (sender_matrix.T + parent_matrix).T

def backward_factor_product(sender_matrix, parent_matrix):
  """Takes a sender matrix and a parent/recipient matrix and 
  returns the backward factor product"""

  return sender_matrix + parent_matrix

def get_log_cliques(model, img_matrix):
  """Takes a model and an image matrix and returns the clique 
  potentials for the label sequence conditioned on the image 
  matrix"""

  transition_matrix = model[0]
  nodes = get_nodes(model, img_matrix)
  num_nodes = nodes.shape[0]
  cliques = [forward_factor_product(transition_matrix, nodes[index])
             for index in range(num_nodes-1)]
  last_clique = backward_factor_product(cliques[-1], nodes[-1])

  return cliques[0:-1] + [last_clique]

def get_parent_matrix(index_pair, tau_cache, num_cliques):
  
  (sender_index, parent_index) = index_pair

  return 0 \
         if parent_index not in range(num_cliques) else \
         tau_cache[sender_index][parent_index]

def get_log_messages(model, cliques):
  """Takes a model and a clique tree (chain) and returns a map 
  from recipient cliques to sender cliques to message vectors"""

  num_cliques = len(cliques)
  last_clique = cliques[num_cliques-1]
  indeces = range(num_cliques)
  tau_cache = defaultdict(lambda: {})

  #Calculate forward messages
  for index in indeces[0:num_cliques-1]:
    sender_matrix = cliques[index]
    parent_matrix = get_parent_matrix((index, index-1), tau_cache, num_cliques)
    factor_product = forward_factor_product(sender_matrix, parent_matrix)
    tau_cache[index+1][index] = logsumexp(factor_product, axis=0)
  
  #Calculate backward messages
  for index in reversed(indeces[1:num_cliques]):
    sender_matrix = cliques[index]
    parent_matrix = get_parent_matrix((index, index+1), tau_cache, num_cliques)
    factor_product = backward_factor_product(sender_matrix, parent_matrix)
    tau_cache[index-1][index] = logsumexp(factor_product, axis=1)

  return tau_cache

def get_log_belief(recipient, clique, message_cache):
  
  return reduce(lambda x,y: backward_factor_product(x,y[1]) \
                  if y[0] > recipient else \
                  forward_factor_product(x,y[1]),
                message_cache.items(),
                clique)

def get_log_beliefs(model, img_matrix):
  """Takes a model and an image matrix and returns a list of
  matrices that each represent the beliefs for the clique associated
  with the index in the list"""

  cliques = get_log_cliques(model, img_matrix)
  messages = get_log_messages(model, cliques)
  
  return [get_log_belief(index, clique, messages[index])
          for index, clique in enumerate(cliques)]

def get_pairwise_marginals(model, img_matrix):
  """Takes a model and an image matrix and returns a list of 
  matrices that represent the pairwise marginal distributions for 
  each clique in the label sequence conditioned on the image matrix"""

  beliefs = get_log_beliefs(model, img_matrix)

  return [np.exp(belief - logsumexp(belief))
          for belief in beliefs]

def get_marginals(model, img_matrix):
  """Takes a model and an image matrix and returns a list of vectors 
  that represent the distributions over letters for each position in 
  the label sequence conditioned on the image matrix"""
  
  pairwise_marginals = get_pairwise_marginals(model, img_matrix)
  first_marginals = [np.sum(pairwise_marginal, axis=1)
                     for pairwise_marginal in pairwise_marginals]
  last_marginal = np.sum(pairwise_marginals[-1], axis=0)
  
  return first_marginals + [last_marginal]

def get_characters(model, img_matrix):
  """Takes a model and an image matrix and returns a list 
  of characters representing the MAP label sequence according to
  positional marginals conditioned on the image matrix"""

  ordering = model[2]
  marginals = get_marginals(model, img_matrix)
  argmax = np.argmax(marginals, axis=1)
  
  return [ordering[arg]
          for arg in argmax.tolist()]

def get_train_model(lambdas, index_info):
  """Takes a weights vector and a 2-tuple with and ordering 
  and a map from letters to indeces and returns a model in the 
  form that the sum-product message-passing code can use"""

  num_labels = len(index_info[0])
  num_features = (lambdas.shape[0] - pow(num_labels,2))/num_labels
  transitions = np.reshape(lambdas[0:pow(num_labels,2)], 
                           (num_labels, num_labels))
  features = np.reshape(lambdas[pow(num_labels,2):], 
                        (num_labels, num_features)).T

  return (transitions, features, index_info[0], index_info[1])

def get_log_likelihood(model, img_matrix, word):
  """Takes a model, a pixel matrix and a word for that 
  matrix and returns the conditional log likelihood for 
  the word conditioned on the observed pixel matrix 
  according to the model"""

  letter_index_map = model[3]
  cliques = get_log_cliques(model, img_matrix) 
  beliefs = get_log_beliefs(model, img_matrix)
  indeces = [letter_index_map[letter]
             for letter in word]
  index_pairs = list2staggered(indeces)
  numerator = sum([clique[letter1, letter2]
                   for clique, (letter1, letter2) in zip(cliques, index_pairs)])
  denominator = logsumexp(beliefs[0])

  return numerator - denominator

def get_average_log_likelihood(lambdas, data, index_info):
  """Takes a weights vector, a data set and a 2-tuple with an 
  ordering and a map from letters to indeces and returns the 
  negative average conditional log likelihood according to the 
  weights vector and the data set"""

  num_labels = len(index_info[0])
  model = get_train_model(lambdas, index_info)
  N = float(len(data))
  log_likelihoods = [get_log_likelihood(model, img_matrix, word)
                     for (img_matrix, word) in data]

  return -sum(log_likelihoods)/N

def get_transition_gradient(model, data):
  """Takes a model and a data set and returns the CRF
  gradient of the transitions weights according to the model
  and data"""

  ordering = model[2]
  pw_marginals = [get_pairwise_marginals(model, img_matrix)
                  for (img_matrix, word) in data]
  gradients = [sum([(1 if labeli == letter1 and labelj == letter2 else 0) - \
                      clique[i,j]
                    for pw_marginal, (img_matrix, word) in zip(pw_marginals, data)
                    for clique, (letter1, letter2) in 
                      zip(pw_marginal, list2staggered(list(word)))])
               for i, labeli in enumerate(ordering)
               for j, labelj in enumerate(ordering)]

  return np.array(gradients)/len(data)

def get_feature_gradient(model, data):
  """Takes a model and a data set and returns the 
  CRF gradient of the feature weights according to the 
  model and the data"""

  ordering = model[2]
  num_features = data[0][0].shape[1]
  marginals = [get_marginals(model, img_matrix)
               for (img_matrix, word) in data] 
  gradients = [sum([(1 if letter == label else 0) - \
                      marginal[letter_num][label_num]
                    for marginal, (img_matrix, word) in zip(marginals, data)
                    for letter_num, letter in enumerate(word)
                    if img_matrix[letter_num, feature_num] == 1])
               for label_num, label in enumerate(ordering)
               for feature_num in range(num_features)]

  return np.array(gradients)/len(data)

def get_gradient(lambdas, data, index_info):
  """Takes a weights vector, a data set and a 2-tuple with 
  an ordering and a map from letters to indeces and returns 
  the negative CRF gradient according to the weights and data"""

  model = get_train_model(lambdas, index_info)
  transition_grad = get_transition_gradient(model, data)
  feature_grad = get_feature_gradient(model, data)
  gradient = np.concatenate((transition_grad, feature_grad))

  return gradient * -1

def train(ordering, data):
  """Takes an ordering and a data set and returns a 
  model trained on the data set"""

  num_labels = len(ordering)
  num_features = data[0][0].shape[1]
  initial_params = np.zeros(pow(num_labels, 2) +
                            num_features*num_labels)
  index_info = (ordering, {letter : 
                           index 
                           for index, letter in enumerate(ordering)})
  min_info = fmin_l_bfgs_b(get_average_log_likelihood, 
                           initial_params,
                           get_gradient,
                           args=(data, index_info))

  return get_train_model(min_info[0], index_info)

def params2string(array):
  """Takes a 2D array of params and returns a string 
  of the elements with spaces between elements and new 
  lines between rows"""

  rows = [reduce(lambda x,y: x + " " + y, 
                 [str(item) for item in row])
          for row in array.tolist()]
  
  return reduce(lambda x,y: x + "\n" + y, rows) + "\n"

def get_accuracy(model, test_data):
  """Takes a model and a list of (image matrix, true word) pairs 
  and returns the accuracy of the model's predictions on the letters
  in the words conditioned on the corresponding image matrices"""

  words = [(word, get_characters(model, img_matrix))
           for (img_matrix, word) in test_data.values()]
  diff_list = [(1 if word[index] == prediction[index] else 0)
                for (word, prediction) in words
                for index in range(len(word))]
  
  return sum(diff_list)/float(len(diff_list))

def get_model_stats(ordering, train_data, test_data):
  """Takes an ordering, training data and test data 
  and returns a tuple of the model trained on the train 
  set, train time in seconds, the prediction error on 
  the test set and the conditional log likelihood on the 
  test set"""

  before = time.clock()
  model = train(ordering, train_data.values())
  after = time.clock()
  num_seconds = after - before
  error = 1 - get_accuracy(model, test_data)
  cond_ll = [get_log_likelihood(model, img_matrix, word)
             for (img_matrix, word) in test_data.values()]
  avg_cond_ll = sum(cond_ll)/len(cond_ll)

  return (model, num_seconds, error, avg_cond_ll)

def plot_stats(model_stats_path):
  """Takes the generic path for all stats files and 
  opens a plot labeled according to the data and labels 
  in the files"""

  stats_file_paths = glob.glob(model_stats_path + "*" + ".txt")
  stats_map = defaultdict(lambda: {})

  for file_path in stats_file_paths:
    with open(file_path, "r") as stats_file:
      for line in stats_file:
        (name, value) = line.split(": ")
        train_size = get_datum_number(file_path)
        stats_map[name][train_size] = float(value)

  stats_map = {key :
               OrderedDict(sorted(val_map.items(), 
                           key = operator.itemgetter(0), 
                           reverse=True))
               for (key, val_map) in stats_map.items()}

  for plot_num, (label, val_map) in enumerate(stats_map.items()):
    plt.subplot(3, 1, plot_num)
    plt.plot(val_map.keys(), val_map.values())
    point_string = "x={0} y={1:.2e}"

    for (x,y) in val_map.items():
      plt.text(x, y, point_string.format(x,float(y)))

    plt.ylabel(label)
    plt.xlabel("Number of Training Examples")

  plt.show()

def main():

  #If you would like to retrain the model to reproduce my results, set "train" to True, and
  #run the program from the terminal

  #If you would like to plot the results with pre-trained parameter files, make sure the path
  #strings are correct, set "train" to False, and run the program from the terminal

  train = False

  model_path = "model/"
  my_transition_weights_path = model_path + "my-transition-params"
  my_feature_weights_path = model_path + "my-feature-params"
  my_model_stats_path = model_path + "model-stats"
  data_path = "data/" 
  train_data = dir2data(data_path, "train")
  test_data = dir2data(data_path, "test")
  ordering = ['e','t','a','i','n','o','s','h','r','d']
  
  if train:
    train_set_sizes = [50*(i+1) for i in range(8)]
    my_models = {num_examples :
                 get_model_stats(ordering, 
                                 {k:v 
                                  for k,v in train_data.items()
                                  if k <= num_examples}, 
                                 test_data)
                 for num_examples in train_set_sizes}

    for model_stats in my_models.items():
      num_examples, (model, num_seconds, error, avg_cond_ll) = model_stats
      num_string = str(num_examples)
      trans_path = my_transition_weights_path + num_string + ".txt"
      feature_path = my_feature_weights_path + num_string + ".txt"
      stats_path = my_model_stats_path + num_string + ".txt"
      stats_string = "Seconds: " + str(num_seconds) + \
                     "\nError: " + str(error) + \
                     "\nAvg. Cond. LL: " + str(avg_cond_ll)
      
      with open(trans_path, "w") as location:
        location.write(params2string(model[0]))

      with open(feature_path, "w") as location:
        location.write(params2string(model[1].T))

      with open(stats_path, "w") as location:
        location.write(stats_string)
  else:
    plot_stats(my_model_stats_path)

if __name__== '__main__': main()
