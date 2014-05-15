import numpy as np
import png, time
from scipy.misc import logsumexp
from matplotlib import gridspec, cm, pyplot

def file2matrix(file_path):
  """Takes a file path and returns a matrix of the 
  file content"""

  with open(file_path) as data_file:
    data = [[float(number) for number in line.split()]
            for line in data_file]
  
  return np.squeeze(np.array(data))

def files2model(pair_path, visible_path, hidden_path):
  """Takes three file paths for the weight sets and returns 
  a model"""

  pair_w = file2matrix(pair_path)
  visible_w = file2matrix(visible_path)
  hidden_w = file2matrix(hidden_path)

  return {"p" : pair_w, "v" : visible_w, "h" : hidden_w}

def get_energy(visibles, hidden, model):
  """Takes the visibles, the hiddens and the model and returns the 
  energy for the provided visibles and hiddens according to the 
  provided model"""

  visible_term = np.sum(model["v"] * visibles)
  hidden_term = np.sum(model["h"] * hidden)
  pair_term = np.sum((model["p"].T * visibles).T * hidden)

  return -visible_term - hidden_term - pair_term

def v_get_pair_terms(hiddens, pair_ws):

  return np.dot(hiddens, pair_ws.T)

def h_get_pair_terms(visibles, pair_ws):

  return np.dot(visibles, pair_ws)

def get_cond_probs(to_cond, model, v_or_h):
  """Takes the values to condition on, the model and 
  a flag for whether the probabilities should be for 
  visibles or hiddens and returns the conditional 
  probabilities"""

  sample_ws = model[v_or_h[0]]
  pair_ws = model["p"]
  get_pair_terms = {"v" : v_get_pair_terms, 
                    "h" : h_get_pair_terms}
  pair_terms = get_pair_terms[v_or_h](to_cond, pair_ws)
  pre_numers = sample_ws[np.newaxis] + pair_terms
  numers = np.exp(pre_numers).astype(float)
  denoms = (numers + 1).astype(float)
  
  return numers / denoms

def get_sample(to_cond, model, v_or_h):
  """Takes the values to condition on, the model and 
  a flag for whether the probabilities should be for 
  visibles or hiddens and returns the conditional 
  probabilities"""

  cond_probs = get_cond_probs(to_cond, model, v_or_h)
  randoms = np.random.uniform(size=cond_probs.shape)

  return (randoms < cond_probs).astype(float)

def get_low_dim(num_pixels, num_hiddens, model, num_its):
  """Takes the number of pixels, the number of hiddens, a model
  and the number of iterations and generates a low dimensional"""

  hiddens = np.random.randint(0, 2, num_hiddens)
  time_steps = []

  for t in range(num_its):
    visibles = get_sample(hiddens, model, "v")
    hiddens = get_sample(visibles, model, "h")
    energy = get_energy(visibles, hiddens, model)
    time_step = (visibles, energy)

    time_steps.append(time_step)

  return time_steps

def get_random_model(num_hiddens, num_visibles):
  """Takes number of hiddens and number of visibles and 
  returns a model initialized to random numbers from a normal
  distribution with mean of 0 and standard deviation of 0.1"""

  std_dev = pow(0.1, 2)
  hidden_w = np.random.normal(0, std_dev, num_hiddens)
  visible_w = np.random.normal(0, std_dev, num_visibles)
  pair_w_shape = (num_visibles, num_hiddens)
  pair_w = np.random.normal(0, std_dev, pair_w_shape)

  return {"p" : pair_w, "v" : visible_w, "h" : hidden_w}

def get_pos_grad(batch, model):
  """Takes a training data batch and a model and returns 
  a positive gradient contribution"""

  pos_v_grad = np.sum(batch, axis=0)
  h_cond_probs = get_cond_probs(batch, model, "h")
  pos_h_grad = np.sum(h_cond_probs, axis=0)
  pos_p_grad = np.dot(batch.T, h_cond_probs)

  return {"v" : pos_v_grad, "h" : pos_h_grad, "p" : pos_p_grad}

def get_neg_grad(hiddens, model):
  """Takes hiddens and a model and returns a negative 
  gradient contribution"""

  visibles = get_sample(hiddens, model, "v")
  hiddens = get_sample(visibles, model, "h")
  neg_v_grad = np.sum(visibles, axis=0)
  h_cond_probs = get_cond_probs(visibles, model, "h")
  neg_h_grad = np.sum(h_cond_probs, axis=0)
  neg_p_grad = np.dot(visibles.T, h_cond_probs)
  grad_map = {"v" : neg_v_grad, 
              "h" : neg_h_grad, 
              "p" : neg_p_grad}

  return (grad_map, hiddens, visibles)

def get_updated_weights(old_weights, pos_grad, neg_grad, hyper_p):
  """Takes old weights, a positive gradient contribution, a negative 
  gradient contribution and a hyperparameter package and returns the 
  updated weights"""

  return old_weights + \
         hyper_p[0] * \
         (pos_grad.astype(float)/hyper_p[1] - \
          neg_grad.astype(float)/hyper_p[2] - \
          hyper_p[3] * old_weights)

def get_updated_model(old_model, pos_grad, neg_grad, hyper_p):
  """Takes an old model, a positive gradient contribution, a 
  negative gradient contribution and a hyperparameter package 
  and returns an updated model"""

  return {key :
          get_updated_weights(old_weights, 
                              pos_grad[key], 
                              neg_grad[key], 
                              hyper_p)
          for (key, old_weights) in old_model.items()}

def train(data, 
          num_its, 
          num_chains, 
          num_batches, 
          num_hiddens, 
          step_size, 
          reg):
  """Takes the training data, number of iterations, number of chains,
  number of batches, number of hiddens, step size and regularization 
  parameter and returns a trained model and the final state of the 
  visibles for the chains"""

  (num_datums, num_visibles) = data.shape
  hiddens = np.random.randint(0, 2, (num_chains, num_hiddens))
  batch_size = num_datums/num_batches
  hyper_p = (step_size, batch_size, num_chains, reg)
  model = get_random_model(num_hiddens, num_visibles)
  batches_shape = (num_batches, batch_size, num_visibles)
  batches = data.reshape(batches_shape)

  for t in range(num_its):
    for b in range(num_batches):
      print "Iteration:", t, "\tBatch:", b
      visibles = batches[b]
      pos_grad = get_pos_grad(visibles, model)
      neg_grad = get_neg_grad(hiddens, model)
      hiddens = neg_grad[1]
      model = get_updated_model(model, 
                                pos_grad, 
                                neg_grad[0], 
                                hyper_p)

  return (model, neg_grad[2])

def save_to_png(pixels, file_path):

  (height, width) = pixels.shape
  grid = (pixels * 255).astype(int).tolist()
  writer = png.Writer(width, height, greyscale=True)

  with open(file_path, 'wb') as png_file:
    writer.write(png_file, grid)

def make_square_grid(images):

  num_images = images.shape[0]
  dim = int(pow(num_images, 0.5))
  figure = pyplot.figure()
  grid = gridspec.GridSpec(dim,dim)
  subplots = []

  for i in range(num_images):
    subplots.append(pyplot.subplot(grid[i]))
    pyplot.tick_params(axis="both", 
                       which="both", 
                       bottom=False, 
                       top=False, 
                       left=False, 
                       right=False,
                       labelbottom=False,
                       labelleft=False)
    pyplot.axis('off')

  for i in range(num_images):
    subplots[i].imshow(images[i].reshape((28,28)), cmap=cm.Greys_r)

  pyplot.show()

def param_line2string(param_line):

  return reduce(lambda x,y: x+y, 
                [str(val) + " "
                 for val in param_line])

def save_model(model, visibles_path, hiddens_path, pair_path):

  visibles_string = param_line2string(model["v"].tolist())
  hiddens_string = param_line2string(model["h"].tolist())
  pair_string = reduce(lambda x,y: x+y,
                       [param_line2string(line) + "\n"
                        for line in model["p"].tolist()])
  file_info = [(visibles_string, visibles_path), 
               (hiddens_string, hiddens_path),
               (pair_string, pair_path)]

  for (string, path) in file_info:

    with open(path, 'w') as param_file:
      param_file.write(string)

def svm_light_write(hiddens, svm_light_path, label_path):

  with open(label_path) as label_file:
    labels = [line[0] for line in label_file]

  hiddens_list = hiddens.tolist()
  labeled_data = zip(labels, hiddens_list)
  data_strings = [reduce(lambda x,y: x+y, 
                         [" %d:%f" % (i+1, val)
                          for i, val in enumerate(datum)
                          if abs(val) > pow(10, -3)],
                         label) + "\n"
                  for (label, datum) in labeled_data]

  with open(svm_light_path, 'w') as svm_light_file:
    for data_string in data_strings:
      svm_light_file.write(data_string)

def main():

  #Make sure to change all of these directories to reflect the paths on the machine on which you are running this code!

  problems = set(["4b"])

  model_path = "Models/"
  data_path = "Data/"
  pair_path = model_path + "MNISTWP.txt"
  my_pair_path = model_path + "MYMNISTWP.txt"
  visible_path = model_path + "MNISTWC.txt"
  my_visible_path = model_path + "MYMNISTWC.txt"
  hidden_path = model_path + "MNISTWB.txt"
  my_hidden_path = model_path + "MYMNISTWB.txt"
  train_path = data_path + "MNISTXtrain.txt"
  test_path = data_path + "MNISTXtest.txt"
  train_label_path = data_path + "MNISTYtrain.txt"
  test_label_path = data_path + "MNISTYtest.txt"
  svm_light_dir = "Code/SVM/SVMlight/Binaries/Linux/"
  model = files2model(pair_path, visible_path, hidden_path)
  my_model = files2model(my_pair_path, my_visible_path, my_hidden_path)
  test_data = file2matrix(test_path)
  train_data = file2matrix(train_path)

  """QUESTION 2"""
  
  num_pixels = 784
  num_iterations = 500
  num_hiddens = 100

  if "2a" in problems:
    np.random.seed(0)

    a_time_steps = get_low_dim(num_pixels, 
                               num_hiddens, 
                               model, 
                               num_iterations)
    a_final_answer = [image
                      for index, (image, energy) in enumerate(a_time_steps)
                      if index % 5 == 0]

    make_square_grid(np.array(a_final_answer))
  
  if "2b" in problems or "2c" in problems:
    np.random.seed(0)

    num_chains = 100
    chains = [get_low_dim(num_pixels,
                          num_hiddens,
                          model,
                          num_iterations)
              for i in range(num_chains)]
  
    if "2b" in problems:
      final_images = np.array([chain[-1][0] for chain in chains])

      make_square_grid(final_images)

    if "2c" in problems:
      to_plot = [map(lambda x: x[1], chain)
                 for chain in chains[0:5]]
      it_list = range(500)

      pyplot.plot(it_list, to_plot[0], 'r-')
      pyplot.plot(it_list, to_plot[1], 'b-')
      pyplot.plot(it_list, to_plot[2], 'g-')
      pyplot.plot(it_list, to_plot[3], 'c-')
      pyplot.plot(it_list, to_plot[4], 'm-')
      pyplot.ylabel("Energy")
      pyplot.xlabel("Number of iterations")
      pyplot.show()


  """QUESTION 3"""
  num_its = 50
  num_batches = 100
  num_chains = 100
  num_hiddens = 400
  step_size = pow(10, -1)
  reg = pow(10, -4)

  if "3a" in problems or "3b" in problems or "4a" in problems:
    np.random.seed(1)

    #"""
    before = time.clock()
    (my_model, final_images) = train(train_data,
                                     num_its,
                                     num_chains,
                                     num_batches,
                                     num_hiddens,
                                     step_size,
                                     reg)
    after = time.clock()
    train_time = after - before

    print "Time:", train_time
    
    save_model(my_model, my_visible_path, my_hidden_path, my_pair_path)

    if "3a" in problems:

      make_square_grid(final_images)
    #"""

    if "3b" in problems:

      receptive_fields = my_model["p"].T
      make_square_grid(receptive_fields)

    if "4a" in problems:

      train_path = svm_light_dir + "SVMLightTrain.txt"
      test_path = svm_light_dir + "SVMLightTest.txt"
      train_hiddens = get_cond_probs(train_data, my_model, "ht")
      test_hiddens = get_cond_probs(test_data, my_model, "ht")

      svm_light_write(train_hiddens, train_path, train_label_path)
      svm_light_write(test_hiddens, test_path, test_label_path)

  if "4b" in problems:

    train_path = svm_light_dir + "SVMLightPixelTrain.txt"
    test_path = svm_light_dir + "SVMLightPixelTest.txt"
    
    svm_light_write(train_data, train_path, train_label_path)
    svm_light_write(test_data, test_path, test_label_path)

if __name__ == "__main__": main()
