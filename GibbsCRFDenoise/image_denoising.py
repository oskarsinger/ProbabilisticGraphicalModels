import numpy as np
import matplotlib.pyplot as plt
import random, math, png, operator
from scipy.misc import logsumexp
from collections import OrderedDict

def file2matrix(file_path):
  """Takes a file path and returns a 
  matrix of the values in the text file 
  at the given path"""

  with open(file_path) as data_file:
    value_grid = [[float(value) for value in line.split()]
                  for line in data_file]

  return np.array(value_grid)

def d_get_cond_prob(neighbors, pixel, weights):
  """Takes a list of neighbor values, a pixel value 
  and a weights pair and returns the probability of 
  that pixel taking value 1 conditioned on its markov 
  blanket"""

  (trans_weight, obs_weight) = weights
  neighbor_weights = [[(1 if n == val else 0)*trans_weight
                       for n in neighbors]
                      for val in range(2)]
  terms = [n_list + [(1 if pixel == val else 0)*obs_weight]
           for val, n_list in enumerate(neighbor_weights)]
  summed_terms = [sum(term_list)
                  for val, term_list in enumerate(terms)]

  return math.exp(summed_terms[1] - logsumexp(summed_terms))

def get_neighbors(pixel_matrix, i, j):
  """Takes an image and pixel coordinates 
  i, j and returns a list of the values of 
  the existing neighbors of the pixel at 
  i, j"""

  (height, width) = pixel_matrix.shape
  top = -1 if i == 0 else pixel_matrix[i-1,j]
  bottom = -1 if i == height-1 else pixel_matrix[i+1,j]
  left = -1 if j == 0 else pixel_matrix[i,j-1]
  right = -1 if j == width-1 else pixel_matrix[i,j+1]

  return filter(lambda x: x >= 0, [top, bottom, left, right])

def d_get_pixel_sample(pixel_matrix, i, j, weights):
  """Takes an image, pixel coordinates i, j and a 
  weights pair and returns a new pixel value sampled 
  from the distribution on the value of the pixel at 
  i, j conditioned on its markov blanket"""
  
  pixel = pixel_matrix[i,j]
  neighbors = get_neighbors(pixel_matrix, i, j)
  cond_prob = d_get_cond_prob(neighbors, pixel, weights)
  
  return (1 if random.random() < cond_prob else 0)

def get_variance(weights, num_neighbors):
  """Takes a weights pair and the number of neighbors 
  and returns the variance of the distribution on a 
  pixel value conditioned on its markov blanket"""

  (trans_weight, obs_weight) = weights

  return 1/(2*((trans_weight * num_neighbors) + obs_weight))

def get_mean(neighbors, pixel, weights):
  """Takes a list of neighbor values, a pixel 
  value and a weights pair and returns the mean 
  of the distribution on the value of the pixel 
  at i, j conditioned on its markove blanket"""

  (trans_weight, obs_weight) = weights
  neighbor_weights = [trans_weight*n
                      for n in neighbors]
  all_weights = neighbor_weights + [obs_weight*pixel]
  denominator = (trans_weight * len(neighbors)) + obs_weight

  return sum(all_weights)/denominator

def g_get_pixel_sample(pixel_matrix, i, j, weights):
  """Takes an image, pixel coordinates i, j and a 
  weights pair and returns a new pixel value from the 
  gaussian distribution of the pixel at i, j conditioned 
  on its markov blanket"""

  pixel = pixel_matrix[i,j]
  neighbors = get_neighbors(pixel_matrix, i, j)
  variance = get_variance(weights, len(neighbors))
  mean = get_mean(neighbors, pixel, weights)

  return mean + random.gauss(0,1)*pow(variance, 0.5)

def get_image_sample(pixel_matrix, weights, d_or_g = "d"):
  """Takes an image, a weights pair and a flag for 
  discrete or gaussian image distribution and returns 
  a new image resulting from one round of the Gibbs sampler"""

  (height, width) = pixel_matrix.shape
  new_sample = np.copy(pixel_matrix)
  sample_func_map = {"d" : d_get_pixel_sample,
                     "g" : g_get_pixel_sample}

  for i in range(height):
    for j in range(width):
      new_sample[i,j] = sample_func_map[d_or_g](pixel_matrix,
                                                i,
                                                j,
                                                weights)

  return new_sample

def get_MAE(sampled, true):
  """Takes a sampled image and a true image and 
  returns the pixel-wise error of the sample image"""

  (height, width) = sampled.shape
  denominator = height * width
  summed_error = np.sum(np.absolute(sampled - true))

  return float(summed_error)/float(denominator)

def denoise(noisy, true, weights, num_iterations, d_or_g = "d"):
  """Takes a noisy image, a true image, a weights pair, 
  a number of iterations and a flag for discrete or gaussian 
  image distribution, runs the Gibbs sampler for the given 
  number of iterations and returns the posterior mean image 
  of the sum of samples at each time step"""

  current = get_image_sample(noisy, weights, d_or_g)
  accumulator = np.copy(current)

  for i in range(num_iterations-1):
    print "\tIterations:", str(i+1), get_MAE(accumulator/(i+1), true)
    current = get_image_sample(current, weights, d_or_g)
    accumulator = accumulator + current

  return accumulator.astype(float) / float(num_iterations)

def run_until_convergence(noisy, true, weights, epsilon, d_or_g = "d"):
  """Takes a noisy image, a true image, a weights pair, an error 
  difference threshold and a flag for discrete or gaussian image 
  distribution, runs the Gibbs sampler until the error difference between 
  consecutive steps does not exceed epsilon for 3 consecutive time steps 
  and returns a list of time steps paired with the MAE of the posterior 
  mean image at that step"""

  current = get_image_sample(noisy, weights, d_or_g)
  prev_MAE = get_MAE(current, true)
  t = 1
  epsilon_count = 0
  step_stats = [(t, prev_MAE)]
  accumulator = current

  while True:
    t += 1
    current = get_image_sample(current, weights, d_or_g)
    accumulator = accumulator + current
    current_MAE = get_MAE(accumulator/t, true)

    step_stats.append((t, current_MAE))

    if prev_MAE - current_MAE <= epsilon:
      epsilon_count += 1
    elif epsilon_count > 0:
      epsilon_count = 0

    if epsilon_count > 2:
      break
    else:
      prev_MAE = current_MAE

  return step_stats

def save_to_png(pixels, file_path):
  """Takes a pixel matrix and a file path and 
  saves the grayscale image based on the pixels 
  to a png file named with the file path"""

  (height, width) = pixels.shape
  grid = (pixels * 255).astype(int).tolist()
  writer = png.Writer(width, height, greyscale=True)

  with open(file_path, 'wb') as png_file:
    writer.write(png_file, grid)

def get_top_and_right(width, i, j):
  """Takes the width of the image and pixel 
  coordinates i, j and returns the pixel's top 
  and right neighbors"""

  top = None if i == 0 else (i-1, j)
  right = None if j == width-1 else (i, j+1)

  return filter(lambda x: x is not None, [top, right])

def get_weight_key(i, j, k, l):
  """Takes pixel coordinates i, j and pixel 
  coordinates k, l and returns the key to access 
  the weight coresponding to the edge between those 
  pixels"""

  ij = str(i)+str(j)
  kl = str(k)+str(l)

  return frozenset([ij, kl])

def g_get_g_weights(noisy, trans_weight):
  """Takes a noisy image and an initial pairwise weight
  and returns a map from index pairs ij and kl to weights"""

  (height, width) = noisy.shape

  return {get_weight_key(i ,j , k , l) :
          trans_weight / \
          (0.01+pow(noisy[i,j]-noisy[k,l], 2))
          for i in range(height)
          for j in range(width)
          for (k,l) in get_top_and_right(width, i, j)}

def get_neighbor_indeces(i,j, shape):
  """Takes pixel coordinates i and j and the size 
  of the image and returns a list of the the indeces 
  of the existing neighbors of the pixel at i,j"""

  (height, width) = shape
  top = None if i == 0 else (i-1, j)
  bottom = None if i == height-1 else (i+1, j)
  left = None if j == 0 else (i, j-1)
  right = None if j == width-1 else (i, j+1)
  neighbors = [top, bottom, left, right]

  return filter(lambda x: x is not None, neighbors)

def get_g_variance(noisy, i, j, neighbors, weights, gg_weights):
  """Takes a noisy image, pixel coordinates i and j, a list of 
  neighbor indeces, a weights pair and generalized pairwise weights 
  map and returns the variance for the distribution for the value of 
  the pixel at i, j conditioned on its markov blanket"""

  (trans_weight, obs_weight) = weights

  pairwise_weights = [gg_weights[get_weight_key(i,j,k,l)]
                      for (k,l) in neighbors]
  return 1/(2*(sum(pairwise_weights) + obs_weight))

def get_g_mean(noisy, i, j, neighbors, weights, gg_weights):
  """Takes a noisy image, pixel coordinates i and j, a list 
  of neighbor coordinates, a weights pair and a generalized 
  pairwise weights map and returns the mean of the distribution 
  for the value of the pixel at i, j conditioned on its markov 
  blankent"""
 
  (trans_weight, obs_weight) = weights
  neighbor_weights = [gg_weights[get_weight_key(i,j,k,l)]
                      for (k,l) in neighbors]
  weighted_neighbors = [noisy[k,l] * weight
                        for (weight, (k,l)) 
                        in zip(neighbor_weights, neighbors)]
  all_weights = weighted_neighbors + [noisy[i,j]*obs_weight]
  denominator = sum(neighbor_weights) + obs_weight

  return sum(all_weights)/denominator

def g_get_g_pixel_sample(noisy, i, j, weights, gg_weights):
  """Takes a noisy image, pixel coordinates i and j, a weights 
  pair and a generalized pairwise weights map and returns the sample 
  for the pixel at i, j conditioned on its Markov blanket"""

  neighbors = get_neighbor_indeces(i, j, noisy.shape)
  variance = get_g_variance(noisy, i, j, neighbors, weights, gg_weights)
  mean = get_g_mean(noisy, i, j, neighbors, weights, gg_weights)

  return mean + random.gauss(0,1)*pow(variance, 0.5)

def get_g_image_sample(pixel_matrix, weights, gg_weights):
  """Takes a noisy image, a weights pair and a generalized 
  pairwise weights map and returns the new image after one 
  round of the Gibbs sampler"""

  (height, width) = pixel_matrix.shape
  new_sample = np.copy(pixel_matrix)

  for i in range(height):
    for j in range(width):
      new_sample[i,j] = g_get_g_pixel_sample(new_sample,
                                             i,
                                             j,
                                             weights,
                                             gg_weights)

  return new_sample

def gg_denoise(noisy, true, weights, num_iterations):
  """Takes a noisy image, a true image, a weights pair and 
  a number of iterations and returns a denoised image based 
  on the generalized pairwise parameter equation"""

  gg_weights = g_get_g_weights(noisy, weights[0])
  current = get_g_image_sample(noisy, weights, gg_weights)
  accumulator = np.copy(current)

  for t in range(num_iterations):
    print "\tIteration:", t, get_MAE(current, true)
    current = get_g_image_sample(current, weights, gg_weights)
    accumulator = accumulator + current

  return accumulator / num_iterations

def plot_stats(step_stats):

  sorted_stats = OrderedDict(sorted(step_stats, 
                                    key = operator.itemgetter(0), 
                                    reverse=True))

  plt.plot(sorted_stats.keys(), sorted_stats.values())

  point_string = "x={0}, y={1:.2e}"
  
  plt.ylabel("Number of iterations")
  plt.xlabel("MAE")
  plt.show()


def main():

  #To reproduce my results, add to the set "problems" the number and 
  #letter of the question that you would like to reproduce, for example "1a"
  #To reproduce the png's and the plot, add "png" and "plt" respectively IN ADDITION 
  #to the numbers of the questions you would like to see reproduced

  problems = set(["1a"])

  data_path = "Data/"
  d_noisy_path = data_path + "stripes-noise.txt"
  d_true_path = data_path + "stripes.txt"
  g_noisy_path = data_path + "swirl-noise.txt"
  g_true_path = data_path + "swirl.txt"
  num_iterations = 100
  epsilon = pow(10, -4)

  """Question 1"""

  d_true = file2matrix(d_true_path)
  d_noisy = file2matrix(d_noisy_path)
  d_weights = (999, 1000)

  """1.a"""
  if "1a" in problems:
    random.seed(1)
    print "Discreet denoising for", num_iterations, "iterations:"
    
    weights_map = {"good" : d_weights, 
                   "over" : (.9, 1), 
                   "under" : (48, 100)}
    d_denoised = {key :
                  denoise(d_noisy, d_true, d_weights, num_iterations)
                  for (key, d_weights) in weights_map.items()}

    if "png" in problems:
      for (key, pixels) in d_denoised.items():
        save_to_png(pixels, key+"-stripes-denoised.png")

  """1.b"""
  if "1b" in problems:
    random.seed(1)
    print "Discrete denoising until convergence:\n", \
          "Error Difference Margin:", epsilon

    d_step_stats = run_until_convergence(d_noisy, 
                                         d_true, 
                                         d_weights, 
                                         epsilon)
    print "Step stats for iteration to convergence:\n", d_step_stats

    if "plt" in problems:
      plot_stats(d_step_stats)

  """Question 2"""

  g_true = file2matrix(g_true_path)
  g_noisy = file2matrix(g_noisy_path)

  """2.a"""
  if "2a" in problems:
    g_weights = (pow(10, 1), 26*pow(10, 2))
    random.seed(1)
    print "Gaussian denoising for", num_iterations, "iterations:"

    g_denoised = denoise(g_noisy,
                         g_true,
                         g_weights,
                         num_iterations,
                         "g")

    if "png" in problems:
      save_to_png(g_denoised, "swirl-denoised.png")

  """2.c"""
  if "2c" in problems:
    gg_weights = (pow(10, 1), 99*pow(10, 3))
    random.seed(1)
    print "Generalized Gaussian denoising for", num_iterations, "iterations:"

    gg_denoised = gg_denoise(g_noisy,
                             g_true,
                             gg_weights,
                             num_iterations)

    if "png" in problems:
      save_to_png(gg_denoised, "swirl-generalized-denoised.png")

if __name__ == '__main__': main()
