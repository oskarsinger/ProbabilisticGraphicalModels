from numpy import *
from collections import defaultdict, OrderedDict
import glob

def populate_parent_matrix(vertex_index_map, parent_lists):
  """Takes a dict of dicts that represents each node mapped to its 
  parent list, returns a parent matrix paired with a dict that maps 
  each node to its index in the matrix"""

  num_vertices = len(parent_lists)
  parent_matrix = zeros((num_vertices, num_vertices))

  for (child, parents) in parent_lists.items():
    for parent in parents: 
      parent_index = vertex_index_map[parent]
      child_index = vertex_index_map[child] 
      parent_matrix[child_index, parent_index] = 1

  return parent_matrix

def get_graph_info_package(vertex_index_map, int_string_map, parent_lists):
  """Takes a dict that maps vertices to indeces, a dict of dicts that maps 
  vertices to the integer representations of their domain to the string 
  representations and a dict that maps vertices to topologically sorted 
  parent lists"""

  parent_matrix = populate_parent_matrix(vertex_index_map, parent_lists)  
  index_vertex_map = {index : 
                      vertex
                      for (vertex, index) in vertex_index_map.items()}

  return (vertex_index_map, index_vertex_map, int_string_map, parent_lists, parent_matrix)

def counts2probs(counts_map, parent_config_sums):
  """Takes a dict of dicts of dicts that maps vertices to parent
  configs to vertex values to counts and a dicts of dicts that maps
  vertices to parent configs to counts and returns a dict of dicts of
  dicts that maps vertices to parent configs to vertex values to 
  Laplace-smoothed probabilities"""

  cpt_map = defaultdict(lambda: {})

  for (vertex, counts) in counts_map.items():
    for (parent_config, value_map) in counts.items():
      config_sum = parent_config_sums[vertex][parent_config]
      cpt_map[vertex][parent_config] = defaultdict(lambda: 1/config_sum)
      for (value, count) in value_map.items(): 
        cpt_map[vertex][parent_config][value] = float(count)/config_sum
  
  return cpt_map

def data2counts(parent_configs):
  """Takes a dict mapping vertices to a list of pairs from the 
  training dataof a parent configuration and the vertex value 
  associated with that parent config in the datum and returns a 
  dict of dicts of dicts from vertices to parent configurations to 
  vertex values to counts, vertex_config_counts, paired with a dict 
  of dicts from vertices to parent configurations to counts, 
  parent_config_sums"""

  counts_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1)))
  parent_config_counts = defaultdict(lambda: defaultdict(int))

  for (vertex, config_list) in parent_configs.items():
    for (parent_config, vertex_value) in config_list:
      parent_config_counts[vertex][parent_config] += 1
      counts_map[vertex][parent_config][vertex_value] += 1

  return (counts_map, parent_config_counts)

def file2data(file_path):
  """Takes a list of file paths and returns a list of arrays, each
  containing the data from a line in one of the files"""

  data_file = open(file_path)

  return [array(map(int, line.split(","))) for line in data_file]

def parent_config2string(parent_array):
  """Takes an array of the parent configuration and returns a string 
  of the values"""

  return reduce(lambda x,y: x + y, [str(int(i)) for i in parent_array])

def cpts2sorted_cpts(model):
  """Takes a dict that maps vertices to cpts and returns the dict 
  with the cpt entries sorted in ascending order by the int values of 
  their """
  
  parent_configs = [parent_config 
                    for (vertex, cpt) in model.items() 
                    for parent_config in cpt.keys()]

  return {vertex : 
          OrderedDict(
            sorted(cpt.items(), 
                   key = lambda (parent_config, other): int(parent_config[::-1]))) 
          for (vertex, cpt) in model.items()}

def file2model(file_path, graph_info_package):
  """Takes a list of training file paths and the graph info package and
  returns a trained model in the form of a dict of dicts of dicts that maps 
  vertices to parent configs to vertex values to probabilities"""

  vertex_index_map = graph_info_package[0]
  parent_matrix = graph_info_package[4]
  data = file2data(file_path)

  #I know, not very readable, but it runs faster that a loop! Explanation:
  #Maps each vertex to a list of pairs of parent config and vertex value from the provided training data
  parent_configs = {vertex : 
                    [(parent_config2string(parent_matrix[index] * datum), datum[index]) 
                     for datum in data] 
                    for (vertex, index) in vertex_index_map.items()}
  (counts_map, parent_config_counts) = data2counts(parent_configs)

  trained_model = counts2probs(counts_map, parent_config_counts)

  return cpts2sorted_cpts(trained_model)

def files2models(file_paths, graph_info_package):
  """Takes a list of file paths and a graph info package and returns 
  a dict that maps each file path to a model trained on that file path"""

  return {file_path : 
          file2model(file_path, graph_info_package) 
          for file_path in file_paths}

def cpt_row2string(vertex, row, graph_info_package, delimiter):
  """Takes a vertex, a 3-tuple of row info, the graph info package 
  and a delimiter and returns a row string for a readable CPT print"""

  index_vertex_map = graph_info_package[1]
  int_string_map = graph_info_package[2]
  (parent_config, vertex_val, prob) = row
  print prob, vertex, vertex_val
  row_string = "{0:<10.5f}{1}{2}{1}".format(prob, delimiter, int_string_map[vertex][vertex_val])
  parent_values = list(parent_config)
  parent_value_strings = [int_string_map[index_vertex_map[i]][int(parent_values[i])]
                          for i in range(len(parent_values)) 
                          if parent_values[i] != '0']

  parent_config_string = delimiter.join(parent_value_strings) + delimiter if parent_value_strings else ""

  return row_string + parent_config_string

def cpt2string(vertex, cpt, graph_info_package, delimiter):
  """Takes a vertex, that vertex's CPT, the graph info package and 
  a delimiter and returns a string of the CPT"""

  parent_lists = graph_info_package[3]
  parent_list = parent_lists[vertex]
  has_parents = len(parent_list) > 0
  row_info = [(parent_config, vertex_val, prob)
              for (parent_config, vertex_prob_map) in cpt.items() 
              for (vertex_val, prob) in vertex_prob_map.items()]
  dependencies = "|" + ",".join(parent_list) if has_parents else ""
  cpt_header_string = "{0:<10}".format("P({}{})".format(vertex, dependencies)) + delimiter + \
                      vertex + delimiter + (delimiter.join(parent_list) + delimiter if has_parents else "")
  row_strings = [cpt_row2string(vertex, row, graph_info_package, delimiter) 
                 for row in row_info]

  return "\n".join([cpt_header_string] + row_strings) + "\n"

def cpt_map2strings(trained_model, graph_info_package, delimiter="\t|\t"):
  """Takes a model, the graph info package and a delimiter and returns 
  a map from vertices to their CPTs' nice looking print strings"""

  index_vertex_map = graph_info_package[1]
  parent_lists = graph_info_package[3]
  parents_and_models = [(vertex, parent_lists[vertex], trained_model[vertex]) 
                        for vertex in index_vertex_map.values()]

  return {vertex : 
          cpt2string(vertex, cpt, graph_info_package, delimiter)
          for (vertex, parent_list, cpt) in parents_and_models}

def calculate_numerators(model, query, datum, graph_info_package):
  """Takes a model, a query variable, a data instance vector, the graph
  info package and the parent matrix and returns the numerators for the CPT
  values for the query variable given the values in the data instance vector"""

  (vertex_index_map, 
   index_vertex_map, 
   int_string_map, 
   parent_lists, 
   parent_matrix) = graph_info_package
  query_index = vertex_index_map[query]
  query_vals = int_string_map[query].keys()
  query_parent_config_string = parent_config2string(parent_matrix[query_index] * datum)
  query_model_entry = model[query][query_parent_config_string]
  relevant_vertices = {vertex 
                       for (vertex, parent_list) in parent_lists.items()
                       if query in parent_list}
  model_lookup_info = [(vertex, parent_matrix[index] * datum, datum[index])
                        for (vertex, index) in vertex_index_map.items()
                        if vertex in relevant_vertices]
  query_val_prob_map = defaultdict(lambda: {})

  for val in query_vals:
    for (vertex, parent_config, vertex_val) in model_lookup_info:
      new_parent_config = copy(parent_config)
      new_parent_config[query_index] = val
      new_parent_config_string = parent_config2string(new_parent_config)
      query_val_prob_map[val][vertex] = model[vertex][new_parent_config_string][vertex_val]

  return {val : 
          reduce(lambda x,y: x*y, query_val_prob_map[val].values(), query_model_entry[val])
          for val in query_vals}

def get_query_cpt(model, query, datum, graph_info_package):
  """Takes a model, a query variable, a data instance vector, a graph info 
  package and a parent matrix and returns the CPT for the queried variable given 
  the data instance vector"""
  
  vertex_index_map = graph_info_package[0]
  parent_matrix = graph_info_package[4]
  query_index = vertex_index_map[query]
  numerators = calculate_numerators(model, query, datum, graph_info_package)
  denominator = sum(numerators.values())

  return {val :
          float(numerator)/denominator
          for (val, numerator) in numerators.items()}

def calculate_probability_query(model, query, datum, graph_info_package, unobserved_vars=None):
  """Takes the same arguments as calculate_probability_query except for an 
  additional argument, unobserved_vars, and returns the cpts for each of the
  new configurations resulting from the unobserved vars"""

  if unobserved_vars is None:
    unobserved_vars = []
  vertex_index_map = graph_info_package[0]
  int_string_map = graph_info_package[2]
  parent_matrix = graph_info_package[4]
  query_index = vertex_index_map[query]
  parent_configs = [datum]

  for unobserved in unobserved_vars:
    unobserved_index = vertex_index_map[unobserved]
    temp_list = parent_configs
    parent_configs = []
    for val in int_string_map[unobserved].keys():
      for parent_config in temp_list:
        new_config = copy(parent_config)
        new_config[unobserved_index] = val
        parent_configs.append(new_config)

  return {parent_config2string(parent_matrix[query_index] * parent_config) :
          get_query_cpt(model, query, parent_config, graph_info_package)
          for parent_config in parent_configs}

def calculate_map_query(model, query, datum, graph_info_package, unobserved_vars = None):
  """Takes a model, a query variable, a data instance vector, a graph info 
  package and a parent matrix and returns the argmax for the CPT of the queried
  variable"""
  
  int_string_map = graph_info_package[2]
  query_cpt = calculate_probability_query(model, query, datum, graph_info_package, unobserved_vars)
  query_result_list = {val : 
                       prob
                       for value_prob_map in query_cpt.values()
                       for (val, prob) in value_prob_map.items()}
  sorted_cpt = sorted(query_cpt.items(), key=lambda (val, prob): prob)
  argmax = sorted_cpt[-1][0]
  
  return int_string_map[query][argmax]

def datum2true_value(datum, query, graph_info_package):
  """Takes a data instance vector, a query variable and a graph info 
  package and returns the string representation of the true value of 
  the queried variable in the data instance vector"""

  vertex_index_map = graph_info_package[0]
  int_string_map = graph_info_package[2]
  query_index = vertex_index_map[query]
  datum_query_val = datum[query_index]

  return int_string_map[query][datum_query_val]

def calculate_accuracy(model, testing_file_path, query, graph_info_package):
  """Takes a model, a file path for a test set, a query variable, a 
  graph info package and a parent matrix and calculates the accuracy 
  of the model on the test set for the given query"""

  testing_data = file2data(testing_file_path)
  prediction_gold_list = [(calculate_map_query(model, query, datum, graph_info_package), 
                           datum2true_value(datum, query, graph_info_package))
                            for datum in testing_data] 
  correctness_list = map(lambda pair: 1 if pair[0] == pair[1] else 0, prediction_gold_list)
  num_correct = reduce(lambda x,y: x + y, correctness_list)

  return float(num_correct)/float(len(correctness_list))

def calculate_accuracies(model_data_list, query, graph_info_package):
  """Takes a list of models mapped to test data, a query variable, a 
  graph info package and a parent matrix and returns a list of accuracies"""

  return {training_file_path : 
          calculate_accuracy(model, testing_file_path, query, graph_info_package) 
          for (training_file_path, model, testing_file_path) in model_data_list}
   
def accuracies2stats(model_stats_map):
  """Takes a list of accuracies and returns a 2-tuple of the mean
  and standard deviation for the given accuracies"""

  accuracies = model_stats_map.values()
  mean = float(sum(accuracies))/float(len(accuracies))
  mean_differences_squared = [(accuracy - mean)**2 
                              for accuracy in accuracies]
  variance = float(sum(mean_differences_squared))/len(mean_differences_squared)
  standard_deviation = variance**(0.5)

  return (mean, standard_deviation)

def models2stats(model_data_list, query, graph_info_package):
  """Takes a list of models paired with test data paths, a query variable,
  a graph info package and a parent matrix and returns a 2-tuple of the 
  mean model accuracy and standard deviation of the accuracies"""

  model_stats_map = calculate_accuracies(model_data_list, query, graph_info_package)
  
  return accuracies2stats(model_stats_map)

def main():
  """Here are some usage examples for the Bayesian Network suite"""

  data_dir_path = "/home/oskar/GitRepos/ProbabilisticGraphicalModels/BayesNets/Data/"
  question_4_output_path = "params.txt"
  question_5_output_path = "queries.txt"
  question_4 = False
  question_5 = True
  questions_6_and_7 = False

  a = 'A'
  g = 'G'
  cp = 'CP'
  bp = 'BP'
  ch = 'CH'
  ecg = 'ECG'
  hr = 'HR'
  eia = 'EIA'
  hd = 'HD'
  vertex_index_map = {a : 0, g : 1, cp : 2, bp : 3, ch : 4, ecg : 5, hr : 6, eia : 7, hd : 8}
  int_string_map = {a : {1 : "< 45", 2 : "45 - 55", 3 : ">= 55"}, 
                    g : {1 : "Female", 2 : "Male"}, 
                    cp : {1 : "T", 2 : "A", 3 : "NA", 4 : "NO"}, 
                    bp : {1 : "Low", 2 : "High"},
                    ch : {1 : "Low", 2 : "High"},
                    ecg : {1: "N", 2 : "A"},
                    hr : {1 : "Low", 2 : "High"},
                    eia : {1 : "N", 2 : "Y"},
                    hd : {1 : "N", 2 : "Y"}}

  parent_lists = {a : [], g : [], cp : [hd], bp : [g], ch : [a,g], ecg : [hd], hr : [a,bp,hd], eia : [hd], hd : [bp,ch]}
  graph_info_package = get_graph_info_package(vertex_index_map, int_string_map, parent_lists)

  if question_4 or question_5:
    (vertex_index_map, 
     index_vertex_map, 
     int_string_map, 
     parent_lists, 
     parent_matrix) = graph_info_package
    file_path = data_dir_path + "data-train-1.txt"
    model = file2model(file_path, graph_info_package)

    if question_4:
      cpt_strings = cpt_map2strings(model, graph_info_package)
      cpt_file = open(question_4_output_path, "w")

      cpt_file.write(cpt_strings[a] + "\n")
      cpt_file.write(cpt_strings[bp] + "\n")
      cpt_file.write(cpt_strings[hd] + "\n")
      cpt_file.write(cpt_strings[hr] + "\n")
    
    """Query A has no unobserved variables other than the query variable. Query B has one unobserved variable."""
    if question_5:
      delimiter = "\t|\t"
      query_results_file = open(question_5_output_path, "w")
      query_a = ch
      query_b = bp
      unobserved = [g]
      datum_a = array([2,2,4,1,0,1,1,1,1])
      datum_b = array([2,0,1,0,2,1,2,2,1])
      query_a_results = calculate_probability_query(model, query_a, datum_a, graph_info_package)
      query_b_results = calculate_probability_query(model, query_b, datum_b, graph_info_package, unobserved)
      query_a_results_string = cpt2string(query_a, query_a_results, graph_info_package, delimiter)
      query_b_results_string = cpt2string(query_b, query_b_results, graph_info_package, delimiter)

      query_results_file.write(query_a_results_string + "\n")
      query_results_file.write(query_b_results_string + "\n")

  """All of the file retrieval is specific to my personal machine's dirs and my testing file names"""
  if questions_6_and_7:
    file_path_model_map = files2models(glob.glob(data_dir_path + 'data-train-[12345].txt'), graph_info_package)
    test_file_path_map = {file_path[-4] :
                          file_path
                          for file_path in glob.glob(data_dir_path + 'data-test-[12345].txt')}
    model_data_list = [(training_file_path, model, test_file_path_map[training_file_path[-4]])
                                for (training_file_path, model) in file_path_model_map.items()]
    (mean, std_dev) = models2stats(model_data_list, hd, graph_info_package)

    print "Accuracy Mean:", str(mean), "\nStandard Deviation:", str(std_dev)

if __name__ == '__main__': main()
