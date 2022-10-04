import pygad
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from fusionData import fusion
import testeClassBalance as tc
import pandas as pd
from classifiers import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import testPreProcessing as tp
from classifiers import tree, logisticReg, svm
from sklearn.svm import SVC
import uuid
import matplotlib.pyplot as plt


gu_id = uuid.uuid1()

X_train, X_test, y_train, y_test,X_test_ga,y_test_ga = tc.balance()

population = fusion.creatingMatrix(X_train)

print(gu_id)
X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
y_test = y_test[:1000]
y_train.ravel()

lines, columns = X_train.shape
linesTest, columnsTest = X_test.shape


randomForest = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)



function_inputs = population
desired_output = 1.0


def on_start(ga_instance):
      print("--------------RandomArvDecisão----------------------")
      print("on_start()")
      print("-------------------------------")

def on_fitness(ga_instance, population_fitness):
    print("Fitness of the solution :", ga_instance.best_solution()[1])
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

def fitness_func(solution, solution_idx):
    
      X_train_turbo = X_train
      X_test_turbo = X_test


      X_train_turbo = np.multiply(X_train_turbo,solution.reshape(1,-1))

      X_test_turbo = np.multiply(X_test_turbo,solution.reshape(1,-1))
      fit = randomForest.fit(X_train_turbo, y_train)
      predictions = fit.predict(X_test_turbo)
      fitness = f1_score(y_test, predictions, zero_division=1, average="micro")
      #print(confusion_matrix(y_test,predictions))
      #print(classification_report(y_test,predictions))
      #print(accuracy_score(y_test,predictions))
      return fitness

fitness_function = fitness_func

num_generations = 20 #50

sol_per_pop = 16

num_parents_mating = sol_per_pop//2


num_genes = len(function_inputs)

init_range_low = -1
init_range_high = 1

parent_selection_type = "rws"
keep_parents = -1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 25

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       callback_generation=callback_gen,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       #initial_population=population,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       #keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_elitism=1,
                       mutation_percent_genes=mutation_percent_genes,
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop,
                       allow_duplicate_genes=True)

ga_instance.population[0,:] = np.ones((1,ga_instance.population.shape[1]))
def main():

      
      print(ga_instance.population[0,:])
      ga_instance.run()

      solution, solution_fitness, solution_idx = ga_instance.best_solution()
      print("Parameters of the best solution : {solution}".format(solution=solution))
      print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

      prediction = np.sum(np.array(function_inputs)*solution)
      print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
      #ga_instance.plot_fitness(save_dir="../src/logs/png/dec_tree/dec_tree_png_" + str(gu_id))
      plt.figure()
      plt.plot(ga_instance.best_solutions_fitness)
      plt.savefig("../src/logs/png/rand_forest/rand_forest_png_" + str(gu_id))
      #plt.show()
      ga_instance.save("../src/saves/rand_forest/rand_forest_GA_" + str(gu_id))