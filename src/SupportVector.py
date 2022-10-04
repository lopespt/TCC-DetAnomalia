import pygad
import numpy as np
from fusionData import fusion
import testeClassBalance as tc
import pandas as pd
from classifiers import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import testPreProcessing as tp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import logging as log
import time
import uuid



gu_id = uuid.uuid1()
iid = str(gu_id)
log.basicConfig(filename="../src/logs/log/svm/"+ str(gu_id) + ".log", filemode="a", format= "%(asctime)s \n %(message)s",datefmt='%d-%b-%y %H:%M:%S')
X_train, X_test, y_train, y_test,X_test_ga,y_test_ga = tc.balance()
count = 0
population = fusion.creatingMatrix(X_train)
inicio=0
tempo = []
# X_train = X_train[:1000]
# X_test = X_test[:1000]
# y_train = y_train[:1000]
# y_test = y_test[:1000]
# X_test_ga = X_test_ga[:1000]
# y_test_ga = y_test_ga[:1000]
y_train.ravel()

lines, columns = X_train.shape
linesTest, columnsTest = X_test.shape
svm = SVC(kernel="linear", C=1.0, probability=True)
function_inputs = population
desired_output = 1.0


def on_start(ga_instance):
      print("--------------SVM----------------------")
      print("on_start()")
      print(iid)
      
      print("-------------------------------")

def on_fitness(ga_instance, population_fitness):
    print("Fitness of the solution :", ga_instance.best_solution()[1])
    
    fim = time.time()
    log.warning("O tempo de fitness é:"+ str((fim - inicio) % 60))
    print("O tempo de fitness é: " + str((fim - inicio) % 60))
    tempo.append((fim - inicio) % 60)
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    array = "Parameters of the pesos \n {solution}".format(solution=solution)
    print("Generation : ", ga_instance.generations_completed)
    log.warning(array)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    logg = "Fitness of the generation solution "+ str(ga_instance.generations_completed) +": " + str(ga_instance.best_solution()[1])
    log.warning(logg)
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

# def callback_gen(ga_instance):
    
#     solution, solution_fitness, solution_idx = ga_instance.best_solution()
#     array = "Parameters of the pesos \n {solution}".format(solution=solution)
#     print("Generation : ", ga_instance.generations_completed)
#     log.warning(array)
#     print("Fitness of the best solution :", ga_instance.best_solution()[1])
#     logg = "Fitness of the generation solution "+ str(ga_instance.generations_completed) +": " + str(ga_instance.best_solution()[1])
#     log.warning(logg)
    
    

def fitness_func(solution, solution_idx):
    inicio = time.time()
    X_train_turbo = X_train
    X_test_turbo = X_test


    X_train_turbo = np.multiply(X_train_turbo,solution.reshape(1,-1))


    X_test_turbo = np.multiply(X_test_turbo,solution.reshape(1,-1))

    
    fit = svm.fit(X_train_turbo, y_train)
    predictions = fit.predict(X_test_turbo)
   
    fitness = f1_score(y_test, predictions, zero_division=1, average="micro")

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
                       #callback_generation=callback_gen,
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

    array_de_pesos = "Parameters of the best solution \n {solution}".format(solution=solution)
    print(array_de_pesos)
    log.warning(array_de_pesos)
    melhor_f1 = "Fitness value of the best solution(f1) \n  {solution_fitness}".format(solution_fitness=solution_fitness)
    print(melhor_f1)
    log.warning(melhor_f1)
    
    log.warning("O tempo medio foi: " + str(np.median(tempo)) +  " Segundos")

    X_train_turbo_ga = np.multiply(X_train,solution.reshape(1,-1))


    X_test_turbo_ga = np.multiply(X_test_ga,solution.reshape(1,-1))

    fit = svm.fit(X_test_turbo_ga, y_train)

    prediction = fit.predict(X_test_turbo_ga) # aqui vai os 10% do 2 teste

    puntuacao = f1_score(y_test_ga, prediction, zero_division=1, average="micro")

    temp = "Predicted output based on the best solution : {puntuacao}".format(puntuacao=puntuacao)
    print(confusion_matrix(y_test_ga,prediction))
    print(classification_report(y_test_ga,prediction))
    print(accuracy_score(y_test_ga,prediction))
    log.warning(confusion_matrix(y_test_ga,prediction))
    log.warning(classification_report(y_test_ga,prediction))
    log.warning(accuracy_score(y_test_ga,prediction))
    print(temp)
    log.warning(temp)

    plt.figure()
    plt.plot(ga_instance.best_solutions_fitness)
    plt.savefig("../src/logs/png/svm/svm_png_" + str(gu_id))
    #plt.show()
    ga_instance.save("../src/saves/svm/svm_GA_" + str(gu_id))