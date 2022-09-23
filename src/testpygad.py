import pygad
import numpy as np
from fusionData import fusion
import testeClassBalance as tc
import pandas as pd
from classifiers import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import testPreProcessing as tp

from sklearn.svm import SVC

x,y = tc.balance()

#print(type(x))
#print(type(y))



#a = pd.DataFrame(tp.prepro(x))
#solution
b = fusion.creatingMatrix(x)


#print(type(b))
#print(b.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

X_train = X_train[:1000]
X_test = X_test[:1000]
y_train = y_train[:1000]
y_test = y_test[:1000]
y_train.ravel()

#teste = X_train[:, 6]
lines, columns = X_train.shape
linesTest, columnsTest = X_test.shape
svm = SVC(kernel="linear", random_state=1, C=1.0, probability=True)
function_inputs = b
desired_output = 1.0

#fit = svm.fit(X_train, y_train)
#predictions = fit.predict(X_test)
#accuracy = accuracy_score(y_test, predictions)

def on_start(ga_instance):
    print("on_start()")

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
#       output = solution*function_inputs
#       teste = output * c_x[:10000] #x_test
#       fitness = 1.0 / np.abs(output - desired_output)
#       svm_model,svc = svm.vectorMachine(teste, c_y[:10000])#y_train)
#       svm_predict = svm_model.predict(np.sum(output * c_x[:10000]))
#       fitness = accuracy_score(c_y[:10000],svm_predict)
    
    X_train_turbo = X_train
    X_test_turbo = X_test
    #aux = np.empty(X_train.shape)
    for i in range(columns):
        #print("Train ____"+ str(i) +"_____")
        for j in range(lines):
            
            X_train_turbo[j][i] = X_train_turbo[j][i] * solution[i]

    for i in range(columnsTest):
        #print("Test ____"+ str(i) +"_____")
        for j in range(linesTest):

            X_test_turbo[j][i] = X_test_turbo[j][i] * solution[i]
        
    #print("cheguei")
    fit = svm.fit(X_train_turbo, y_train)
    predictions = fit.predict(X_test_turbo)
    # print("HyperParameters")
    # print(solution)
    # print("F1_score-------------------")
    # print(f1_score(y_test, predictions))
    # print("fitness--------------------")
    fitness = f1_score(y_test, predictions, zero_division=1, average="micro")
    # print(classification_report(y_test, predictions))
    # print(fitness)
    return fitness

fitness_function = fitness_func

num_generations = 20 #50
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -1
init_range_high = 1

keep_elitism = 1

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
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop,
                       allow_duplicate_genes=False)


ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = np.sum(np.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
ga_instance.plot_fitness()
ga_instance.save("pygad_GA")