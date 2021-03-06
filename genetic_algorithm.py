# imports
from numpy import random
import numpy as np

#! wild card import is not good practise
from function_for_optimization import *
from animation import *

def binarToDecim(binary_list):
    decimal_number = 0
    for index, binary in enumerate (binary_list[:: -1]):
        decimal_number += 2**index*int(binary)
    return decimal_number

def nullGeneration(num_individuals,string_len,values,num_coordinates): 
    # Duplicate values are possible
    nullgeneration = random.randint(values, size=(num_individuals,num_coordinates,string_len))

    return nullgeneration

def evaluation(individuals, num_individuals=10): 
    valued_individuals = np.array([],dtype=object) 
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    for individual in individuals:      
        decimal_index_x = binarToDecim(individual[0])
        decimal_index_y = binarToDecim(individual[1])

        #coordinates of an individual on Rastrigin plot 
        x = xy_domain[decimal_index_x]
        y = xy_domain[decimal_index_y]
        z = RastriginFun( x,y ) 

        fitness = 1/z
        valued_individuals = np.append( np.array([individual,fitness],dtype=object,),valued_individuals )     

        # add individual for visualization
        x_coordinates.append(x)
        y_coordinates.append(y)
        z_coordinates.append(z)

    valued_individuals = valued_individuals.reshape(num_individuals,2) #! can I do it without reshaping?
    return valued_individuals,x_coordinates,y_coordinates,z_coordinates

def selection(valued_individuals,num_parents,num_individuals):
    sum_fitness = sum(valued_individuals[:,1])

    # propabality of individuals becoming a parent based on their fitness
    propability_individuals = valued_individuals[:,1]/(sum_fitness)
    
    #changing data type from float64 to float so that can be used as keyword "p" in random.choice
    propability_individuals = propability_individuals.astype('float')

    parents = np.random.choice(valued_individuals[0:num_individuals,0],size=(num_parents,2),p=propability_individuals )
    return parents
                                                                               
def recombine(prnt_1_bitstr,prnt_2_bitstr,point_recombin=4,string_len=8): 
    # function for recombining two binary strings, point_recombin stands for point of recombination

    child_x= np.array([prnt_1_bitstr[ :point_recombin],prnt_2_bitstr[point_recombin:]])
    child_y = np.array([prnt_2_bitstr[ :point_recombin],prnt_1_bitstr[point_recombin:]])

    child_x = child_x.reshape(string_len) 
    child_y = child_y.reshape(string_len)

    return [child_x,child_y]

def crossover(parents,string_len,point_recombin):
    children =  []

    # every couple generates two children
    for couple in parents:
        child_1 = recombine(couple[0][0],couple[1][0],point_recombin,string_len)
        child_2 = recombine(couple[0][1],couple[1][1],point_recombin,string_len)

        children.append(child_1) 
        children.append(child_2)
    
    children = np.array(children)
    return children

def mutation(children,indi_to_mutate,num_children,string_len,num_coordinates): #!clean up
    
    for i in range (indi_to_mutate):
        child = random.randint(num_children) # pick one of the children
        coordinate = random.randint(num_coordinates) # pick X or Y coordinate index (cooded in bit string)
        gene = random.randint(string_len) # pick one gene (bit) to change

        #! this code only work with binary so the mutation negates
        if children[child,coordinate,gene] == 1:
            children[child,coordinate,gene] = 0
        else:
            children[child,coordinate,gene] = 1

    return children

def geneticAglorithm (cycles=30,num_individuals=40,
	num_parents=20,num_children=40,string_len=10,
	values=2,num_coordinates=2,point_recombin=5,indi_to_mutate=2):
    #incialization of the null gen
    generation = nullGeneration(num_individuals,string_len,values,
    							num_coordinates) 
    for cycle in range (cycles):
        generation,x_coordinates,y_coordinates,z_coordinates = evaluation(generation, num_individuals)
        # creating frames for visualization
        add_frames(x_coordinates,y_coordinates,z_coordinates,cycle,frames_all)

        generation = selection(generation,num_parents,
        	                   num_individuals)
        generation = crossover(generation,string_len,point_recombin)
        generation = mutation(generation,indi_to_mutate,
        	                  num_children,string_len,num_coordinates)

    visualization(frames_all,X,Y,Z,lower_bound,upper_bound)
    
geneticAglorithm()

