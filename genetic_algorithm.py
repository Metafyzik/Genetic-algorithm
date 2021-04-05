
"""Building simple Genetic algorithm from sratch (more or less) and using it to find global extreme on Rastingin function. Using modul nympy for arrays and Plotly for interatice vizualization."""
# imports
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import cos, pi
import plotly.express as px
import plotly.graph_objects as go

#frames to be vizualized
frame_x = [] #! why dont use np array?
frame_y = []
frame_z = []

frames_all = []
text_anima = "Genetic algorithm, generation {}"

# Domain of a Rastringin function
precision = 2**10
lower_bound = -5.12
upper_bound = 5.12
xy_coordinates = np.linspace(lower_bound,upper_bound, precision)
                                                                               
# Rastingin plot
X,Y = xy_coordinates,xy_coordinates
X, Y = np.meshgrid(X, Y)
A = 10
n = 2
Z =  A*n + (X**2 - A*np.cos(2*pi*X) )  +  (Y**2 -A*np.cos(2*pi*Y)) 


def nullGeneration(num_individuals=10,string_len=8,values=2,num_coordinates=2): 
    # Duplicate values are possible
    nullgeneration = random.randint(values, size=(num_individuals,num_coordinates,string_len))

    return nullgeneration

def binarToDecim(binary_list):
    decimal_number = 0
    for index, binary in enumerate (binary_list[:: -1]):
        decimal_number += 2**index*int(binary)
    return decimal_number

def RastriginFun(X,Y,A=10,n=2):
    Z =  A*n + (X**2 - A*np.cos(2*pi*X) )  +  (Y**2 -A*np.cos(2*pi*Y))
    return Z

"""
def ObjectiveFun(individual): 
    INPUT: individual cooded as two binary "strings" standing for x,y coordinates 
        -> "translate" into decimal numbers -> OUTPUT: RastriginFun(x,y)

    decimal_index_X = binarToDecim(individual[0])
    decimal_index_Y = binarToDecim(individual[1])
gi
    print("decimal_index_X",,decimal_index_X)

    X = xy_coordinates[decimal_index_X] 
    Y = xy_coordinates[decimal_index_Y] 


    return RastriginFun(X,Y)
"""

def evaluation(individuals,  cycle, num_individuals=10): 
    indi_fitnes = np.array([]) 
                                                                              
    for individual in individuals:
        decimal_index_x = binarToDecim(individual[0])
        decimal_index_y = binarToDecim(individual[1])


        #coordinates of an individual on Rastrigin plot
        x = xy_coordinates[decimal_index_x]
        y = xy_coordinates[decimal_index_y]
        z = RastriginFun( x,y ) 

        fitness = 1/z

        indi_fitnes = np.append( [individual,fitness],indi_fitnes ) 

        # add individual for visualization
        frame_x.append(x)
        frame_y.append(y)
        frame_z.append(z)
    
    # add whole generation visualization
    add_frames(frame_x,frame_y,frame_z,cycle)
      
    
    frame_x.clear() # emptying lists
    frame_y.clear()
    frame_z.clear()

    indi_fitnes = indi_fitnes.reshape(num_individuals,2)


    return indi_fitnes

def add_frames(x,y,z,cycle): 
	frames_all.append( go.Frame(data=[go.Scatter3d(x=x,  
    				                  y=y,z=z,mode="markers",
                                      marker=dict(color="red", size=8),)   
                                     ], 
                               layout=go.Layout(title_text=text_anima.format(cycle)),
                                                name=str(cycle) 
                                ) 
                     )


def selection(valued_individuals,num_parents=5,num_individuals=15):

    sum_fitness = sum(valued_individuals[:,1])
    propability_individuals = valued_individuals[:,1]/(sum_fitness)

    #! changing data type from float64 to float so that can be used as keyword "p" in random.choice #!inst there a prretier way to do that
    propability_individuals = propability_individuals.astype('float')

    parents = np.random.choice(valued_individuals[0:num_individuals,0],size=(num_parents,2),p=propability_individuals )
    return parents
                                                                               
def recombine(indi0,indi1,point_recombine=4,string_len=8,num_children=10,num_coordinates=2): # function for recombining two binary strings, point_recombine stands for point of recombination

    new_indi0 = np.array([indi0[ :point_recombine],indi1[point_recombine:]])

    new_indi1 = np.array([indi1[ :point_recombine],indi0[point_recombine:]])

    new_indi0 = new_indi0.reshape(string_len)
    new_indi1 = new_indi1.reshape(string_len)

    return [new_indi1,new_indi0]

def crossover(parents,string_len=8,num_children=10,num_coordinates=2,point_recombine=4):
    children = [] #! meh way
    for couple in parents:

        children.append(recombine(couple[0][0],couple[1][0],point_recombine,string_len,num_children,num_coordinates)) #! super hard to read
        children.append(recombine(couple[0][1],couple[1][1],point_recombine,string_len,num_children,num_coordinates))

    children = np.array([children])
    children = children.reshape(num_children,num_coordinates,string_len)
    return children

def mutation(children,indi_to_mutate=1,num_children=10,string_len=8,num_coordinates=2): #!clean up
    for i in range (indi_to_mutate):
        child = random.randint(num_children) # pick one of the children
        coordinate = random.randint(num_coordinates) # pick X or Y coordinate (cooded in bit string)
        gene = random.randint(string_len) # pick one gene (bit) to change

        if children[child,coordinate,gene] == 1:
            children[child,coordinate,gene] = 0
        else:
            children[child,coordinate,gene] = 1

    return children

def frame_args(duration):
    return {
            "frame": {"duration": duration,"redraw": "redraw"},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 150, "easing": "linear"},
        }

def vizualization():
    sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.3,
                "x": 0.1,
                "y": 0.1,
                "steps": [
                    {
                        "args": [[frame.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, frame in enumerate(frames_all)
                ],
            }
        ]
    
    text_anima = "Genetic algorithm, generation {}"
    fig = go.Figure(
        data=[go.Surface(z=Z,x=X,y=Y),
              go.Surface(z=Z,x=X,y=Y )],
        layout=go.Layout(
               xaxis=dict(range=[lower_bound, upper_bound], 
            	       autorange=False, 
            	       zeroline=False),
               yaxis=dict(range=[lower_bound, upper_bound], 
            	       autorange=False, 
            	       zeroline=False),
        title_text="Genetic algorithm", hovermode="closest",
        updatemenus=[dict(type="buttons",
        				  direction = "left",
        				  x = 0.1,
        				  y = 0.1,
        				  pad = dict(r= 10, t=70),
                                buttons=[dict(label="Run",
                                            method="animate",
                                            args=[None, frame_args(200)] ),
                                        dict(label="Stop",
                                            method="animate",
                                            args=[[None],frame_args(0)] ) ],
                          
                        )
                    ],

        sliders=sliders),
        frames=frames_all
    )
                                                                                            
    fig.show()

def geneticAglorithm(cycles=30,num_individuals=40,
	num_parents=20,num_children=40,string_len=10,
	values=2,num_coordinates=2,point_recombine=5,indi_to_mutate=2): # amount of cycles is equivalent to the number of generation
    #incialization of the null gen
    generation = nullGeneration(num_individuals,string_len,values,
    							num_coordinates) 
    for cycle in range (cycles):
        #print("Generation {}".format(cycle))

        generation = evaluation(generation,cycle, num_individuals)
        #print("individuals Rastingin fun output",1/generation[:,1])

        generation = selection(generation,num_parents,
        	                   num_individuals)
        generation = crossover(generation,string_len,num_children,
                              num_coordinates,point_recombine) #hard to read
        generation = mutation(generation,indi_to_mutate,
        	                  num_children,string_len,num_coordinates)

    #vizualization()
    
geneticAglorithm()

