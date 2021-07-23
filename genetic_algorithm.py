"""Building a simple Genetic algorithm from sratch (more or less) and using it to find global extreme on Rastingin function. 
   Using modul numpy for arrays and Plotly for interatice visualization."""
# imports
from numpy import random
import numpy as np
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

def RastriginFun(X,Y,A=10,n=2):
    Z =  A*n + (X**2 - A*np.cos(2*pi*X) )  +  (Y**2 -A*np.cos(2*pi*Y))
    return Z

def binarToDecim(binary_list):
    decimal_number = 0
    for index, binary in enumerate (binary_list[:: -1]):
        decimal_number += 2**index*int(binary)
    return decimal_number

def nullGeneration(num_individuals=10,string_len=8,values=2,num_coordinates=2): 
    # Duplicate values are possible
    nullgeneration = random.randint(values, size=(num_individuals,num_coordinates,string_len))

    return nullgeneration

def evaluation(individuals,  cycle, num_individuals=10): 
    valued_individuals = np.array([]) 
                                                                   
    for individual in individuals: #!      
        decimal_index_x = binarToDecim(individual[0])
        decimal_index_y = binarToDecim(individual[1])

        #coordinates of an individual on Rastrigin plot
        x = xy_coordinates[decimal_index_x]
        y = xy_coordinates[decimal_index_y]
        z = RastriginFun( x,y ) 

        fitness = 1/z
        valued_individuals = np.append( np.array([individual,fitness]),valued_individuals )

        # add individual for visualization
        frame_x.append(x)
        frame_y.append(y)
        frame_z.append(z)
    
    # add whole generation visualization
    add_frames(frame_x,frame_y,frame_z,cycle)
         
    frame_x.clear() # emptying lists
    frame_y.clear()
    frame_z.clear()

    valued_individuals = valued_individuals.reshape(num_individuals,2) #! can I do it without reshaping?

    return valued_individuals

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

    # propabality of individuals becoming a parent based on their fitness
    propability_individuals = valued_individuals[:,1]/(sum_fitness)
    
    #changing data type from float64 to float so that can be used as keyword "p" in random.choice
    propability_individuals = propability_individuals.astype('float')

    parents = np.random.choice(valued_individuals[0:num_individuals,0],size=(num_parents,2),p=propability_individuals )
    return parents
                                                                               
def recombine(prnt_1_bitstr,prnt_2_bitstr,point_recombin=4,string_len=8,num_children=10,num_coordinates=2): 
    # function for recombining two binary strings, point_recombin stands for point of recombination

    child_x= np.array([prnt_1_bitstr[ :point_recombin],prnt_2_bitstr[point_recombin:]])
    child_y = np.array([prnt_1_bitstr[ :point_recombin],prnt_2_bitstr[point_recombin:]])

    child_x = child_x.reshape(string_len) 
    child_y = child_y.reshape(string_len)

    return [child_x,child_y]

def crossover(parents,string_len=8,num_children=10,num_coordinates=2,point_recombin=4):
    children =  [] #!

    # every couple generates two children
    for couple in parents:
        child_1 = recombine(couple[0][0],couple[1][0],point_recombin,string_len,num_children,num_coordinates)
        child_2 = recombine(couple[0][1],couple[1][1],point_recombin,string_len,num_children,num_coordinates)

        children.append(child_1) 
        children.append(child_2)
    
    children = np.array(children)
    return children

def mutation(children,indi_to_mutate=1,num_children=10,string_len=8,num_coordinates=2): #!clean up
    
    for i in range (indi_to_mutate): #! rename "i"
        child = random.randint(num_children) # pick one of the children
        coordinate = random.randint(num_coordinates) # pick X or Y coordinate index (cooded in bit string)
        gene = random.randint(string_len) # pick one gene (bit) to change

        #! this code only work with binary so the mutation negates
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

def geneticAglorithm (cycles=30,num_individuals=40,
	num_parents=20,num_children=40,string_len=10,
	values=2,num_coordinates=2,point_recombin=5,indi_to_mutate=2): # amount of cycles is equivalent to the number of generation
    #incialization of the null gen
    generation = nullGeneration(num_individuals,string_len,values,
    							num_coordinates) 
    for cycle in range (cycles):
        generation = evaluation(generation,cycle, num_individuals)

        generation = selection(generation,num_parents,
        	                   num_individuals)
        generation = crossover(generation,string_len,num_children,
                              num_coordinates,point_recombin) #! hard to read
        generation = mutation(generation,indi_to_mutate,
        	                  num_children,string_len,num_coordinates)

    vizualization()
    
geneticAglorithm()

