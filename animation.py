import plotly.express as px
import plotly.graph_objects as go
#frames to be vizualized

frames_all = []
text_anima = "Genetic algorithm, generation {}"

def add_frames(x,y,z,cycle,frames_all): 
	frames_all.append( go.Frame(data=[go.Scatter3d(x=x,  
    				                  y=y,z=z,mode="markers",
                                      marker=dict(color="red", size=8),)   
                                     ], 
                               layout=go.Layout(title_text=text_anima.format(cycle)),
                                                name=str(cycle) 
                                ) 
                     )

def frame_args(duration):
    return {
            "frame": {"duration": duration,"redraw": "redraw"},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 150, "easing": "linear"},
        }

def visualization(frames_all,X,Y,Z,lower_bound,upper_bound):
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