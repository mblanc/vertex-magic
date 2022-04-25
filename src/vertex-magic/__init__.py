from .vertex import Vertex

def load_ipython_extension(ipython):
    ipython.register_magics(Vertex)
