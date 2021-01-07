

def _init():
    global _globals
    _globals={'cpu':True,'base_speed':59.45232}

def set_value(key,value):
    # define a global variable
    if not key in _globals.keys():
        raise Exception("invalid key")
    _globals[key]=value

def get_value(key):
    # get a global variable
    return _globals[key]