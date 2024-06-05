import numpy as np
import copy
import misc_functions as misc_fns


#============
#============
#
# Variables
#
#===========
#===========

class D_nodes():
    """Class for D_nodes
    """
    def __init__(self, D_nodes):
        self._D_nodes = clean_D_nodes(D_nodes=D_nodes)

    def get_D_nodes(self):

        return self._D_nodes
    
    def clean_nodes(self, nodes=None, unique=None):
        
        return clean_nodes(D_nodes=self.get_D_nodes(), nodes=nodes, unique=unique)
    
    def calc_names(self, name=None, name_rm=None):
        name = misc_fns.A_rm(A=name, A_rm=name_rm)
        return [element for lis in [[i for i in misc_fns.make_iterable_array(self.__dict__[param]._names)] for param in name] for element in lis] 
    


class p_range:
    """Class for valid range of parameter
    """
    def __init__(self, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None):

        self._messages =    {
                                "permitted": "must take one of the following values:",
                                "incl_low": "must be greater than or equal to",
                                "incl_high": "must be less than or equal to",
                                "excl_low": "must be greater than",
                                "excl_high": "must be less than",
                            }

        self._condition_names = [i for i in self._messages] 
        self._values =  {i: None for i in self._messages}

        self.set_range(permitted=permitted, incl_low=incl_low, incl_high=incl_high, excl_low=excl_low, excl_high=excl_high) 


    def set_range(self, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None):
        self.set_permitted(value=permitted)
        self.set_incl_low(value=incl_low)  
        self.set_incl_high(value=incl_high)
        self.set_excl_low(value=excl_low)  
        self.set_excl_high(value=excl_high) 

    def set_permitted(self, value):
        self._values["permitted"] = value

    def set_incl_low(self, value):
        self._values["incl_low"] = value

    def set_incl_high(self, value):
        self._values["incl_high"] = value

    def set_excl_low(self, value):
        self._values["excl_low"] = value

    def set_excl_high(self, value):
        self._values["excl_high"] = value
    

    def check(self, condition_name, values):

        if condition_name == "permitted":
            return self.check_permitted(values=values)
        elif condition_name == "incl_low":
            return self.check_incl_low(values=values)
        elif condition_name == "incl_high":
            return self.check_incl_high(values=values)
        elif condition_name == "excl_low":
            return self.check_excl_low(values=values)
        elif condition_name == "excl_high":
            return self.check_excl_high(values=values)
        else:
            raise ValueError("condition_name {} {}".format(self._messages["permitted"], self._condition_names))

    def check_permitted(self, values):

        passed = None
        if type(self._values["permitted"]) != type(None):
            passed = np.all(np.isin(values, self._values["permitted"]))
        return passed

    def check_incl_low(self, values):
        
        passed = None
        if type(self._values["incl_low"]) != type(None):
            passed = np.all(values >= self._values["incl_low"])
        return passed

    def check_incl_high(self, values):
        
        passed = None
        if type(self._values["incl_high"]) != type(None):
            passed = np.all(values <= self._values["incl_high"])
        return passed

    def check_excl_low(self, values):
        
        passed = None
        if type(self._values["excl_low"]) != type(None):
            passed = np.all(values > self._values["excl_low"])
        return passed

    def check_excl_high(self, values):
        
        passed = None
        if type(self._values["excl_high"]) != type(None):
            passed = np.all(values < self._values["excl_high"])
        return passed

    def check_range(self, values):

        passed = {i: None for i in self._condition_names}
        for i in self._condition_names:
            passed[i] = self.check(i, values=values)
                
        return passed


    def get_Bounds(self, delta_excl=None):

        if delta_excl is None:
            delta_excl = 1E-8

        if self._values["incl_low"] is not None:
            lb = self._values["incl_low"]
        elif self._values["excl_low"] is not None:
            lb = self._values["excl_low"] + delta_excl
        else:
            lb = -np.inf

        if self._values["incl_high"] is not None:
            ub = self._values["incl_high"]
        elif self._values["excl_high"] is not None:
            ub = self._values["excl_high"] - delta_excl
        else:
            ub = np.inf

        return np.array([lb, ub])

    



class parameter(D_nodes):
    """Class for parameters of Hawkes process
    """
    def __init__(self, D_nodes, stype, name="parameter", etype=None, family=None, unit=None,
                    p_rng=None, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None,
                    opt_rng=None, opt_incl_low=None, opt_incl_high=None, opt_excl_low=None, opt_excl_high=None):

        super().__init__(D_nodes=D_nodes)
        
        self._value = None
        self._stype = stype
        self._name = name
        self._etype = etype
        self._shape = None
        self.family = family
        self.default = None
        self.set_unit(unit)

        if p_rng is not None:
            self._p_range = p_rng
        else:
            self._p_range = p_range(permitted=permitted, incl_low=incl_low, incl_high=incl_high, excl_low=excl_low, excl_high=excl_high)
        if opt_rng is not None:
            self._opt_range = opt_rng
        else:
            self._opt_range = p_range(incl_low=opt_incl_low, incl_high=opt_incl_high, excl_low=opt_excl_low, excl_high=opt_excl_high)
            for con in self._p_range._values:
                if self._opt_range._values[con] is None:
                    self._opt_range._values[con] = copy.deepcopy(self._p_range._values[con])

    
    def set_unit(self, unit=None):

        if type(unit) != str:
            unit = ""
        else:
            self._unit = unit

    def get_unit(self, shape=False):

        if shape:
            np.full(self._shape, self._unit)
        else:
            return self._unit


    
    def get_error_prefix(self):

        scalar = True
        if self._stype != "scalar":
            if self._D_nodes != 1:
                scalar = False

        return get_error_prefix(name=self._name, scalar=scalar)


    def check_p_range(self, values):

        p_rng = self._p_range
        passed = p_rng.check_range(values=values)

        for i in p_rng._condition_names:
            if np.any(passed[i] == False):
                raise ValueError("{} {} {}".format(self.get_error_prefix(), p_rng._messages[i], p_rng._values[i]))

        return passed


    def check_etype(self, values, elementwise=None):

        passed = misc_fns.check_type(values=values, types=self._etype, elementwise=elementwise)
        if passed == False:
            raise TypeError("{} must be of type: {}".format(self.get_error_prefix(), self._etype))

        return passed

    def set_quick(self, value):
        self._value = value



class p_scalar(parameter):
    """Class for scalar parameters of Hawkes process
    """
    def __init__(self, D_nodes=None, name="scalar", etype=[int, float], family=None, unit=None, default=None, 
                    p_rng=None, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None,
                    opt_rng=None, opt_incl_low=None, opt_incl_high=None, opt_excl_low=None, opt_excl_high=None,
                    value=None):
        

        if D_nodes is None:
            D_nodes = 1
        super().__init__(D_nodes=D_nodes, stype="scalar", name=name, etype=etype, family=family, unit=None,
                            p_rng=p_rng, permitted=permitted, incl_low=incl_low, incl_high=incl_high, excl_low=excl_low, excl_high=excl_high,
                            opt_rng=opt_rng, opt_incl_low=opt_incl_low, opt_incl_high=opt_incl_high, opt_excl_low=opt_excl_low, opt_excl_high=opt_excl_high)
        self._shape = ()
        self.set_default(default=default)

        self._value = self._default
        #self.set_value(reset=True)

        self.set_value(value=value, reset=True)

        self._names = self._name

    


    def clean(self, value=None):

        if value is None:
            if np.isin(type(None), self._etype):
                return None
            else:
                value = self.get_default()
        value = np.squeeze(value)

        if value.ndim == 0:
            value = np.resize(value, 1)[0]
        else:
            raise TypeError("{} must be of ndim 0".format(self._name))

        if type(value) == np.int32 or type(value) == np.int64:
            value = int(value)
        elif type(value) == np.float64:
            value = float(value)
        elif type(value) == bool or type(value) == np.bool_:
            value = bool(value)
        elif type(value) == np.str_:
            value = str(value)
        else:
            raise TypeError("Unsupported type")

        return value


    def check(self, value=None):

        value = self.clean(value=value)
        self.check_etype(values=value, elementwise=False)
        if np.isin(float, self._etype):
            if np.isnan(value):
                return value
        self.check_p_range(values=value)
        return value


    def set_default(self, default):
        if default is None:
            if np.isin(type(None), self._etype):
                self._default = None
            else:
                raise ValueError("A default value must be set")
        self._default = self.check(value=default)

    def get_default(self):
        return self._default


    def _set_value(self, value=None):
        self._value = self.check(value=value)


    def set_value(self, value=None, reset=None):

        if reset is None:
            reset = False

        if value is not None:
            self._set_value(value=value)
        elif reset:
            self._set_value(value=self.get_default())

    def get_value(self):
        return self._value
    
    def calc_dict(self):
        return {self._names: self._value}



class p_vector(parameter):
    """Class for vector parameters of Hawkes process
    """

    def __init__(self, D_nodes, name="vector", etype=[int, float], family=None, unit=None, default=None, 
                    p_rng=None, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None,
                    opt_rng=None, opt_incl_low=None, opt_incl_high=None, opt_excl_low=None, opt_excl_high=None,
                    value=None, nodes=None):
        super().__init__(D_nodes=D_nodes, stype="vector", name=name, etype=etype, family=family, unit=None,
                            p_rng=p_rng, permitted=permitted, incl_low=incl_low, incl_high=incl_high, excl_low=excl_low, excl_high=excl_high,
                            opt_rng=opt_rng, opt_incl_low=opt_incl_low, opt_incl_high=opt_incl_high, opt_excl_low=opt_excl_low, opt_excl_high=opt_excl_high)
        self._shape = (self._D_nodes)

        self.set_default(default=default)

        self._value = self._default
        #self.set_value(reset=True)

        self.set_value(value=value, nodes=nodes)

        self._names = [self._name + "_" + str(1+i) for i in range(self._D_nodes)]

    


    def clean(self, value=None, nodes=None, unique=True):

        if type(value) == type(None):
            if np.isin(type(None), self._etype):
                return None
            else:
                value = self.get_default()
        value = np.squeeze(value)

        nodes = self.clean_nodes(nodes=nodes, unique=unique)

        if value.ndim == 0:
            if np.isin(float, self._etype):
                value = np.full(nodes.size, value, dtype=float)
            else:
                value = np.full(nodes.size, value)
        elif value.ndim == 1:
            if value.size != nodes.size:
                raise TypeError("{} must be of size nodes.size={} or 1".format(self._name, nodes.size))
        else:
            raise TypeError("{} must be of ndim 0 or 1".format(self._name))

        return value, nodes


    def check(self, value=None, nodes=None):

        [value, nodes] = self.clean(value=value, nodes=nodes)
        self.check_etype(values=value, elementwise=True)
        self.check_p_range(values=value)

        return value, nodes


    def set_default(self, default):
        if default is None:
            raise ValueError("A default value must be set")
        self._default = self.check(value=default)[0]

    def get_default(self):
        return self._default


    def _set_value(self, value=None, nodes=None):


        [value, nodes] = self.check(value=value, nodes=nodes)
        self._value[nodes] = value


    def set_value(self, value=None, nodes=None, reset=None):

        if reset is None:
            reset = False

        if value is not None:
            self._set_value(value=value)
        elif reset:
            self._set_value(value=self.get_default())


    def get_value(self, nodes=None):

        if type(nodes) == type(None):
            return self._value
        else:
            return self._value[nodes]
        
    def calc_dict(self, nodes=None, unique=None):

        nodes = self.clean_nodes(nodes=nodes, unique=unique)

        return {self._names[i]: self._value[i] for i in nodes}



class p_matrix(parameter):
    """Class for matrix parameters of Hawkes process
    """

    def __init__(self, D_nodes, name="matrix", etype=[int, float], family=None, unit=None, default=None, 
                    p_rng=None, permitted=None, incl_low=None, incl_high=None, excl_low=None, excl_high=None,
                    opt_rng=None, opt_incl_low=None, opt_incl_high=None, opt_excl_low=None, opt_excl_high=None,
                    value=None, nodes_i=None, nodes_j=None):
        super().__init__(D_nodes=D_nodes, stype="matrix", name=name, etype=etype, family=family, unit=None,
                            p_rng=p_rng, permitted=permitted, incl_low=incl_low, incl_high=incl_high, excl_low=excl_low, excl_high=excl_high,
                            opt_rng=opt_rng, opt_incl_low=opt_incl_low, opt_incl_high=opt_incl_high, opt_excl_low=opt_excl_low, opt_excl_high=opt_excl_high)
        self._shape = (self._D_nodes, self._D_nodes)
        self.set_default(default=default)

        self._value = self._default
        #self.set_value(reset=True)

        self.set_value(value=value, nodes_i=nodes_i, nodes_j=nodes_j)

        self._names = [[self._name + "_" + str(i+1) + "_" + str(j+1) for j in range(self._D_nodes)] for i in range(self._D_nodes)]





    def clean(self, value=None, nodes_i=None, nodes_j=None):

        if type(value) == type(None):
            if np.isin(type(None), self._etype):
                return None
            else:
                value = self.get_default()
        value = np.squeeze(value)

        if value.ndim == 2 or type(nodes_i) ==  type(nodes_j) == type(None):
            if value.ndim == 0:
                if np.isin(float, self._etype):
                    value = np.full(self._shape, value, dtype=float)
                else:
                    value = np.full(self._shape, value)
            elif value.ndim == 2:
                if value.shape != self._shape:
                    raise TypeError("{} must be of shape (D_nodes, D_nodes) = {} or (1,1)".format(self._name, self._shape))
            else:
                raise TypeError("{} must be of ndim 0 or 2".format(self._name))
            nodes_i = None
            nodes_j = None
            pairs = None
        else:
            if type(nodes_j) == None:
                nodes_j = nodes_i
            elif type(nodes_i) == None:
                nodes_i = nodes_j
            nodes_i = self.clean_nodes(nodes=nodes_i, unique=False)
            nodes_j = self.clean_nodes(nodes=nodes_j, unique=False)

            if nodes_i.size != nodes_j.size:
                if nodes_i.size == 1:
                    nodes_i = np.full(nodes_j.shape, nodes_i)
                elif nodes_j.size == 1:
                    nodes_j = np.full(nodes_i.shape, nodes_j)
                else:
                    raise ValueError("nodes_i and nodes_j must be of equal length or one must be of ndim 1")
            if nodes_i.size > self._D_nodes**2:
                raise ValueError("nodes_i and nodes_j must be of length less than or equal to D_nodes^2 = {}".format(self._D_nodes**2))
        
            if value.ndim == 0 or value.size == 1:
                if np.isin(float, self._etype):
                    value = np.full(nodes_i.size, value, dtype=float)
                else:
                    value = np.full(nodes_i.size, value)
            elif value.ndim == 1:
                if value.size != nodes_i.size:
                    raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(self._name, self._shape))
                if value.size != nodes_i.size:
                    raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(self._name, self._shape))
            else:
                raise TypeError("{} must be of ndim 0 or 2".format(self._name))
            
            pairs = np.transpose(np.squeeze([[nodes_i], [nodes_j]]))
            duplicates = np.full((nodes_i.size, nodes_j.size), False)
            for j in range(nodes_j.size):
                for i in range(j):
                    duplicates[i,j] = memoryview(pairs[i]) == memoryview(pairs[j])
            if np.any(duplicates):
                raise ValueError("All pairs of node indices (nodes_i[k], nodes_j[k]) must be unique")

        return value, nodes_i, nodes_j, pairs


    def check(self, value=None, nodes_i=None, nodes_j=None):

        [value, nodes_i, nodes_j, pairs] = self.clean(value=value, nodes_i=nodes_i, nodes_j=nodes_j)
        self.check_etype(values=value, elementwise=True)
        self.check_p_range(values=value)

        return value, nodes_i, nodes_j, pairs

    
    def set_default(self, default):
        if type(default) == type(None):
            raise ValueError("A default value must be set")
        self._default = self.check(value=default)[0]

    def get_default(self):
        return self._default


    def _set_value(self, value=None, nodes_i=None, nodes_j=None):
        [value_set, nodes_i_set, nodes_j_set, pairs_set] = self.check(value=value, nodes_i=nodes_i, nodes_j=nodes_j)
        if nodes_i != None or nodes_j != None:
            self._value[nodes_i_set, nodes_j_set] = value_set
        else:
            self._value = value_set

    def set_value(self, value=None, nodes_i=None, nodes_j=None, reset=None):

        if reset is None:
            reset = False

        if value is not None:
            self._set_value(value=value, nodes_i=nodes_i, nodes_j=nodes_j)
        elif reset:
            self._set_value(value=self.get_default())

    def get_value(self, nodes_i=None, nodes_j=None):
        if type(nodes_i) == type(None) and type(nodes_j) == type(None):
            return self._value
        elif type(nodes_i) != type(None) and type(nodes_j) == type(None):
            return self._value[nodes_i,:]
        elif type(nodes_i) == type(None) and type(nodes_j) != type(None):
            return self._value[:,nodes_j]
        else:
            return self._value[nodes_i,nodes_j]
        

    def calc_dict(self, nodes_i=None, nodes_j=None, unique=None):

        nodes_i = self.clean_nodes(nodes=nodes_i, unique=unique)
        nodes_j = self.clean_nodes(nodes=nodes_j, unique=unique)

        return {self._names[i][j]: self._value[i,j] for i in nodes_i for j in nodes_j}
    


#============
#============
#
# Functions
#
#============
#============



def get_error_prefix(name, scalar=True):

    if scalar:
        return name
    else:
        return "All elements of {}".format(name)




def clean_D_nodes(D_nodes):
    """Check function for D_nodes 
            -- the number of point processes
    """
    if type(D_nodes) != int:
        raise TypeError("D_nodes must be of type int")
    if D_nodes < 1:
        raise ValueError("D_nodes must be greater than or equal to 1")
    return D_nodes


def clean_scalar(value=None, name="scalar"):

    if value is None:
        value = 0
    value = np.squeeze(value)

    if value.ndim == 0:
        value = np.resize(value, 1)[0]
    else:
        raise TypeError("{} must be of ndim 0".format(name))

    if type(value) == np.int32 or type(value) == np.int64:
        value = int(value)
    elif type(value) == np.float64:
        value = float(value)
    elif type(value) == bool or type(value) == np.bool_:
            value = bool(value)
    elif type(value) == np.str_:
        value = str(value)
    else:
        raise TypeError("Unsupported type")

    return value


def clean_vector(D_nodes, value=None, nodes=None, unique=True, name="vector"):

    if type(value) == type(None):
        value = 0
    value = np.squeeze(value)

    nodes = clean_nodes(D_nodes=D_nodes, nodes=nodes, unique=unique)

    if value.ndim == 0:
        value = np.full(nodes.size, value)
    elif value.ndim == 1:
        if value.size != nodes.size:
            raise TypeError("{} must be of size nodes.size={} or 1".format(name, nodes.size))
    else:
        raise TypeError("{} must be of ndim 0 or 1".format(name))

    return value, nodes

def clean_matrix(D_nodes, value=None, nodes_i=None, nodes_j=None, name="matrix"):

    if type(value) == type(None):
        value = 0
    value = np.squeeze(value)

    shape = (D_nodes, D_nodes)

    if value.ndim == 2 or type(nodes_i) ==  type(nodes_j) == type(None):
        if value.ndim == 0:
            value = value * np.ones(shape)
        elif value.ndim == 2:
            if value.shape != shape:
                raise TypeError("{} must be of shape (D_nodes, D_nodes) = {} or (1,1)".format(name, shape))
        else:
            raise TypeError("{} must be of ndim 0 or 2".format(name))
        nodes_i = None
        nodes_j = None
        pairs = None
    else:
        if type(nodes_j) == None:
            nodes_j = nodes_i
        elif type(nodes_i) == None:
            nodes_i = nodes_j
        nodes_i = clean_nodes(D_nodes, nodes=nodes_i, unique=False)
        nodes_j = clean_nodes(D_nodes, nodes=nodes_j, unique=False)

        if nodes_i.size != nodes_j.size:
            if nodes_i.size == 1:
                nodes_i = np.full(nodes_j.shape, nodes_i)
            elif nodes_j.size == 1:
                nodes_j = np.full(nodes_i.shape, nodes_j)
            else:
                raise ValueError("nodes_i and nodes_j must be of equal length or one must be of ndim 1")
        if nodes_i.size > D_nodes**2:
            raise ValueError("nodes_i and nodes_j must be of length less than or equal to D_nodes^2 = {}".format(D_nodes**2))
    
        if value.ndim == 0:
            value = value * np.ones(nodes_i.size)
        elif value.ndim == 1:
            if value.size != nodes_i.size:
                raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(name, shape))
        if value.size != nodes_i.size:
            raise TypeError("{} must be of size nodes_i.size = nodes_j.size = {} or 1".format(name, shape))
        else:
            raise TypeError("{} must be of ndim 0 or 2".format(name))
        
        pairs = np.transpose(np.squeeze([[nodes_i], [nodes_j]]))
        duplicates = np.full((nodes_i.size, nodes_j.size), False)
        for j in range(nodes_j.size):
            for i in range(j):
                duplicates[i,j] = memoryview(pairs[i]) == memoryview(pairs[j])
        if np.any(duplicates):
            raise ValueError("All pairs of node indices (nodes_i[k], nodes_j[k]) must be unique")

    return value, nodes_i, nodes_j, pairs


def clean_nodes(D_nodes, nodes=None, unique=True):

    if type(nodes) == type(None):
        nodes = np.arange(D_nodes)
    else:
        #nodes = np.squeeze(nodes)#
        nodes = np.squeeze(nodes)
        if nodes.dtype != int:
            raise TypeError("All elements of nodes must of type int")
        
        if nodes.ndim > 1:
            raise TypeError("nodes must be of ndim 0 or 1")
        if nodes.size > D_nodes:
            raise TypeError("nodes must be of size D_nodes or smaller")
        if np.all(np.isin(nodes, range(-D_nodes, D_nodes))) == False:
            raise ValueError("All elements of nodes must be valid indices within the range of the number of nodes")

        #nodes[nodes<0] = nodes[nodes<0] + D_nodes

        if unique:
            if nodes.size != np.unique(nodes).size:
                raise ValueError("All elements of nodes must be unique")

    return misc_fns.make_iterable_array(nodes)