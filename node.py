class plus_node:
  def __init__(self):
    self._x=None
    self._y=None
    self._z=None
  
  def forward(self,x,y):
    self._x=x
    self._y=y
    self._z=self._x+self._y
    return self._z

  def backward(self,dz):
    return dz,dz

class minus_node:
  def __init__(self):
    self._x=None
    self._y=None
    self._z=None
  
  def forward(self,x,y):
    self._x=x
    self._y=y
    self._z=self._x-self._y
    return self._z

  def backward(self,dz):
    return dz,-dz

class mul_node:
  def __init__(self):
    self._x=None
    self._y=None
    self._z=None
  
  def forward(self,x,y):
    self._x=x
    self._y=y
    self._z=self._x*self._y
    return self._z

  def backward(self,dz):
    return dz*self._y,dz*self._x

class square:
  def __init__(self):
    self._x=None
    self._z=None
  
  def forward(self,x):
    self._x=x
    self._z=self._x*self._x
    return self._z

  def backward(self,dz):
    return 2*dz*self._x

class mean:
  def __init__(self):
    self._x=None
    self._z=None
  
  def forward(self,x):
    self._x=x
    self._z=np.mean(self._x)
    return self._z

  def backward(self,dz):
    return dz*1/len(self._x)*np.ones_like(self._x)
