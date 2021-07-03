class Data_Generator:
  def __init__(self,feature_dim,n_sample=1000,noise=0.5):
    self._feature_dim=feature_dim
    self._n_sample=n_sample
    self._noise=noise


    self._coefficient_list=None
    self._params=None
    self._dataset=None

    self._init_coefficient()
    self._init_params()

  def _init_coefficient(self):
    self._coefficient_list=[0]+[1 for i in range(self._feature_dim)]

  def _init_params(self):
    self._params={f:{"mean":0,"std":1} for f in range(1,self._feature_dim+1)}

  def make_dataset(self,init_noise=False):
    x_data=np.zeros(shape=(self._n_sample,1))
    y_data=np.zeros(shape=(self._n_sample,1))

    for f_idx in range(1,self._feature_dim+1):
      feature_data=np.random.normal(loc=self._params[f_idx]["mean"],scale=self._params[f_idx]["std"],size=(self._n_sample,1))

      x_data=np.hstack((x_data,feature_data))
      y_data += self._coefficient_list[f_idx]*feature_data #임의로 뿌린 값에 계수를 곱해 y_data를 만든다
      y_data += self._coefficient_list[0]

    if init_noise==True:
      y_data+=5*np.random.normal(loc=0,scale=self._noise,size=(self._n_sample,1))

    self._dataset=np.hstack((x_data,y_data))

    return self._dataset

  def set_coefficient(self,data):
    self._coefficient_list=data
    
datagen=Data_Generator(feature_dim=1,n_sample=200,noise=0.4)
datagen.set_coefficient([0,5])
dataset=datagen.make_dataset(init_noise=True)

x_data=dataset[:,1]
y_data=dataset[:,-1]
