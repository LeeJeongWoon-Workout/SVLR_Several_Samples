for epoch in range(epochs):
  z1=node1.forward(th,x_data)
  z2=node2.forward(y_data,z1)
  z3=node3.forward(z2)
  z4=node4.forward(z3)

  dz4=node4.backward(1)
  dz3=node3.backward(dz4)
  dy,dz2=node2.backward(dz3)
  dth,dx=node1.backward(dz2)

  th=th-lr*np.sum(dth)
  
  th_list.append(th)
  cost_list.append(z4)
