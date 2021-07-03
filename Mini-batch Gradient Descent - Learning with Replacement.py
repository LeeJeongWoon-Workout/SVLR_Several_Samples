batch_size=20
t_iteration=1000

for _ in range(t_iteration):
    idx_np=np.arange(len(x_data))
    random_idx=np.random.choice(idx_np,batch_size)

    x=x_data[random_idx]
    y=y_data[random_idx]

    z1=node1.forward(th,x)
    z2=node2.forward(y,z1)
    l=node3.forward(z2)
    j=node4.forward(l)

    dj=node4.backward(1)
    dl=node3.backward(dj)
    dy,dz2=node2.backward(dl)
    dth,dx=node1.backward(dz2)

    th=th-lr*np.sum(dth)
    th_list.append(th)
    cost_list.append(j)
