
fig, ax = plt.subplots(2, 1, figsize = (50,10)) 
ax[0].plot(th_list)
title_font = {'size':30, 'alpha':0.8, 'color':'navy'} 
label_font = {'size':20, 'alpha':0.8} 
plt.style.use('seaborn') 
ax[0].set_title(r'$\theta$', fontdict = title_font)
ax[1].set_title("Loss", fontdict = title_font)
ax[1].set_xlabel("Iteration", fontdict = label_font)
ax[1].plot(cost_list)



N_lines=100

cmap=cm.get_cmap("rainbow",lut=N_lines)

fig,ax=plt.subplots(1,1,figsize=(10,10))
ax.scatter(x_data,y_data)

test_th=th_list[:N_lines]
x_range=np.array([np.min(x_data),np.max(x_data)])

for line_idx in range(N_lines):
  pred_line=np.array([x_range[0]*test_th[line_idx],x_range[1]*test_th[line_idx]])

  ax.plot(x_range,pred_line,color=cmap(line_idx),alpha=0.1)
