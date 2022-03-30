import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import spatial

def save_curve(x,y,z,idx,break_num,fname):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('Break %d'%break_num)
    plt.savefig(fname)
    plt.close()

def plot_curve(x,y,z,idx,break_num):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[idx[1:-1]],y[idx[1:-1]],z[idx[1:-1]],c='b')
    ax.scatter(x[idx[0]],y[idx[0]],z[idx[0]],c='r',s=100)
    ax.scatter(x[idx[-1]],y[idx[-1]],z[idx[-1]],c='g',s=100)
    ax.plot(x[idx],y[idx],z[idx])
    ax.set_title('Break %d'%break_num)

def tsp_path(D,i):
    n = D.shape[0]
    path = [i]
    length = []
    for j in range(n-1):
        path += [np.argmin(D[path[j],:])]
        length += [D[path[j],path[j+1]]]
        D[:,path[j]] = np.inf
    
    return path,np.array(length)

def tsp_order(x,y,z):
    n = len(x)
    X = np.stack((x,y,z)).T
    D = spatial.distance_matrix(X,X)
    D[range(n),range(n)] = np.inf

    #Try starting from each point
    min_length = np.ones((n-1,))*np.inf
    min_path = list(range(n))
    for i in range(n):
        path,length = tsp_path(D.copy(),i)
        if np.sum(length) < np.sum(min_length):
            min_length = length
            min_path = path
    return min_path

#Load angle data dictionary
with open('break_curve_data.pkl', 'rb') as f:
    angledata = pickle.load(f)

#Loop over the breaks for meshname
for mesh_name in angledata:
    # print(mesh_name)
    for break_num in angledata[mesh_name]:
        # print('Break #:',break_num)
        x = angledata[mesh_name][break_num]['x']
        y = angledata[mesh_name][break_num]['y']
        z = angledata[mesh_name][break_num]['z']

        idx = tsp_order(x,y,z)
        
        # This completely reorders the xs, ys, and zs. 
        new_x = [x[i] for i in idx]
        new_y = [y[i] for i in idx]
        new_z = [z[i] for i in idx]
        
        angledata[mesh_name][break_num]['x'] = new_x
        angledata[mesh_name][break_num]['y'] = new_y
        angledata[mesh_name][break_num]['z'] = new_z

f = open('break_curve_data.pkl','wb')
pickle.dump(angledata,f)
f.close()
