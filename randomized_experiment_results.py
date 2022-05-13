import pickle
import numpy as np

def print_results(d):
    for key in d:
        acc = np.array(d[key])
        name = key.replace(',',';')
        if name.startswith('NeuralNetwork'):
            name = 'NeuralNetwork'
        print(name+',%.1f (%.1f)'%(np.mean(acc),np.std(acc)))

with open('results/randomized_break.pkl','rb') as f:
    break_acc = pickle.load(f)
with open('results/randomized_frag.pkl','rb') as f:
    frag_acc = pickle.load(f)
with open('results/randomized_boot.pkl','rb') as f:
    boot_acc = pickle.load(f)


print('Break-Level Split')
print('Method,Accuracy (Standard Deviation)')
print_results(break_acc)

print('\nBootstrapping')
print('Method,Accuracy (Standard Deviation)')
print_results(boot_acc)

print('\nFrag-Level split')
print('Method,Accuracy (Standard Deviation)')
print_results(frag_acc)

