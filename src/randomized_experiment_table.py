import pickle
import numpy as np

def print_results(d):
    for key in d:
        acc = np.array(d[key])
        name = key.replace(',',';')
        if name.startswith('NeuralNetwork'):
            name = 'NeuralNetwork'
        print(name+',%.1f (%.1f)'%(np.mean(acc),np.std(acc)))

with open('../results/randomized_break.pkl','rb') as f:
    break_acc = pickle.load(f)
with open('../results/randomized_frag.pkl','rb') as f:
    frag_acc = pickle.load(f)
with open('../results/randomized_boot.pkl','rb') as f:
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


savefile = '../tables/randomized_experiment.tex'
caption = 'Results of randomized ML experiment. The mean accuracy over 100 trials is reproted, with the standard deviation in paraentheses.'
table_key = 'rand'
fontsize = 'small'
small_caps = True

#LaTeX preamble Code
f = open(savefile,"w")
f.write("\\documentclass{article}\n")
f.write("\\usepackage[T1]{fontenc}\n")
f.write("\\usepackage{booktabs}\n")
f.write("\\usepackage[margin=1in]{geometry}\n")
f.write("\\begin{document}\n")
f.write("\n\n\n")
f.write("\\begin{table}[t!]\n")
f.write("\\vspace{-3mm}\n")
f.write("\\caption{"+caption+"}\n")
f.write("\\vspace{-3mm}\n")
f.write("\\label{tab:"+table_key+"}\n")
f.write("\\vskip 0.15in\n")
f.write("\\begin{center}\n")
f.write("\\begin{"+fontsize+"}\n")
if small_caps:
    f.write("\\begin{sc}\n")
f.write("\\begin{tabular}{lccc}\n")
f.write("\\toprule\n")
f.write("& {\\bf Break-Level} & {\\bf Frag-Level Split} & {\\bf Frag-Level}\\\\ \n")
f.write("{\\bf Algorithm} & {\\bf Split} & {\\bf with Bootstrapping} &{\\bf Split} \\\\ \n")
f.write("\\midrule\n")

#Loop over table and write to file
for key in break_acc:
    name = key.split('(',1)[0]
    if 'SVC' in key:
        name = 'RBF SVM'
    if "kernel='linear'" in key:
        name = 'Linear SVM'
    if name == 'LinearDiscriminantAnalysis':
        name = 'LDA'
    if name == 'RandomForestClassifier':
        name = 'Random Forests'
    if name.startswith('KN'):
        name = "Nearest neighbor"
    if name.startswith('Neural'):
        name = 'Neural Network'
    
    f.write(name)
    brk = np.array(break_acc[key])
    f.write('& %.1f (%.1f)'%(np.mean(brk),np.std(brk)))
    boot = np.array(boot_acc[key])
    f.write('& %.1f (%.1f)'%(np.mean(boot),np.std(boot)))
    frag = np.array(frag_acc[key])
    f.write('& %.1f (%.1f)'%(np.mean(frag),np.std(frag)))
    f.write('\\\\ \n')

#End of tex file
f.write("\\bottomrule\n")
f.write("\\end{tabular}\n")
if small_caps:
    f.write("\\end{sc}\n")
f.write("\\end{"+fontsize+"}\n")
f.write("\\end{center}\n")
f.write("\\vskip -0.1in\n")
f.write("\\end{table}")
f.write("\n\n\n")
f.write("\\end{document}\n")
f.close()
