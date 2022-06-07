from inventory import sample_inventory
import sys
import pickle
import numpy as np
import pandas as pd

#Fields to grab
df = sample_inventory(['Specimen','Species', 'Element', 'Effector'])
df = df.replace('mtcar','mtpod')
df = df.replace('mttar','mtpod')

species = ['Cervus canadensis','Odocoileus virginianus']
element = ['fem','hum','mtpod','other','rad-uln','tib','unknown']
#element = ['fem','hum','mtcar','mtpod','mttar','other','rad-uln','tib','unknown']
effector = ['HSAnv','teeth']

savefile = '../tables/inventory.tex'
caption = 'Table caption'
table_key = 'inventory'
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
f.write("&{\\bf "+effector[0]+"} & {\\bf "+effector[1]+"} & {\\bf Total}\\\\ \n")
f.write("&{\\bf Fragments (Breaks)} & {\\bf Fragments (Breaks)} & {\\bf Fragments (Breaks)}\\\\ \n")

#Loop over table and write to file
for spec in species:
    f.write("\\midrule\n")
    f.write('{\\bf '+spec+'}')
    #Totals for species/effector
    for eff in effector:
        I = (df['Species'] == spec) & (df['Effector'] == eff) 
        num_frag = np.sum(I)
        num_breaks = np.sum(df['NumBreaks'][I])
        f.write('& {\\bf %d (%d)}'%(num_frag,num_breaks))
        #print(spec,eff,num_frag,num_breaks)

    #Total for the species
    I = df['Species'] == spec
    num_frag = np.sum(I)
    num_breaks = np.sum(df['NumBreaks'][I])
    f.write('& {\\bf %d (%d)}'%(num_frag,num_breaks))
    f.write('\\\\ \n')
    #print(spec,num_frag,num_breaks)

    #Rows for each element for the species
    for elem in element:
        f.write('\\hspace{3mm} '+elem)
        #Totals for species/element/effector
        for eff in effector:
            I = (df['Species'] == spec) & (df['Element'] == elem) & (df['Effector'] == eff)
            num_frag = np.sum(I)
            num_breaks = np.sum(df['NumBreaks'][I])
            f.write('& %d (%d)'%(num_frag,num_breaks))
            #print(spec,elem,eff,num_frag,num_breaks)

        #Total for species/element
        I = (df['Species'] == spec) & (df['Element'] == elem)
        num_frag = np.sum(I)
        num_breaks = np.sum(df['NumBreaks'][I])
        f.write('& %d (%d)'%(num_frag,num_breaks))
        f.write('\\\\ \n')
        #print(spec,elem,num_frag,num_breaks)

f.write("\\midrule\n")
f.write('{\\bf Total}')
#Grand totals
for eff in effector:
    I = df['Effector'] == eff
    num_frag = np.sum(I)
    num_breaks = np.sum(df['NumBreaks'][I])
    f.write('& {\\bf %d (%d)}'%(num_frag,num_breaks))
num_frag = len(df)
num_breaks = np.sum(df['NumBreaks'])
f.write('& {\\bf %d (%d)}'%(num_frag,num_breaks))
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
