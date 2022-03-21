import pandas as pd
import numpy as np

#Fields to compute summary statistics of
sum_stats_fields = ['Mean','Median','STD','Max','Min','Range','euclid_len']#,'arc_len','arc_angle',]

#Fields to count categorical values of
counts_fields =  ['interior_edge','Interrupted','ridge_notch','interior_notch']#,'arc_angle_class']

#Fields to directly copy
copy_fields = ['Species','Element','Effector']

#All fields at frag level
all_fields = copy_fields + counts_fields + sum_stats_fields 

#Read CSV file
df = pd.read_csv('finaldata_break_level.csv')

#Initialize frag level dictionary
frag_data = {}

#Keep track of possible categorical values
cat_values = {}
for field in counts_fields:
    cat_values[field]=[]

#Build dictionary at fragment level
for i in range(len(df)):
    
    Specimen = df['Specimen'][i]

    #Add specimen to dictionary if not already present
    if Specimen not in frag_data.keys():
        frag_data[Specimen] = {'Break Count':0}  #Start with 0 break count
        for field in all_fields:
            frag_data[Specimen][field] = []  #Initialize with empty list

    #Increment break count
    frag_data[Specimen]['Break Count'] += 1

    #Copy copy fields
    for field in copy_fields:
        frag_data[Specimen][field] = df[field][i]

    #Append all counts fields to list and update cat_values
    for field in counts_fields:
        s = str(df[field][i])
        frag_data[Specimen][field] += [s]
        if s not in cat_values[field]:
            cat_values[field] += [s]

    #Append all summary stats fields to list
    for field in sum_stats_fields:
        frag_data[Specimen][field] += [df[field][i]]


#Loop over fragments and compute summary statistics, counts, and copy data
frag_sum_data = {}
for Specimen in frag_data:

    frag_sum_data[Specimen]={}

    #Copy fields
    for field in copy_fields:
        frag_sum_data[Specimen][field] = frag_data[Specimen][field]

    #Beak count
    frag_sum_data[Specimen]['Break Count'] = frag_data[Specimen]['Break Count']

    #Counting fields
    for field in counts_fields:
        data = np.array(frag_data[Specimen][field])
        for value in cat_values[field]:
            count = np.sum(data == value)
            frag_sum_data[Specimen][field+'_'+value] = count

    #Summary statistics fields
    for field in sum_stats_fields:
        data = np.array(frag_data[Specimen][field])
        frag_sum_data[Specimen][field+'_min'] = np.min(data)
        frag_sum_data[Specimen][field+'_max'] = np.max(data)
        frag_sum_data[Specimen][field+'_mean'] = np.mean(data)
        frag_sum_data[Specimen][field+'_median'] = np.median(data)
        frag_sum_data[Specimen][field+'_std'] = np.std(data)


#Print out to CSV file
#Print header (column titles) of CSV file
specimen = list(frag_sum_data.keys())[0]
columns = ['Specimen'] + list(frag_sum_data[specimen].keys())
print(','.join(columns))

#Print all data to CSV file
for Specimen in frag_sum_data:
    print(Specimen,end='')
    for field in frag_sum_data[Specimen]:
        print(',',end='')
        print(frag_sum_data[Specimen][field],end='')
    print('') 



























