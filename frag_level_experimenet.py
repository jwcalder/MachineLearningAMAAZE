import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def frag_to_break(x_frag,y_frag,num_breaks_per_frag):
    num_fragments = x_frag.shape[0]
    n = num_fragments*num_breaks_per_frag
    x = np.ones((num_fragments,num_breaks_per_frag))*x_frag
    y = np.ones((num_fragments,num_breaks_per_frag))*y_frag
    x = np.reshape(x,(n,1))
    y = np.reshape(y,(n,))

    return x,y

num_fragments = 100
num_breaks_per_frag = 10

x_frag = np.random.rand(num_fragments,1)
y_frag = (np.random.rand(num_fragments,1) > 0.5).astype(int)

#Splitting at break level
x,y = frag_to_break(x_frag,y_frag,num_breaks_per_frag)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
accuracy = accuracy_score(rf_pred, y_test)
print('Break-Level Accuarcy: ',accuracy)

#Splitting at fragment level
x_frag_train, x_frag_test, y_frag_train, y_frag_test = train_test_split(x_frag, y_frag, test_size=0.25)
x_train,y_train = frag_to_break(x_frag_train,y_frag_train,num_breaks_per_frag)
x_test,y_test = frag_to_break(x_frag_test,y_frag_test,num_breaks_per_frag)

rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
accuracy = accuracy_score(rf_pred, y_test)
print('Frag-Level Accuarcy: ',accuracy)
