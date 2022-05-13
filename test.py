def Merge(D1,D2):
    py = D1 | D2
    return py
D1 = {"RollNo": "10", "Age":"18"}
D2 = {"Marks": "90", "Grade": "A"}
D3 = Merge(D1, D2)
print(D3)
