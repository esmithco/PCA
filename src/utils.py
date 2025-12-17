import pandas as pd

# create function to import the data and create a matrix
def matrix_excel(ruta):
    data_all = pd.read_csv(ruta)
    data_num = data_all.select_dtypes(include='number')
    data = data_num.to_numpy()
    return data

# Function to select the values more important, 90 percent (at least two values)
def nin_vari(ls):
    sm = ls[0]
    tl = sum(ls)
    vc = [ls[0]]
    vc_p = [ls[0]/tl]
    for i in ls[1:]:
        sm = sm + i
        vc_p.append(i/tl)
        vc.append(i)
        if sm/tl >= 0.7:
            return vc, vc_p