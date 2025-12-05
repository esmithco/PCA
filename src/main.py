import utils
from pca_code import PCA

def main():
    # Create the matrix to study
    data = utils.matrix_excel('./archive/Country-data.csv')

    # instance of the class to access everything
    pca = PCA(data)
    pca.fit_transform()

    

    # print the data needed
    vc_p = pca.M_PCA()
    new_data = pca.new_data
    print('This is the matrix with the recopiled data')
    print(new_data[0])
    print('These are the percentage of each component')
    print([float(round(i,5)) for i in vc_p])

    # Let's calculate the error to interpret the closeness
    err = pca.error_ca()
    print('This is the percentage of error:', err)

if __name__ == '__main__':
    main()