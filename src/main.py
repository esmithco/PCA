import utils
from pca_code import PCA_b
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    # Create the matrix to study
    data = utils.matrix_excel('./archive/Country-data.csv')

    # instance of the class to access everything
    pca = PCA_b(data)
    pca.fit_transform()

    

    # print the data needed
    vc_p = pca.M_PCA()
    new_data = pca.new_data
    # print('This is the matrix with the recopiled data')
    # print(new_data[0])
    print('These are the percentage of each component')
    print([float(round(i,5)) for i in vc_p])

    # Make a scree plot 
    #pca.screeplot()
    # Let's calculate the error to interpret the closeness
    err = pca.error_ca()
    print('This is the percentage of error:', err)

    #Lets compare with scikitlearn
    scaler = StandardScaler()
    data_sk = scaler.fit_transform(data)

    pca_sk = PCA(n_components=0.7)
    la_data = pca_sk.fit_transform(data_sk)
    print("now with scikitlearn")
    print(pca_sk.explained_variance_)
    print("now mine")
    print(pca.vals_2)


if __name__ == '__main__':
    main()