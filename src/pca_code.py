import numpy as np
import utils

class PCA:
    def __init__(self, data):
        self.data = data

    def center_m(self):
        # Calculate the mean of each variable to centered the data, axis=0 columns
        self.data_mean = np.mean(self.data, axis=0, keepdims=True)

        # total or rows
        self.total_r = self.data.shape[0]

        # Center the data
        self.data_centered = self.data - self.data_mean

    def cov_matrix(self):
        # Calculate the covariance matrix
        self.m_cov = (1/(self.total_r - 1)) * self.data_centered.T @ self.data_centered

        # The covariance matrix describe the variance of each varaible and the covariance each two variables
        # This is important because we're able to select the linear combination of variables with more variance
        # We're able to construct a base of a space where the data is condense in perpendicular axes, this means, 
        # each data can be describe by a lienar combination of all the variables, and each component will have 
        # a sense of quality or characteristic of the data
    
    def eigen(self):
        # Obtanin the eigen-values and vectors to see the variance of each variable
        self.vals, self.vecs = np.linalg.eig(self.m_cov)
        # Sort the values and vector from the highest  
        idx = np.argsort(self.vals)[::-1] 
        self.vals = self.vals[idx]
        self.vecs = self.vecs[:, idx]

    def M_PCA(self):
        # Obtain the highest values
        self.vc, self.vc_p = utils.nin_vari(self.vals)
        # Create the matriz to change the base or to describe the data in the linear combination
        self.m_pca = self.vecs[:, :len(self.vc)] 
        return self.vc_p

    def new_data_cal(self):
        self.new_data = self.data_centered @ self.m_pca
        return self.new_data
    
    def error_ca(self):
        self.data_rebuild = self.new_data @ self.m_pca.T
        self.data_rebuild = self.data_rebuild + self.data_mean
        return np.sum(self.vals[len(self.vc):]) / np.sum(self.vals)
    
    def fit_transform(self):
        # execute pipeline
        self.center_m()
        self.cov_matrix()
        self.eigen()
        self.M_PCA()
        self.new_data_cal()
        self.error_ca()

        
