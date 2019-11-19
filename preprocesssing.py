import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


datas = np.load('E_origninal.npy')
numero_senseur = np.load('numeroDesCapteur.npy')
voisin = np.load('tousLesVoisinsDeTouslesPionts.npy',allow_pickle = True)
p = 0.4
sensor = voisin[0]
n,m = datas.shape

def imputation_precedent(data):
    # naive imputation by the precedent value
    # return data imputated in pandas DataFrame format
    m,n = data.shape
    for i in range(m):
        for j in range(n):
            if data[i,j] == -1:
                data[i,j] = data[i,j-1]
    datas_pandas = pd.DataFrame(data)
    datas_pandas.index = numero_senseur
    return datas_pandas


def neighbor_data(data):
    # creating a table of data of the no.i sensor and its neighbor sensors' data
    model_cap = pd.DataFrame(columns=np.arange(datas.shape[1]))
    for ind in range(len(sensor)):
        model_cap.loc[ind] = data.loc[sensor[ind]]

    # numerate neighbor sensor and efface their sensor number
    model_cap.astype(int)
    return model_cap

def noise_to_data(model_cap,p,mu_perc = 0.5,sigma_perc=0.3):
    # adding gaussian noise manually to a sensor
    # sampling of p% of model_sensors and creation of label
    np.random.seed(42)
    model_sensor = model_cap.iloc[0, :].copy()
    sample = np.random.choice(range(model_sensor.shape[0]), int(p * model_sensor.shape[0]), replace=False)
    sample = np.unique(sample)
    label = np.zeros(model_sensor.shape)

    # adding gaussien noise to p% of the model_capet, mu mean, sigma ecart-type
    mu = mu_perc * np.mean(model_sensor)
    sigma = sigma_perc * np.mean(model_sensor)
    for i in range(len(sample)):
        label[sample[i]] = 1
        model_sensor[sample[i]] += np.random.normal(mu, sigma)

    # replace sensor data in the original data
    model_noise = model_cap.copy()
    model_noise.iloc[0, :] = model_sensor
    return model_noise, label

def feature_difference(model_cap, model_noise):
    # replace feature in data by difference between the center sensor and actual sensor and difference square
    model_diff = pd.DataFrame(columns=np.arange(datas.shape[1]))
    model_noise_diff = pd.DataFrame(columns=np.arange(datas.shape[1]))
    for ind in range(1, len(sensor)):
        # model_diff.loc['cap{}-cap0'.format(str(ind))] = np.abs(datas_pandas.loc[cap[ind]] - datas_pandas.iloc[0])
        model_diff.loc['cap{}-cap0'.format(str(ind))] = (model_cap.loc[ind] - model_cap.iloc[0])
        model_noise_diff.loc['cap{}-cap0'.format(str(ind))] = (model_noise.loc[ind] - model_noise.iloc[0])
        # model_diff.loc['cap{}-cap0 square'.format(str(ind))] = (datas_pandas.loc[cap[ind]]
        # - datas_pandas.iloc[0])**2
    model_diff = model_diff.astype(int)
    model_diff = model_diff.transpose()
    model_noise_diff = model_noise_diff.astype(int)
    model_noise_diff = model_noise_diff.transpose()
    return model_diff, model_noise_diff


def main(data):
    data = imputation_precedent(data)
    model_cap = neighbor_data(data)
    model_noise, label = noise_to_data(model_cap, p)
    model_diff,model_noise_diff = feature_difference(model_cap, model_noise)
    return model_diff, model_noise_diff,label

model_diff, model_noise_diff, label = main(datas)