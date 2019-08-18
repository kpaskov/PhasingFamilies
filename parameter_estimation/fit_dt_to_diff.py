import sys
import os
import scipy.stats
import numpy as np
from itertools import product

train_dir = sys.argv[1] #'data/train_with_missing'
prediction_dir = sys.argv[2] #'predictions/robustgm'
out_dir = sys.argv[3] #'predictions/robustgm+tree+sigma'
depth = int(sys.argv[4]) #6

labtests = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 
            'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

# ----------------------------------------------------------------------
# Load data
train_diff = []
pred_sigma = []
num_datapoints = []

filenames = set(os.listdir(prediction_dir))
filenames = [x for x in filenames if x.endswith('.csv') and x[:-4] + '_sigma.csv' in filenames]
print('Patients', len(filenames))

for filename in sorted(filenames, key=lambda x: int(x[:-4])):
    # each file represents a patient
    patient = filename[:-4]
        
    # read data
    train_data = np.genfromtxt('%s/%s' % (train_dir, filename), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    timepoints = train_data[:, 0].astype(int)
    train_data = train_data[:, 1:]
        
    prediction_data = np.genfromtxt('%s/%s' % (prediction_dir, filename), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    prediction_data = prediction_data[:, 1:]
    
    prediction_sigma = np.genfromtxt('%s/%s_sigma.csv' % (prediction_dir, patient), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    prediction_sigma = prediction_sigma[:, 1:]
    
    # calculate the number of training outliers for each timepoint
    diff = train_data-prediction_data
    diff[np.isnan(train_data)] = np.nan
    train_diff.append(diff)
    pred_sigma.append(prediction_sigma)
    num_datapoints.append(np.repeat(np.sum(~np.isnan(train_data), axis=0), timepoints.shape[0]))
    
train_diff = np.vstack(train_diff)
pred_sigma = np.vstack(pred_sigma)
num_datapoints = np.vstack(num_datapoints)

print(train_diff.shape, pred_sigma.shape, num_datapoints.shape)

indices = np.all(train_diff != 0, axis=1)
train_diff_nomissing = train_diff[indices, :]
pred_sigma_nomissing = pred_sigma[indices, :]
print(train_diff_data_nomissing.shape, pred_sigma_nomissing.shape)

# ----------------------------------------------------------------------
# Model

models = []
nonzero_coeffs = []

for index, labtest in enumerate(labtests):
    print(labtest)

    # prepare training data
    train_data_X = np.hstack((train_diff_nomissing[:, [i for i in range(train_diff_nomissing.shape[1]) if i != index]], pred_sigma_nomissing, num_datapoints[:, index]))
    train_data_y = train_diff_nomissing[:, index]

    # fit decision tree
    model = sklearn.tree.DecisionTreeRegressor(max_depth=depth, min_samples_leaf=10, criterion='mse')
    model.fit(train_data_X, train_data_y)

    # important coefficients
    nonzero_coefficients = model.feature_importances_ != 0
    print('Nonzero coefficients', np.sum(nonzero_coefficients))
    
    models.append(model)
    nonzero_coeffs.append(nonzero_coefficients)
    
    # predict
    train_p = model.predict(train_data_X)
    test_p = model.predict(test_data_X)

# ----------------------------------------------------------------------
# Write to file

for n, filename in enumerate(sorted(filenames, key=lambda x: int(x[:-4]))):
    # each file represents a patient
    patient = filename[:-4]
        
    # ----------------------------------------------------------------------
    # Load data

    # read data
    train_data = np.genfromtxt('%s/%s' % (train_dir, filename), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    timepoints = train_data[:, 0].astype(int)
    train_data = train_data[:, 1:]
        
    prediction_data = np.genfromtxt('%s/%s' % (prediction_dir, filename), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    prediction_data = prediction_data[:, 1:]
    
    prediction_sigma = np.genfromtxt('%s/%s_sigma.csv' % (prediction_dir, patient), delimiter=',', skip_header=1, missing_values=['NA'], filling_values=np.nan)
    prediction_sigma = prediction_sigma[:, 1:]
    
    diff_data = train_data - prediction_data
    num_datapoints = np.repeat(np.sum(~np.isnan(train_data), axis=0), timepoints.shape[0])
    
    new_prediction_data = prediction_data.copy()
     
    for j in range(len(labtests)): 
        model = models[j]
        features = np.hstack((diff_data[:, [k for k in range(len(labtests)) if k != j]], prediction_sigma, num_datapoints[:, j]))
        features_nomissing = features.copy()
        features_nomissing[np.isnan(features)] = 0
        predictions = model.predict(features_nomissing)
        new_prediction_data[:, j] = new_prediction_data[:, j] + predictions
        
    # write to file
    np.savetxt('%s/%s.csv' % (out_dir, patient), np.hstack((timepoints[:, np.newaxis], new_prediction_data)), delimiter=',', 
            header=','.join(['CHARTTIME']+labtests), fmt=['%d'] + ['%f']*len(labtests))

    if n%200 == 0:
        print(n)



