import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.interpolate import interp1d

# Function that creates the X matrix as defined for fitting our model
def create_X(x,deg):
    X = np.ones((len(x),deg+1))
    for i in range(1,deg+1):
        X[:,i] = x**i
    return X

# Function for predicting the response
def predict_y(x,beta):
    return np.dot(create_X(x,len(beta)-1),beta)

# Function for fitting the model
def fit_beta(df,deg):
    return np.linalg.lstsq(create_X(df.x,deg),df.y,rcond=None)[0]

# Function for computing the MSE
def rmse(y,yPred):
    se = (y-yPred)**2
    return np.sqrt(np.mean(se))

# Function for computing training and validation error as function of degree.
def computeError(maxDegree,dfTrain,dfVal):
    # Initializing range of degree values to be tested and errors
    deg = list(range(0,maxDegree+1))
    errTrain = np.zeros(len(deg))
    errVal = np.zeros(len(deg))

    # Computing training and validation RMSE errors for each corresponding
	# degree value.
	# TODO: Implement me
	for d in deg:
		beta = fit_beta(dfTrain, d)
		yPredTrain = predict_y(dfTrain.x, beta)
		errTrain[d] = rmse(dfTrain.y, yPredTrain)
		yPredVal = predict_y(dfVal.x, beta)
		errVal[d] = rmse(dfVal.y, yPredVal)

    return {'deg':deg,'errTrain':errTrain,'errVal':errVal}

def plotError(err):
    # Creating interpolation function
    tmp = np.linspace(err['deg'][0],err['deg'][-1],2*len(err['deg'])-1)
    tmpF = interp1d(err['deg'],err['errTrain'],kind='linear')
    intTrain = interp1d(tmp, tmpF(tmp), kind='cubic')
    tmpF = interp1d(err['deg'],err['errVal'],kind='linear')
    intVal = interp1d(tmp,tmpF(tmp),kind='cubic')
    
    # Plotting results
    intDeg = np.linspace(err['deg'][0],err['deg'][-1],100)
    plt.plot(err['deg'],err['errTrain'],'bs',err['deg'],err['errVal'],'rs')
    plt.plot(intDeg,intTrain(intDeg),'b-',intDeg,intVal(intDeg),'r-')

    plt.legend(('Training Error','Validation Error'))
    plt.show()
