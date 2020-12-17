import numpy as np
import matplotlib.pyplot as mpl
import copy

#This function normalizes data, this means it scales the data to have a value between 0 and 1
#Normalizing the data is only done with the x_values/input_values/features, in logistic regression though the y or predicted values are always normalized by default
#Normalizing the data can be very helpful but is not always necessary
#Normalizing is necessary when you use different features with completely different value scales
#Normalizing data can help you increase the algorithm speed by making it converge quicker, this is useful when your data contains huge values
def minmax_normalization(x):
    i = 0
    max = float(np.max(x))
    min = float(np.min(x))
    range = max - min
    return np.divide(np.subtract(x, min), range)

def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x.transpose(), set_y.transpose())

#Used to split the data you have into training data and test data
#The training data should be used to train your algorithm so it can fit the theta values and test for underfitting
#The test data is used to detect underfitting, once you have a low cost for your test data you are good to go
def data_spliter(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return split_x_y(training_set) + split_x_y(test_set) #x_train = data[0], y_train = data[1], x_test = data[2], y_test = data[3]

#Used to facilitate calculations with matrices, theta_0 represents the y-intercept, set to 1 for following multiplication because x_values have no influence on theta0
def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

#Used to facilitate calculations with matrices for regularization
#Because theta0 as a bias and not a weight should not be implicated in regularization it is put to zero
def theta_0(theta):
    theta[0] = 0
    return theta

#To find the greatest descend, we use a derivative which gives us the slope of the line, based on the cost function graph
#The slope will naturally slow down when nearing the local minima and change direction when you go over the local minima
# It will thus nomatter what go towards the local minima nomatter how much steps are taken
# Here the linear_gradient is regularized meaning its bias/theta0/y-intercept is not used in calculations to avoid overfitting
def regularized_linear_gradient(expected_values, x_values, theta, lambda_):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    res = x_values.transpose().dot(np.subtract(x_values.dot(theta), expected_values))
    res = np.add(res, np.multiply(lambda_, theta_0(copy.deepcopy(theta))))
    return np.divide(res, lenght)

#Function used to perform regularization, theta0 as a bias and not a weight should not by implicated by regularization and thus is put to zero for matrix operation
#l2 is the most common regularization technque, dot product of theta without theta0 is returned and used by cost function to minimize the theta values
#l2 regularization is used by cost function to limit overfitting, by removing the bias/theta0/y-intercept
def l2_regularization(theta):
    theta[0] = 0
    return theta.transpose().dot(theta)

#The degree of a function is its highest exponent
#When using a function of degree 1, our prediction line is linear
#By increasing our function degree by adding exponentials on our x_values/input_values/features
#we can draw a non-straight line that fits the data better
#This is called polynomial features, each new polynomial feature also needs a weight or theta
#number of theta values = 1 + (features * power)
#using polynomial features makes a better fit and is necessary for some datasets but it increases the algorithms complexity and time
#Polynomial features also increase, the chance of overfitting, regularization can help combat that
#Usually used with a power between one and 10, look at plotted data and experiment to find ideal power
def add_polynomial_features(x, power):
    power = range(1, power + 1)
    init = x
    for pow in power:
        x = np.column_stack((x, init ** pow))
    return x

#Linear regression is used to predict values based on a dataset,
#it does this by creating a function that adapts itself to fit the data
#The function is made of one bias(theta0) and features with each their appropriate weight(theta_n)
class linear_regression:
#To perform linear regression you need data, this data can be split in:
#y or expected_values or the correct answers
#x or the features, those values are used to predict an answer
#predicted values, are the values you predicted with your current algorithm
    def __init__(self, thetas, alpha=0.0001, n_cycle=1000000, lambda_=0):
        #The theta values can be seen as weights of the features, besides theta_0 as a bias
        self.theta = thetas
        #Alpha is used as a weight to calculate the step size of the gradient descend,
        #(stepsize is calculated with alpha * gradient) gradient descend is somewhat sensitive to the alpha
        #because gradient descend uses the slope that naturally goes towards 0 when coming close to the local minma, gradient descend can by itself modulate the stepsize, by decreasing stepsize as it comes closer to convergence
        #When alpha is too small algorithm needs to perform more steps until convergence and become slower
        #When alpha is too big potentially no convergence(finding the local minima) will occur, because it will go over the local minima
        #The ideal size of alpha is dependent upon the dataset, thus you need to test by yourself, what alpha value is best
        self.alpha = alpha
        #n_cycle represents the number of steps that will be performed to converge(find local minima)
        #Too much can unnecessarily slow down the algorithm, too few and the algorithm won't converge
        #The ideal n_cycle and alpha is dependent on the dataset and needs to be tested
        self.n_cycle = n_cycle
        #Is the weight used for regularization, regularization is useful for polynomial models that suffer from overfitting.
        #Overfitting is caused by acting on irrelevant signals, regularization works by reducing the weight on certain features
        #We do this by adding the dot_product of theta to the cost function, which will minimize the theta values
        #theta0 as a bias is not regularized only weights are.
        #By using lambda 0 you use no regulariztion, and by using a higher lambda value you drive thetas to zeros
        #Usually lambda is used with a value between 0 and 1
        self.lambda_ = lambda_


#To find the best fit of theta values, imagine a graph of the cost function with one local minima,
#we start somewhere randomly on the graph and need to move towards the local minima,
#to do this we use gradient descend that finds the steepest descend to go towards the local miniama
#Each theta has own local minima and needs to go through the gradient descend algorithm
    def fit_(self, x_values, expected_values):
        for x in range(0,self.n_cycle):
            gradient = regularized_linear_gradient(expected_values, x_values, self.theta, self.lambda_)
            self.theta = np.subtract(self.theta, (np.multiply(self.alpha, gradient)))
        return self.theta

#The cost function is there to find the error rate or loss, a lower cost is better
#An extremely low cost on the training set, will maybe equal a higher cost on the test set, this is overfitting
#Overfitting means the thetas are too well adapted/specialized to the training data, while it should be adapted to all data
#A high cost on training set means underfitting, which means the theta values are adapted to no data at all.
    def cost_(self, predicted_values, expected_values):
        res = np.subtract(predicted_values, expected_values)
        res = res.transpose().dot(res)
        res = np.add(res, np.multiply(self.lambda_, l2_regularization(copy.deepcopy(self.theta))))
        return np.divide(res, (2 * predicted_values.shape[0]))

#This function makes the predicted values(y_hat), it does this by letting the features or input_variables go through the
#function created with the linear regression
    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
        return input_variables.dot(self.theta)

#To detect underfitting and overfitting it is important to be able to visualize the data
#This function plots, in point format the expected values or y and in line format your predictions or y_hat
#It also shows the associated cost as a title
#To detect underfitting plot the training data, and to detect overfitting plot the test data
    def plot_(self, x_values, predicted_values, expected_values, cost):
        mpl.plot(x_values, predicted_values, color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show() #SEGFAULT on new mac update "Big Sur", use VM
