from __future__ import print_function
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from utils import SZMO,HARD,OLS

data_root = "data_music/"
def main():

    d = 66  # dim of data,m,no-changeable
    n = 500 # number of sample changeable

    fold = 5
    y_index = 0
    alpha = 0.5
    # load data
    #x samples
    x_data = np.loadtxt(data_root+'sample.txt',dtype=np.float32)
    x_data = x_data[0:n,0:d]

    #perturbation data
    #y = y* + \epsilon + b
    y_perturbation_data = np.loadtxt(data_root+'label_b.txt',dtype = np.float32)
    y_perturbation_data = y_perturbation_data[0:n,y_index]
    y_perturbation_data = y_perturbation_data

    #y = y*
    y_standard_data = np.loadtxt(data_root+'label.txt',dtype = np.float32)
    y_standard_data = y_standard_data[0:n,y_index]



    #sign information of perturbation b
    perturbation = np.loadtxt(data_root+'s_b.txt',dtype = np.float32)
    perturbation = perturbation[0:n]


    #k for hard
    k = np.loadtxt(data_root+'k_hard.txt',dtype = np.float32)

    y_data = [y_perturbation_data, y_standard_data,perturbation]
    y_data = np.array(y_data)
    y_data = y_data.T


    y_loss = np.zeros([fold,6])

    for ii in range(fold):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.5)
        l, m = np.shape(x_train)
        #print("l=", l)
        print("fold:%d"%ii)

        #sign information of perturbation b
        s_train = y_train[:,2]
        #training data label with perturbation b
        y_train= y_train[:,0]
        y_test = y_test[:,1]
        mean = np.mean(y_test)

        # SZMO
        w_szmo = SZMO(x_train,y_train,s_train)
        #print "szmo:{0:.4}".format(np.linalg.norm(w_optimal-w_szmo,ord=2))

        y_predict = np.dot(x_test, w_szmo)
        y_loss[ii,0] = np.linalg.norm(y_predict-y_test,ord=2)/l

        #ols
        w_ols = OLS(x_train,y_train)
        #print "ols:{0:.4}".format(np.linalg.norm(w_optimal-w_ols,ord=2))

        y_predict = np.dot(x_test, w_ols)
        y_loss[ii,1] = np.linalg.norm(y_predict-y_test,ord=2)/l


        #HARD
        w_hard = HARD(x_train,y_train,k)
        #print "hard:{0:.4}".format(np.linalg.norm(w_optimal-w_hard,ord=2))

        y_predict = np.dot(x_test, w_hard)
        y_loss[ii,2] = np.linalg.norm(y_predict-y_test,ord=2)/l

        # Ridge
        clf = Ridge(alpha=0.3)
        clf.fit(x_train, y_train)
        w_r = clf.coef_
        #print "Ridge:{0:.4}".format(np.linalg.norm(w_optimal-w_r,ord=2))

        y_predict = np.dot(x_test, w_r)
        y_loss[ii,3] = np.linalg.norm(y_predict-y_test,ord=2)/l

        # Lasso
        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(x_train, y_train)
        w_lasso = reg.coef_
        #print "lasso:{0:.4}".format(np.linalg.norm(w_optimal-w_lasso,ord=2))

        y_predict = np.dot(x_test, w_lasso)
        y_loss[ii,4] = np.linalg.norm(y_predict-y_test,ord=2)/l

        # Huber
        huber = linear_model.HuberRegressor()
        huber.fit(x_train, y_train)
        w_huber = huber.coef_
        #print "huber:{0:.4}".format(np.linalg.norm(w_optimal-w_huber,ord=2))

        y_predict = np.dot(x_test, w_huber)
        y_loss[ii,5] = np.linalg.norm(y_predict-y_test,ord=2)/l

        #     k = np.loadtxt('k_hard.txt',dtype = np.float32)
    #     num = int( k[0] * l)
    #     print("num= ", num)
        #Assign the sign value of perturbation b according to b values.
    #with shape[5,6]
    #w_diff = np.linalg.norm(w_optimal - w_predict,ord = 2,axis = 2)
    print ("m:%d,n:%d,alpha %.1f,index:%d"%(d,n/2,alpha,y_index))
    print ("attention:[alpha should be Manually changed based on generated files]")
    print ("              SZMO    OLS   HARD   Ridge  LASSO   Huber")
    #y_diff = np.linalg.norm(w_optimal - w_predict,ord = 2,axis = 2)
    mean_y_loss = y_loss.mean(axis=0)/10
    var_y_loss = y_loss.var(axis=0)/10
    print ("mean_y_loss:",end=' ')
    for i in mean_y_loss:
        print("%.4f" %i,end=',')
    print("")
    print ("var_y_loss :",end=' ')
    for i in var_y_loss:
        print("%.4f" %i,end=',')
    print("")

if __name__ == '__main__':
    main()
