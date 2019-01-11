import numpy as np
e = 0.0001

def SZMO(x_train,y_train,s_train):
    l, m = np.shape(x_train)
    x_train = x_train.T
    b_sign = [0] * l
    for i in range(l):
        if s_train[i] > 0:
            b_sign[i] = 1
        elif s_train[i] == 0:
            b_sign[i] = 0
        else:
            b_sign[i] = -1

    b0 = [0] * l
    b1 = [0.1] * l
    b2 = [0] * l
    b =  [0] * l


    #P = X^T * (X * X^T) ^ -1 * X
    p_inv = np.linalg.inv(np.dot(x_train,x_train.T))
    p = np.dot(np.dot(x_train.T,p_inv),x_train)

    I = np.eye(l, dtype=int)
    j = 0

    #while np.linalg.norm(np.array(b1)-np.array(b0), ord=2) >= e:
    b0 = b1
    m0 = np.dot(p, b2)
    m1 = I - p
    m2 = np.dot(m1, y_train)
    m = m0 + m2
    #set-zero-operation
    for i in range(l):
        if b_sign[i] == 1:
            if m[i] <= 0:
                b[i] = 0
            else:
                b[i] = m[i]
        elif b_sign[i] == 0:
            b[i] = 0
        else:
            if m[i] >= 0:
                b[i] = 0
            else:
                b[i] = m[i]
    b1 = b

    # our method
    w_SZMO = np.dot(np.dot(p_inv,x_train),y_train-b1)
    return w_SZMO

def HARD(x_train,y_train,k):
    #hard
    l, m = np.shape(x_train)
    num = int( k[0] * l)
    x_train = x_train.T
    p_inv = np.linalg.inv(np.dot(x_train,x_train.T))
    p = np.dot(np.dot(x_train.T,p_inv),x_train)


    b0_h = [0] * l
    b1_h = [1] * l
    b2_h = [0] * l

    b0 = np.array(b0_h)
    b_h = np.array(b0_h)

    I = np.eye(l, dtype=int)
    j_h = 0
    while np.linalg.norm(np.array(b1_h)-np.array(b0_h), ord=2) >= e:
        b0_h = b1_h
        if j_h == 0:
            m0 = np.dot(p, b2_h)
        else:
            m0 = np.dot(p, b0_h)
        m1 = I - p
        m2 = np.dot(m1, y_train)
        m = m0 + m2
        #print type(b0_h)
        r_h = np.argsort(-b0)
        for i in range(num):
            b_h[r_h[i]] = 0
        b1_h = b_h
        j_h = j_h+1
    #print j_h
    #  hard
    w_hard = np.dot(np.dot(p_inv,x_train),y_train - b1_h)
    return w_hard
def OLS(x_train,y_train):
    x_train = x_train.T
    p_inv = np.linalg.inv(np.dot(x_train,x_train.T))

    w_ols = np.dot(np.dot(p_inv,x_train), y_train)
    return w_ols
