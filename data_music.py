import xlrd
import numpy as np
# print("error !! data fixed")
# exit(0)
data_root = "data_music/"

d= 66
n0 = 600 # number of sample
n1 = 400#  no noise
n = n1 + n0
k = n0/n
k0 = np.ones(5) * k
print(k0)
s_1 = np.random.rand(n0)-0.5
s_2 = np.zeros(n1)
s = np.hstack((s_1, s_2))
np.random.shuffle(s)
b = np.ones(n)
s_b = [0] * n
y = np.ones((n,2))
X = np.ones((n,d))
print(X.shape)

f = open(data_root+'music.txt')
read = f.readlines()
result_0 = []
for item in read:
    reads = item.strip()
    temp = reads.split(",")
    result_0.append(temp)

for i in range(n):
    y[i] = result_0[i][68:]

for i in range(n):
    for j in range(d):
        X[i,j] = result_0[i][j]

ny = np.linalg.norm(y)
for i in range(n):
    if s[i] > 0:
        s_b[i] = 1
        b[i] =np.random.rand(1) * ny
    elif s[i]==0:
        b[i] = 0
        s_b[i] = 0
    else:
        b[i] = (np.random.rand(1)-1) * ny
        s_b[i] = -1


# w_s = np.random.normal(loc=0, scale=1000, size=d)
# err = np.random.normal(loc=5, scale=10, size=n)
# y = np.dot(X,w_s) + err
# for i, row in enumerate(read):
#     if i <=5 and i>=0:
#         print(row)



# csv_reader = csv.reader(open("super_train.csv"))
# for i, row in enumerate(csv_reader):
#     if i <=n and i>=1:
#         X[i-1]= row
#
#
b = b[...,np.newaxis]
print b.shape
y_b = y + b
#
np.savetxt(data_root+'sample.txt', X, fmt='%0.8f')
np.savetxt(data_root+'label.txt', y, fmt='%0.8f')
np.savetxt(data_root+'label_b.txt', y_b, fmt='%0.8f')
# np.savetxt('w_s.txt', w_s, fmt='%0.8f')
np.savetxt(data_root+'s_b.txt', s_b, fmt='%0.8f')
np.savetxt(data_root+'k_hard.txt', k0, fmt='%0.8f')
