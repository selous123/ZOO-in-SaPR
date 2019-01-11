import numpy as np
def main():
	data_root = "data_generator/"
	# generator data and noise
	mu, sigma = 1, 3  # gaussion parameter
	er = 1
	d = 600   # dim of data
	n0 = 5000  # number of sample
	n1 = 5000  #  no noise
	n = n1 + n0
	k = n0 / n
	k0 = np.ones(5) * k
	# init parameter, generator w*
	y = np.arange(n)
	w_s = np.random.rand(d)
	# w_s = np.random.normal(loc=3, scale=100, size=d)
	nw = np.linalg.norm(w_s)
	w_s = w_s/nw * d
	err = np.random.normal(loc=0, scale=er, size=n)
	# generator matrix X
	x = np.random.normal(loc=mu, scale=sigma, size=(n, d))
	# generator vector y
	y_x = np.dot(x,w_s)
	y = np.dot(x,w_s) + err
	b = np.ones(n)
	s_b = [0] * n
	# generator N and P label
	s_1 = np.random.rand(n0)-0.5
	s_2 = np.zeros(n1)
	s = np.hstack((s_1, s_2))
	np.random.shuffle(s) # shuffle s
	print(s)
	for i in range(n):
	    if s[i] > 0:
		s_b[i] = 1
		b[i] =np.random.rand(1) * nw
	    elif s[i]==0:
		b[i] = 0
		s_b[i] = 0
	    else:
		b[i] = (np.random.rand(1)-1) * nw
		s_b[i] = -1

	y_b = y + b
	print(y_b.shape,"yb")
	print(y.shape,"y")
	print(b.shape,"b")
	# save data
	np.savetxt(data_root+'sample.txt', x, fmt='%0.8f')
	np.savetxt(data_root+'label.txt', y_x, fmt='%0.8f')
	np.savetxt(data_root+'label_b.txt', y_b, fmt='%0.8f')
	np.savetxt(data_root+'w_s.txt', w_s, fmt='%0.8f')
	np.savetxt(data_root+'s_b.txt', s_b, fmt='%0.8f')
	np.savetxt(data_root+'k_hard.txt', k0, fmt='%0.8f')
if __name__ =="__main__":
	main()
