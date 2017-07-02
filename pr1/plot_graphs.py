import pickle
import matplotlib.pyplot as plt

times_n1 = pickle.load(open('./outcomes/bowt_t1.pk', 'rb'))
times_n4 = pickle.load(open('./outcomes/bowt_t4.pk', 'rb'))
z = list(range(20,50,5))

plt.legend()
plt.grid()

plt.plot(z, times_n1, '-or', label='single thread')
plt.plot(z, times_n4, '-ob', label='four threads')
plt.show()

perfs = pickle.load(open('./outcomes/acc.pk', 'rb'))
z = list(range(50,150,5))
plt.xlabel('# centroids')
plt.ylabel('performance')
plt.grid()
plt.plot(z, perfs, '-ob', label='performance')
plt.show()


