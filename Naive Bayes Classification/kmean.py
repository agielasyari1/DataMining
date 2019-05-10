import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv ('kmean.csv')
print("Input Data dan Shape")
print(data.shape)
data.head()

f1 = data ['v1'].values
f2 = data ['v2'].values
X = np.array(list(zip(f1, f2)))

k = 2
cx = np.random.randint(0, np.max(X)-20, size=k)
cy = np.random.randint(0, np.max(X)-20, size=k)
c = np.array(list(zip(cx, cy)), dtype=np.float32)

plt.scatter(f1, f2, c='#050505', s=20)
plt.scatter(cx, cy, marker='*', s=200, c='g')
plt.xlabel("Distance Feature")
plt.ylabel("Speeding Feature")
plt.title('Raw Delivery Fleet Data')

kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'o', 'w']
fig2 = plt.figure()
kx = fig2.add_subplot(111)

for i in range(k):
    point = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    kx.scatter(point[:, 0], point[:,1], s=20, cmap='rainbow')
kx.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
print("Final Centroids")
print(centroids)
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('Number of clusters={}'.format(k))

krange = range(1,10)
distortions = []
for i in krange:
    kmean_model = kmeans(n_clusters=i)
    kmean_model.fit(X)
    distortions.append(sum(np.min(cdist(X, kmean_model.cluster_center_, 'euclidean'), axis=1)) / X.shape[0])
fig1 = plt.figure()
ex = fig1.add_subplot(111)
ex.plot(krange, distortions, 'b*-')

plt.grid(True)

plt.ylim([0,45])
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('Selection k with elbow method    ')