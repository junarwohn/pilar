import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

labels = []
points = []
with open("log_data.txt", "r") as f:
    for line in f:
        d = line.replace(" ", "").split(",")
        labels.append(d[0])
        points.append([float(d[1].split(":")[-1]), float(d[2].split(":")[-1])])

# second data point
x_same = []
y_same = []

for p, l in zip(points, labels):
    if l == 'SAME':
        x_same.append(p[0])
        y_same.append(p[1])

 
# depict second scatted plot
plt.scatter(x_same, y_same, c='red')

x_diff = []
y_diff = []
for p, l in zip(points, labels):
    if l == 'DIFF':
        x_diff.append(p[0])
        y_diff.append(p[1])

plt.scatter(x_diff, y_diff, c='blue')

plt.show()

filename = 'svm_models.sav'

        
x = x_same + x_diff
y = y_same + y_diff

X = np.array([[i, j] for i, j in zip(x, y)])
y = np.array([1 for i in x_same] + [0 for i in x_diff])
"""
C = 1   # SVM의 regularization parameter
clf = svm.SVC(kernel = "linear", C=C)
clf.fit(X, y)
from sklearn.metrics import confusion_matrix    # confusion_matrix라이브러리
y_pred = clf.predict(X)                         # 학습데이터 분류예측
print(y_pred)
print(confusion_matrix(y, y_pred))                     # 정확성검정



 
C = 1.0 #regularization parameter
mods = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X, y) for clf in mods)
models_sav = [clf.fit(X, y) for clf in mods]
pickle.dump(models_sav, open(filename, 'wb'))

"""
mods = pickle.load(open(filename, 'rb'))
models = (clf for clf in mods)

print(mods[0].predict(X[:]))
print(y)

# plot title 형성
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# plot 그리기

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

