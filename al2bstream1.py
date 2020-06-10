import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import load_iris

data = pd.read_csv('devindex.csv')
scaler = StandardScaler()
for col in data.columns.tolist()[:-1]:
    temp = pd.DataFrame(data[col])
    temp = scaler.fit_transform(temp)
    data[col] = temp

print(data.head())

# creating the image
# im_width = 500
# im_height = 500
# im = np.zeros((im_height, im_width))
# im[100:im_width - 1 - 100, 100:im_height - 1 - 100] = 1

# create the data to stream from
# X_full = np.transpose(
#     [np.tile(np.asarray(range(im.shape[0])), im.shape[1]),
#      np.repeat(np.asarray(range(im.shape[1])), im.shape[0])]
# )
# # map the intensity values against the grid
# y_full = np.asarray([im[P[0], P[1]] for P in X_full])

# # create the data to stream from
# X_full = np.transpose(
#     [np.tile(np.asarray(range(im.shape[0])), im.shape[1]),
#      np.repeat(np.asarray(range(im.shape[1])), im.shape[0])]
# )
# # map the intensity values against the grid
# y_full = np.asarray([im[P[0], P[1]] for P in X_full])

X_raw = np.array(data.iloc[:, :-1])
y_raw = np.array(data.iloc[:, -1])
# print(X_full, y_full)

dataset = load_iris()
X_raw = dataset['data']
y_raw = dataset['target']

import matplotlib as mpl
import matplotlib.pyplot as plt

'''
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.imshow(im)
    plt.title('The shape to learn')
    plt.show()
'''

from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

percent = 0.1
n_labeled_examples = X_raw.shape[0] # len
training_indices = np.random.randint(low=0, high=n_labeled_examples, size=int(n_labeled_examples * percent))

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

X_stream = np.delete(X_raw, training_indices, axis=0)
y_stream = np.delete(y_raw, training_indices, axis=0)

# assembling initial training set
# n_initial = 5
# initial_idx = np.random.choice(range(len(X_full)), size=n_initial, replace=False)
# X_train, y_train = X_full[initial_idx], y_full[initial_idx]

# initialize the learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_train, y_training=y_train
)
unqueried_score = learner.score(X_raw, y_raw)

print('Initial prediction accuracy: %f' % unqueried_score)

'''
# visualizing initial prediciton
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = learner.predict_proba(X_full)[:, 0]
    plt.imshow(prediction.reshape(im_width, im_height))
    plt.title('Initial prediction accuracy: %f' % unqueried_score)
    plt.show()
'''

from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy

performance_history = [unqueried_score]

num_queried = 0

# learning until the accuracy reaches a given threshold
while num_queried < 20:
    stream_idx = np.random.choice(range(len(X_stream)))
    print(stream_idx)
    if classifier_uncertainty(learner, X_stream[stream_idx].reshape(1, -1)) >= 0.4:
        learner.teach(X_stream[stream_idx].reshape(1, -1), y_stream[stream_idx].reshape(-1, ))
        new_score = learner.score(X_stream, y_stream)
        performance_history.append(new_score)
        print('Pixel no. %d queried, new accuracy: %f' % (stream_idx, new_score))
        num_queried += 1

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

plt.show()