import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling
from sklearn.ensemble import RandomForestClassifier
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy
from modAL.disagreement import vote_entropy, KL_max_disagreement


def random_sampling(classifier, X, n_instances=1):

  query_index = np.random.randint(low=0, high=len(X), size=1)
  return query_index, X[query_index]


def streamify(qs):

  def fun(classifier, X, n_instances=1):

    count = 0

    while (True):
      count+= 1
      stream_idx = np.random.choice(range(len(X)))

      if qs(classifier, X[stream_idx].reshape(1, -1)) >= 0.4 or count > 300 :
        return stream_idx, X[stream_idx]

  return fun


def get_learner(query_strategy):

  global X_train, y_train

  # Specify our core estimator along with it's active learning model.
  learner = ActiveLearner(estimator=RandomForestClassifier(),
                          query_strategy=query_strategy,
                          X_training=X_train, y_training=y_train)

  return learner


def get_committee(query_strategy):

  global X_train, y_train

  # initializing Committee members
  n_members = 5
  learner_list = list()

  for member_idx in range(n_members):
      # initializing learner
      learner = ActiveLearner(
          estimator=RandomForestClassifier(),
          X_training=X_train, y_training=y_train
      )
      learner_list.append(learner)

  # assembling the committee
  committee = Committee(learner_list=learner_list,
                          query_strategy=query_strategy)

  return committee


def plot_predictions(learner, percent, query_strategy):

  global X_raw, y_raw

  # Isolate the data we'll need for plotting.
  predictions = learner.predict(X_raw)
  is_correct = (predictions == y_raw)

  # Plot our classification results.
  fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
  ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',   alpha=8/10)
  ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
  ax.legend(loc='lower right')
  ax.set_title(str(percent * 100) + " percent labelled " + query_strategy + " class predictions (Accuracy: {score:.3f})".format(score= (is_correct.sum()/ len(y_raw))))
  plt.savefig(str(percent * 100) + " percent labelled " + query_strategy + " predictions.png")
  plt.show()


def sampling(learner, num_queries, performance):

  global X_pool, y_pool, X_raw, y_raw

  # Allow our model to query our unlabeled dataset for the most
  # informative points according to our query strategy.
  for index in range(num_queries):

    query_index, query_instance = learner.query(X_pool)
    print(query_index)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    model_accuracy = learner.score(X_raw, y_raw)
    print('Accuracy after query {n}: {acc:0.4f}\n'.format(n=index + 1, acc=model_accuracy))

    # Save our model's performance for plotting.
    performance.append(model_accuracy)

  return performance


def plot_accuracies(performance_history, fig, ax, label ):

  ax.plot(performance_history, label = label)
  ax.scatter(range(len(performance_history)), performance_history, s=13)

  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
  ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

  ax.set_ylim(bottom=min(performance_history), top=1)
  ax.grid(True)

  ax.set_title('Iterative Accuracy')
  ax.set_xlabel('Query Iteration')
  ax.set_ylabel('Classification Accuracy')

def comparison_plot(performances, labels):

  fig = plt.figure()
  ax = plt.subplot(111)

  label = ""

  for i in range(len(performances)):
    plot_accuracies(performances[i], fig, ax, labels[i])
    label = label + labels[i] + '_'

  ax.legend()
  plt.show()
  fig.savefig("Accuracies_" + label + ".png")


def main(query_strategy, str_query_strategy, is_committee):

  global X_pool, y_pool, training_indices
  # Isolate the non-training examples we'll be querying.
  X_pool = np.delete(X_raw, training_indices, axis=0)
  y_pool = np.delete(y_raw, training_indices, axis=0)

  if is_committee == False:
    learner = get_learner(query_strategy)
  else:
    learner = get_committee(query_strategy)

  plot_predictions(learner, 0.1, str_query_strategy)

  # Record our learner's score on the raw data.
  unqueried_score = learner.score(X_raw, y_raw)
  performance = [unqueried_score]

  performance = sampling(learner, int(4 * len(X_raw) / 10), performance)

  plot_predictions(learner, 0.5, str_query_strategy)
  return performance


# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

dataset = load_iris()
X_raw = dataset['data']
y_raw = dataset['target']
num_classes = 3

str_qs = ["Least Confident", "Margin", "Entropy"]
qs_pool = [uncertainty_sampling, margin_sampling, entropy_sampling]
qs_stream = [streamify(classifier_uncertainty), streamify(classifier_margin), streamify(classifier_entropy)]

str_qs_qbc = ["Vote Entropy", "KL Divergence"]
qs_pool_qbc = [vote_entropy_sampling, max_disagreement_sampling]
qs_stream_qbc = [streamify(vote_entropy), streamify(KL_max_disagreement)]



# -------------for visualization---------------
# # Define our PCA transformer and fit it onto our raw dataset.
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_dataset = pca.fit_transform(X=X_raw)

# Isolate the data we'll need for plotting.
x_component, y_component = transformed_dataset[:, 0], transformed_dataset[:, 1]

#Plot our dimensionality-reduced (via PCA) dataset.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
plt.title('Dataset classes after PCA transformation')
plt.savefig("Dataset.png")
plt.show()


# Isolate our examples for our labeled dataset.
percent = 0.1
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples, size=int(n_labeled_examples * percent))

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]
X_pool = None
y_pool = None

print()
choice = input("Select a choice:\n\n1) Uncertainty Sampling\n2) QBC Sampling\n3) Version Space Sampling\n4) Random Sampling\n5) Cluster based sampling\n\n")
choice = int(choice)


if (choice == 1):

  pf_pool = []
  pf_stream = []

  for i in range(len(qs_pool)):

    print("\n**************** Pool based " + str_qs[i] + " *******************\n")
    pf_pool.append(main(qs_pool[i], str_qs[i] + " Pool", False))
    print("\n**************** Stream based " + str_qs[i] + " *******************\n")
    pf_stream.append(main(qs_stream[i], str_qs[i] + " Stream", False))
    comparison_plot([pf_pool[i], pf_stream[i]], [str_qs[i] + ' Pool', str_qs[i] + ' Stream'])


  comparison_plot(pf_pool, [i + " Pool" for i in str_qs])
  comparison_plot(pf_stream, [i + " Stream" for i in str_qs])


elif (choice == 2):

  pf_pool = []
  pf_stream = []

  for i in range(len(qs_pool_qbc)):

    print("\n**************** Pool based " + str_qs_qbc[i] + " *******************\n")
    pf_pool.append(main(qs_pool_qbc[i], str_qs_qbc[i]+ " Pool", True))
    print("\n**************** Stream based " + str_qs_qbc[i] + " *******************\n")
    pf_stream.append(main(qs_stream_qbc[i], str_qs_qbc[i] + " Stream", True))
    comparison_plot([pf_pool[i], pf_stream[i]], [str_qs_qbc[i] + ' Pool', str_qs_qbc[i] + ' Stream'])


  comparison_plot(pf_pool, [i + " Pool" for i in str_qs_qbc])
  comparison_plot(pf_stream, [i + " Stream" for i in str_qs_qbc])

elif (choice == 4):

  pf_pool = [0, 0]
  pf_stream = [0, 0]
  labels_pool = ["", ""]
  labels_stream = ["", ""]

  for i in range(len(qs_pool)):

    print("\n**************** Pool based " + str_qs[i] + " *******************\n")
    pf = main(qs_pool[i], str_qs[i] + " Pool", False)
    if pf > pf_pool[0]:
      pf_pool[0] = pf
      labels_pool[0] = str_qs[i] + " Pool"

    print("\n**************** Stream based " + str_qs[i] + " *******************\n")
    pf = main(qs_stream[i], str_qs[i] + " Stream", False)
    if pf > pf_stream[0]:
      pf_stream[0] = pf
      labels_stream[0] = str_qs[i] + " Stream"

  for i in range(len(qs_pool_qbc)):

    print("\n**************** Pool based " + str_qs_qbc[i] + " *******************\n")
    pf = main(qs_pool_qbc[i], str_qs_qbc[i] + " Pool", True)
    if pf > pf_pool[0]:
      pf_pool[1] = pf
      labels_pool[1] = str_qs[i] + " Pool"
    print("\n**************** Stream based " + str_qs_qbc[i] + " *******************\n")
    pf = main(qs_stream_qbc[i], str_qs_qbc[i] + " Stream", True)
    if pf > pf_stream[1]:
      pf_stream[1] = pf
      labels_stream[1] = str_qs_qbc[i] + " Stream"

  pf_stream[2] = main(random_sampling, "Random Sampling")
  labels_pool[2] = "Random Sampling"
  pf_pool[2] = pf_stream[2]
  labels_stream[2] = labels_pool[2]

  comparison_plot(pf_pool, labels_pool)
  comparison_plot(pf_stream, labels_stream)



elif (choice == 5):

  X_pool = np.delete(X_raw, training_indices, axis=0)
  y_pool = np.delete(y_raw, training_indices, axis=0)

  learner = ActiveLearner(
          estimator=RandomForestClassifier(),
          X_training=X_train, y_training=y_train
      )

  plot_predictions(learner, 0.1, "Cluster Sampling")

  clust_indices = np.random.randint(low=0, high=len(X_pool), size=int(len(X_raw) * 0.4))
  X_clust = X_pool[clust_indices]
  y_clust = y_pool[clust_indices]

  scaler = MinMaxScaler()
  X_clust = scaler.fit_transform(X_clust)

  kmeans = KMeans(n_clusters = num_classes)
  kmeans.fit(X_clust)
  predictions = kmeans.predict(X_clust)

  clusters = np.array([X_clust[predictions == i] for i in range(3)])
  cluster_targets = np.array([y_clust[predictions == i] for i in range(3)])

  label_count = 0.1 * len(X_raw)

  for i in range(len(clusters)):

    sample_indices = np.random.randint(low=0, high=len(clusters[i]), size=int(len(clusters[i]) * 0.2))
    sample = cluster_targets[i][sample_indices]
    label_count+= len(sample_indices)
    label = np.bincount(sample).argmax()
    cluster_targets[i].fill(label)
    learner.teach(X= clusters[i], y=cluster_targets[i])

  model_accuracy = learner.score(X_raw, y_raw)
  print("Cluster Sampling Accuracy is: ", model_accuracy)
  print("Cost saved = Rs.", (len(X_raw) - label_count) * 100)
  print("Hours saved: ", len(X_raw) - label_count)

  plot_predictions(learner, 0.5, "Cluster Sampling")
