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
                          query_strategy=vote_entropy_sampling)

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
  plt.show()


def sampling(learner, num_queries, performance):

  global X_pool, y_pool, X_raw, y_raw

  # Allow our model to query our unlabeled dataset for the most
  # informative points according to our query strategy (uncertainty sampling).
  for index in range(num_queries):

    query_index, query_instance = learner.query(X_pool)

    # Teach our ActiveLearner model the record it has requested.
    X, y = X_pool[query_index].reshape(1, -1), y_pool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

    # Remove the queried instance from the unlabeled pool.
    X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

    # Calculate and report our model's accuracy.
    model_accuracy = learner.score(X_raw, y_raw)
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

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

  plot_predictions(learner, 0.4, str_query_strategy)
  return performance




# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)

dataset = load_iris()
X_raw = dataset['data']
y_raw = dataset['target']

# -------------for visualization---------------
# # Define our PCA transformer and fit it onto our raw dataset.
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_dataset = pca.fit_transform(X=X_raw)

# Isolate the data we'll need for plotting.
x_component, y_component = transformed_dataset[:, 0], transformed_dataset[:, 1]

# Plot our dimensionality-reduced (via PCA) dataset.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
plt.title('Dataset classes after PCA transformation')
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

  performance1 = main(uncertainty_sampling, "Uncertainty Sampling", False)
  performance2 = main(margin_sampling, "Margin Sampling", False)
  performance3 = main(entropy_sampling, "Entropy Sampling", False)

  fig = plt.figure()
  ax = plt.subplot(111)
  plot_accuracies(performance1, fig, ax, "Uncertainty Sampling")
  plot_accuracies(performance2, fig, ax, "Margin Sampling")
  plot_accuracies(performance3, fig, ax, "Entropy Sampling")
  ax.legend()
  plt.show()
  fig.savefig("Accuracies_Uncertainty.png")

elif (choice == 2):

  performance1 = main(uncertainty_sampling, "Vote Entropy Sampling", True)
  performance2 = main(margin_sampling, "KL Divergence Sampling", True)

  fig = plt.figure()
  ax = plt.subplot(111)
  plot_accuracies(performance1, fig, ax, "Vote Entropy Sampling")
  plot_accuracies(performance2, fig, ax, "KL Divergence Sampling")
  ax.legend()
  plt.show()
  fig.savefig("Accuracies_QBC.png")