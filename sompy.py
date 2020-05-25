import SimpSOM as sps
import sklearn.datasets as s

raw_data = s.load_iris().data
print(raw_data.shape)


#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions. 
net = sps.somNet(4, 4, raw_data, PBC=True)

# #Train the network for 10000 epochs and with initial learning rate of 0.01. 
net.train(0.01, 1)

# #Save the weights to file
# net.save('filename_weights')

# #Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
# #and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()

# #Project the datapoints on the new 2D network map.
net.project(raw_data, labels=[0, 1, 2])

# #Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(raw_data, type='qthresh')