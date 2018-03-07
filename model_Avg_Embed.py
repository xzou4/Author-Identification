import numpy as np
import tensorflow as tf
import csv
import io
from sklearn import preprocessing
from string import punctuation
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pdb
from collections import Counter

tf.reset_default_graph()
random_seed = 320
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# In[3]:

reviews = []
labels = []
test = []
test_ids = []
external_authors = []
external_lines = []
with io.open('text.csv', 'r',encoding="latin-1") as f:
	text_reader = csv.reader(f,delimiter=",")
	next(text_reader)
	for row in text_reader:
		reviews.append(row[0])

with open('author.csv', 'r') as labels_csv:
	author_reader = csv.reader(labels_csv,delimiter =",")
	next(author_reader) #ignore header
	for row in author_reader:
		labels.append(row[0])
with open("test.csv",'r') as test_csv:
	text_reader = csv.reader(test_csv, delimiter= ",")
	next(text_reader) #ignore header
	
	for row in text_reader:
		test_ids.append(row[0])
		test.append(row[1])
		
with io.open("newData.csv",'r',encoding="latin-1") as external_data:
	text_reader = csv.reader(external_data,delimiter=",")
	
	for row in text_reader:
		line = row[1].strip()
		
		if len(line.split() )> 5: #if the line has at least 5 words, keep it
			external_authors.append(row[0].strip())
			external_lines.append(line)



lstm_size = 128
fully_connected_size = 128
lstm_layers = 1
fully_connected_layers = 1
batch_size = 512
display_every_iterations = 20 
dropout_lstm = 1
dropout_fully_connected = 0.8
learning_rate = 0.001


def get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout):
	configuration_string = "lstm_layers={}&lstm_size={}&fully_connected_layers={}&fully_connected_size={}&batch_size={}&learning_rate={}&lstm_dropout={}&fully_connected_dropout={}"     .format(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout)
	
	return configuration_string
	
get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,dropout_lstm,dropout_fully_connected)


graph = tf.Graph()

def loadGloveModel(gloveFile):
	print("Loading Glove Model")
	#f = open(gloveFile,'r')
	f = io.open(gloveFile, 'r',encoding="utf-8")
	word_list = []
	embeddings = []
	for line in f:
		splitLine = line.split()
		try:
			embedding = np.array([float(val) for val in splitLine[1:]])
			#model[word] = embedding
			word_list.append(splitLine[0])
			embeddings.append(embedding)
		except ValueError:
			print(line)
			continue
	print("Done.",len(word_list)," words loaded!")
	assert(len(word_list) == len(embeddings))
	return word_list, np.array(embeddings)

word_list, embeddings = loadGloveModel("f:\glove.6B.300d.txt")

import re
from nltk.tokenize import word_tokenize
my_words = [word for line in reviews + test +external_lines for word in word_tokenize(line.lower())]

vocab_to_int = {word:index for index,word in enumerate(set(my_words), 1)}
vocab_to_int['****pad*****'] = 0

reviews_ints = []
reviews_ints_len = []
for review in reviews:
	reviews_ints.append([vocab_to_int[word] for word in word_tokenize(review.lower() + ' ')])
	
	
test_ints = []
test_ints_len = []
for test_line in test:
	test_ints.append([vocab_to_int[word] for word in word_tokenize(test_line.lower())])
	
external_ints = []
external_ints_len = []
for external_line in external_lines:
	external_ints.append([vocab_to_int[word] for word in word_tokenize(external_line.lower())])


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
print("Minimum length: {}".format(min(review_lens)))
print("Average length: {}".format(sum(review_lens)/len(review_lens)))


seq_len = 150
features = []

for review in reviews_ints:
	review_size = len(review)
	if review_size < seq_len:
		padded_review = [0] * seq_len
		padded_review[seq_len-len(review):seq_len] = review
	elif review_size > seq_len:
		padded_review = review[:seq_len]
	reviews_ints_len.append(min(review_size, seq_len))
	
	features.append(padded_review)
features  = np.array(features)
reviews_ints_len = np.array(reviews_ints_len)


test_features = []

for test_line in test_ints:
	line_size = len(test_line)
	if line_size < seq_len:
		padded_line = [0] * seq_len
		padded_line[seq_len-len(test_line):seq_len] = test_line
	elif line_size > seq_len:
		padded_line = test_line[:seq_len]
	test_ints_len.append(min(line_size, seq_len))
		
	test_features.append(padded_line)

test_features = np.array(test_features)
test_ints_len = np.array(test_ints_len)
test_ints_len = np.reshape(test_ints_len, [len(test_ints_len), 1])


external_features = []

for external_line in external_ints:
	line_size = len(external_line)
	if line_size < seq_len:
		padded_line = [0] * seq_len
		padded_line[seq_len-len(external_line):seq_len] = external_line
	elif line_size > seq_len:
		padded_line = external_line[:seq_len]
	external_ints_len.append(min(line_size, seq_len))
		
	external_features.append(padded_line)

external_features = np.array(external_features)
external_ints_len = np.array(external_ints_len)

embed_size = 300
n_words = len(vocab_to_int)
word_dict = np.zeros((n_words, embed_size))

for word, index in vocab_to_int.items():
	if word in word_list:
		word_dict[index, :] = embeddings[word_list.index(word), :]
	else:
		word_dict[index, :] = np.random.uniform(low=-0.10, high=0.10, size=(embed_size,))

word_dict_ = word_dict
word_dict_norm = np.zeros(word_dict.shape)
d = (np.sum(word_dict ** 2, 1) ** (0.5))
word_dict_norm = (word_dict.T / d).T
word_dict_norm[np.isnan(word_dict_norm)] = 0
word_dict = word_dict_norm

#embed_size = 200
#word_dict = np.random.uniform(-0.1, 0.1, (word_dict.shape[0], embed_size))

print(word_dict.shape)

labels_to_int = {}
int_to_labels = {}
unique_labels = list(set(labels))
for i,label in enumerate(unique_labels):
	labels_to_int[label] = i
	int_to_labels[i] = label
	
int_labels = []

for label in labels:
	int_labels.append(labels_to_int[label])
	
int_external_labels = []

for label in external_authors:
	int_external_labels.append(labels_to_int[label])

encoder = preprocessing.LabelBinarizer()
encoder.fit(list(set(int_labels)))
one_hot_labels = encoder.transform(int_labels)

one_hot_external_labels = encoder.transform(int_external_labels)

split_frac = 0.80

split_index  = int(len(features)*split_frac)

indices = np.array(list(range(len(features))))
np.random.shuffle(indices)

train_x, val_x = features[indices[:split_index],:],features[indices[split_index:],:]
train_y, val_y = one_hot_labels[indices[:split_index],:],one_hot_labels[indices[split_index:],:]
train_x_len, val_x_len = reviews_ints_len[indices[:split_index]], reviews_ints_len[indices[split_index:]]

train_x = np.append(train_x, external_features,axis = 0)
train_y = np.append(train_y, one_hot_external_labels,axis= 0)
train_x_len = np.append(train_x_len, external_ints_len, axis = 0)
random_indexes = np.random.permutation(train_x.shape[0])

train_x = train_x[random_indexes]
train_y = train_y[random_indexes]
train_x_len = train_x_len[random_indexes]

with graph.as_default():
	inputs_ = tf.placeholder(tf.int32,shape=[batch_size,seq_len],name="inputs")
	labels_ = tf.placeholder(tf.int32,shape=[batch_size,3],name = "labels")
	inputs_len = tf.placeholder(tf.float32, shape = [batch_size], name = "len")
	keep_prob = tf.placeholder(tf.float32,name ="keep_prob")
	fully_connected_keep_prob = tf.placeholder(tf.float32,name = "fc_keep_prob")
	global_step = tf.Variable(0, trainable=False)

	learning_rate_ = tf.train.exponential_decay(learning_rate, global_step, 100, decay_rate = 0.99997592083)

with graph.as_default():
	embedding = tf.Variable(word_dict, dtype = tf.float32)
	embed = tf.nn.embedding_lookup(embedding,inputs_)
	outputs = tf.reduce_sum(embed, 1)/tf.reshape(inputs_len, [batch_size, -1])
	outputs_ = tf.reduce_max(embed, 1)
	outputs = tf.concat([outputs, outputs_], axis = 1)

	#outputs = tf.layers.batch_normalization(outputs)
'''
with graph.as_default():
	cell_list = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=lstm_size, initializer=tf.orthogonal_initializer) ,output_keep_prob=keep_prob)  ]
	cell = tf.contrib.rnn.MultiRNNCell(cell_list)
	# Getting an initial state of all zeros
	initial_state = cell.zero_state(batch_size, tf.float32)

	# cell_list_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units=lstm_size, initializer=tf.orthogonal_initializer) ,output_keep_prob=keep_prob)  ]
	# cell_bw = tf.contrib.rnn.MultiRNNCell(cell_list_bw)
	# Getting an initial state of all zeros
	# initial_state_bw = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
	outputs, final_state = tf.nn.dynamic_rnn(cell,embed,initial_state=initial_state)
	# outputs, state_fw, state_bw = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, embed, initial_state_fw = initial_state, initial_state_bw = initial_state_bw)
	# final_state = tf.concat(2, [state_fw, state_bw])
'''

with graph.as_default():
	fully_connected = tf.contrib.layers.fully_connected(outputs, fully_connected_size, activation_fn=tf.nn.relu)
	
	fully_connected = tf.nn.dropout(fully_connected,fully_connected_keep_prob)
	
	
	for layer in range(fully_connected_layers - 1):
		fully_connected = tf.contrib.layers.fully_connected(fully_connected, fully_connected_size, activation_fn=tf.nn.relu)
		fully_connected = tf.nn.dropout(fully_connected,fully_connected_keep_prob)
		
	fully_connected = tf.contrib.layers.fully_connected(fully_connected, 3, activation_fn=tf.nn.relu)
	logits = tf.identity(fully_connected)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=labels_))
	optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)


with graph.as_default():
	predictions = tf.nn.softmax(logits)
	predictions_hardmax = tf.argmax(predictions,1)

def get_batches(x, y, z, batch_size=100):
	#shuffle batches at every ecoch    
	n_batches = len(x)//batch_size
	x, y, z = x[:n_batches*batch_size], y[:n_batches*batch_size], z[:n_batches*batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size], z[ii:ii+batch_size]

def calc_classification_metrics(predictions,real_values):
	accuracy =  sum(predictions == real_values)/predictions.shape[0] # metrics.accuracy_score(predictions,real_values)
	error = 1 - accuracy
	precision = 0# metrics.precision_score(predictions,real_values)
	recall = 0#metrics.recall_score(predictions,real_values)
	
	return accuracy,error,precision,recall


def error_analysis(configuration_string,predictions,real_values):
	headers = []
	
	for correct_label in int_to_labels.values():
		for predicted_label in int_to_labels.values():
			if correct_label != predicted_label:
				analysis_column = "correct_label:"+correct_label+"_predicted_label:"+predicted_label
				headers.append(analysis_column)
					
	with open("error_analysis/"+configuration_string+".csv","w") as error_file:
				header_writer = csv.writer(error_file)
				writer = csv.DictWriter(error_file, fieldnames= headers, quoting=csv.QUOTE_ALL)
				header_writer.writerow(headers)
	
				for i in range(predictions.shape[0]):
					analysis_colums = {}

					for header in headers:
						analysis_colums[header] = 0

					if predictions[i] != real_values[i]:
						correct_label = int_to_labels[real_values[i]]
						predicted_label = int_to_labels[predictions[i]]
						analysis_column = "correct_label:"+correct_label+"_predicted_label:"+predicted_label
						analysis_colums[analysis_column]  = 1
						writer.writerow(analysis_colums)
			
			
def train(graph,saver,epochs,lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout):
	
	log_file = open("log_book.csv","a")
	csv_writer = csv.writer(log_file,delimiter = ",")
	
	configuration_string = get_configuration_string(lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,lstm_dropout,fully_connected_dropout)
	print("Starting training for: ",configuration_string)
	
	best_cost = float("inf")
	with tf.Session(graph=graph) as sess:
		sess.run(tf.global_variables_initializer())
		iteration = 1
		for e in range(epochs):
			for ii, (x, y, z) in enumerate(get_batches(train_x, train_y, train_x_len, batch_size), 1):
				feed_dict = {inputs_: x,
					labels_: y,
					inputs_len: z,
					keep_prob: lstm_dropout,
					fully_connected_keep_prob:fully_connected_dropout,
						learning_rate_ : learning_rate}
				loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
			
				if iteration%display_every_iterations ==0:
					val_acc = []
					val_costs = []
				
					train_prediction_hardmax = sess.run(predictions_hardmax,feed_dict=feed_dict)
					train_real_hardmax = np.argmax(y,1)
					train_accuracy,train_error,train_precision,train_recall  = calc_classification_metrics(train_prediction_hardmax,train_real_hardmax)
				
					epoch_predictions = np.array([])
					epoch_real_values = np.array([])
					for x, y, z in get_batches(val_x, val_y, val_x_len, batch_size):
						feed_dict = {inputs_: x,
							labels_: y,
							inputs_len: z,
							keep_prob: 1,
							fully_connected_keep_prob:1,
							learning_rate_ : learning_rate}
					
						val_prediction = sess.run(predictions,feed_dict=feed_dict)
						val_cost = sess.run(cost,feed_dict=feed_dict)
						val_prediction_hardmax = sess.run(predictions_hardmax,feed_dict=feed_dict)
						val_real_hardmax = np.argmax(y,1)
						val_accuracy,val_error,val_precision,val_recall  = calc_classification_metrics(val_prediction_hardmax,val_real_hardmax)
						val_acc.append(val_accuracy)
						val_costs.append(val_cost)
						epoch_predictions= np.append(epoch_predictions,val_prediction_hardmax,axis = 0)
						epoch_real_values = np.append(epoch_real_values,val_real_hardmax ,axis = 0)
						
					val_cost = np.mean(val_costs)  
					val_acc  = np.mean(val_acc)
					print("Epoch: {}/{}".format(e+1, epochs),
					  "Iteration: {}".format(iteration),
					  "Train loss: {:.3f}".format(loss),
					  "Train accuracy: {:.3f}".format(train_accuracy),
					 "Train error: {:.3f}".format(train_error),
					  "Val cost: {:.3f}".format(val_cost),
					  "Val acc: {:.3f}".format(val_acc)
					 )
					if best_cost > val_cost:
						best_cost = val_cost
						saver.save(sess, "./checkpoints/checkpoint.ckpt".format(configuration_string))
					
				

					
				iteration +=1
			# learning_rate = 0.95 * learning_rate
		
		#saver.save(sess, "/checkpoints/checkpoint.ckpt")
		
		csv_writer.writerow([configuration_string,loss,train_accuracy,train_error,val_cost,val_acc,(1-val_acc)])
		log_file.close()
		
		epoch_predictions = np.array(epoch_predictions)
		epoch_real_values = np.array(epoch_real_values)
		
		error_analysis(configuration_string,epoch_predictions,epoch_real_values)
		
		print("Finished training for: ",configuration_string)


with graph.as_default():
	saver = tf.train.Saver()
	
train(graph,saver,8,lstm_layers,lstm_size,fully_connected_layers,fully_connected_size,batch_size,learning_rate,dropout_lstm,dropout_fully_connected)

with graph.as_default():
	saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
	#saver.restore(sess, tf.train.latest_checkpoint('checkpoints/lstm_layers=1&lstm_size=512&fully_connected_layers=2&fully_connected_size=20&batch_size=512&learning_rate=0.005&lstm_dropout=1&fully_connected_dropout=1'))
	saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
	
	full_batches =  int(test_features.shape[0]/batch_size)
	submit_file = open("submission.csv","w")
	csv_writer = csv.writer(submit_file,delimiter = ",")
	header = ["id","EAP","HPL","MWS"]
	csv_writer.writerow(header)
	
	# TODO: find a way to send 1 line at a time without having to take special care of last incomplete batch
	for i in range(full_batches):
		batch = test_features[ (i*batch_size):((i+1)*batch_size)]
		batch_len = test_ints_len[ (i*batch_size):((i+1)*batch_size)]
		batch_len = np.reshape(batch_len, [len(batch_len)])

		feed_dict = {inputs_: batch,
				keep_prob: 1,
				inputs_len: batch_len,
				fully_connected_keep_prob:1,
						learning_rate_ : learning_rate}
	
		test_prediction = sess.run(predictions,feed_dict=feed_dict)
		test_prediction_hardmax = np.argmax(test_prediction,1)
	
		for j,prediction in enumerate(test_prediction_hardmax):
			EAP_prob = test_prediction[j,labels_to_int["EAP"]]
			HPL_prob = test_prediction[j,labels_to_int["HPL"]]
			MWS_prob = test_prediction[j,labels_to_int["MWS"]]
		
			line = [test_ids[(i*batch_size + j)] , EAP_prob, HPL_prob, MWS_prob]
			csv_writer.writerow(line)
		
		# print("finished batch {}".format(i))
			#print("Finished writing {}".format(i))
			#print(test_ids[i] ,test[i][:50] , " ", int_to_labels[test_prediction_hardmax] , EAP_prob, HPL_prob, MWS_prob)
	   
	if test_features.shape[0]%batch_size != 0:
		print("Last minibatch" ,full_batches*batch_size,",", test_features.shape[0])
		
		i+=1
		batch = np.zeros((batch_size,test_features.shape[1]))
		batch_len = np.zeros((batch_size, 1))
		batch[0:(test_features.shape[0]-full_batches*batch_size),:] = test_features[full_batches*batch_size:,:]
		batch_len[0:(test_features.shape[0]-full_batches*batch_size)] = test_ints_len[full_batches*batch_size:]
		batch_len = np.reshape(batch_len, [len(batch_len)])
		
		feed_dict = {inputs_: batch,
				keep_prob: 1,
				inputs_len: batch_len,
				fully_connected_keep_prob:1,
						learning_rate_ : learning_rate}
	
		test_prediction = sess.run(predictions,feed_dict=feed_dict)
		test_prediction_hardmax = np.argmax(test_prediction,1)

		for j in range(test_features.shape[0]-full_batches*batch_size):
			EAP_prob = test_prediction[j,labels_to_int["EAP"]]
			HPL_prob = test_prediction[j,labels_to_int["HPL"]]
			MWS_prob = test_prediction[j,labels_to_int["MWS"]]
		
			line = [test_ids[(i*batch_size + j)] , EAP_prob, HPL_prob, MWS_prob]
			csv_writer.writerow(line)
		
		print("finished batch {}".format(i))

	submit_file.close()
