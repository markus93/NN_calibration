#Code base from https://github.com/aravindsiv/dan_qa

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import pickle as pkl
import numpy as np

class PreProcessor:
	def __init__(self,INPUT_FILE,WE_FILE):
		self.input_file = INPUT_FILE
		self.we_file = WE_FILE

	def tokenize(self):
		data = pkl.load(open(self.input_file,"rb"))

		self.train_data = np.array(data["train"])
		self.val_data = np.array(data["dev"])

		questions = np.array(["".join(self.train_data[i,0]) for i in range(self.train_data.shape[0])])
		questions_val = np.array(["".join(self.val_data[i,0]) for i in range(self.val_data.shape[0])])

		print(questions[0])
		#questions = np.array([self.train_data[i,0] for i in range(self.train_data.shape[0])])
		#questions_val = np.array([self.train_data[i,0] for i in range(self.val_data.shape[0])])

		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(questions)

		self.sequences = tokenizer.texts_to_sequences(questions)
		self.sequences_val = tokenizer.texts_to_sequences(questions_val)

		self.word_index = tokenizer.word_index
		print("Found %s unique tokens" %(len(self.word_index)))

	def make_data(self):
		self.MAX_SEQUENCE_LENGTH = max([len(self.sequences[i]) for i in range(len(self.sequences))])

		data = pad_sequences(self.sequences,maxlen=self.MAX_SEQUENCE_LENGTH)
		data_val = pad_sequences(self.sequences_val,maxlen=self.MAX_SEQUENCE_LENGTH)

		answers_train = set(self.train_data[:,1])
		answers_val = set(self.val_data[:,1])
		answers = answers_train.union(answers_val)

		labels_index = {} # labels_index["Henry IV of France"]
		answers_index = {} # answers_index[0]

		for i,j in enumerate(answers):
		    labels_index[j] = i
		    answers_index[i] = j
		    
		labels = np.zeros((len(self.sequences),1))
		labels_val = np.zeros((len(self.sequences_val),1))

		for i in range(len(self.sequences)):
		    labels[i] = labels_index[self.train_data[i,1]]
		    
		for i in range(len(self.sequences_val)):
		    labels_val[i] = labels_index[self.val_data[i,1]]
		    
		labels = to_categorical(labels,num_classes=len(answers))
		labels_val = to_categorical(labels_val,num_classes=len(answers))

		print("Shape of data tensor: " +str(data.shape))
		print("Shape of label tensor: " +str(labels.shape))

		return data, labels, data_val, labels_val

	def get_word_embedding_matrix(self,EMBEDDING_DIM=100):
		embeddings_index = {}

		if self.we_file == "rand":
			return None

		f = open(self.we_file)

		for line in f:
		    values = line.split()
		    word = values[0]
		    coefs = np.asarray(values[1:], dtype='float32')
		    embeddings_index[word] = coefs
		f.close()

		print('Found %s word vectors.' % len(embeddings_index))

		self.embedding_matrix = np.zeros((len(self.word_index)+1, EMBEDDING_DIM))

		for word, i in self.word_index.items():
		    embedding_vector = embeddings_index.get(word)
		    if embedding_vector is not None:
		        # words not found in embedding index will be all-zeros.
		        self.embedding_matrix[i] = embedding_vector

		return self.embedding_matrix

if __name__ == "__main__":
	pass
