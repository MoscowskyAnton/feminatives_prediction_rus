import numpy as np
from numpy import array
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from pickle import dump

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 4
batch_size = 32

print('Loading data...')
in_filename = '../data_embeddings/WaP_I_char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    # integer encode line
    if len(line) < maxlen+1 :
        continue
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)
 
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
 
# separate into input and output
sequences = array(sequences)
#print(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
#sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
#X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)
'''
print(X)
print(y)
exit()
'''
#X = X[:100,:]
#y = y[:100,:]

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(vocab_size, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X, y,
          batch_size=batch_size,
          epochs=5)

# get embeddings
embeddings = model.layers[0].get_weights()[0]

chars_embeddings = {c:embeddings[idx] for c, idx in mapping.items()}
print(chars_embeddings)

#with open('../data_embeddings/embeddings.yaml', 'w') as outfile:
    #yaml.dump(chars_embeddings, outfile)#, default_flow_style=False)
    
dump(chars_embeddings, open('../data_embeddings/char_to_vec_map.pkl', 'wb'))
dump(mapping, open('../data_embeddings/char_to_index.pkl', 'wb'))
