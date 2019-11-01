from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

from typing import List
import string

def one(x):

	def two():
		return (x)
	return two

f=one(1)
print(f)




#vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)


'''
vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)

data = array([
	[0,0,1,0,0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])
docs=["well bad"]
for x in docs[0]:
    print (x)
data = data.reshape((2, 10, 2))

print(data)


'''
'''
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

docs2 =[ "bad job",
         "amazing",
         "poor performance"
]

x=len(docs[len(docs)-1])
print(x)
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = len(vocab)
encoded_docs = []
for d in docs :
    sentence_array = []
    for c in d :
        sentence_array.append(one_hot(c, 256))
		print
    encoded_docs.append(sentence_array)
print(encoded_docs)
'''
'''
# pad documents to a max length of 4 words
max_length = x
padded_docs=[]
padded_docs.append(pad_sequences(encoded_docs, maxlen=max_length, padding='post'))

print(padded_docs)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten)
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print(loss)
print('Accuracy: %f' % (accuracy*100))

encoded_docs2 = [one_hot(d, vocab_size) for d in docs2]
print(encoded_docs2)
# pad documents to a max length of 4 words

padded_docs2 = pad_sequences(encoded_docs2, maxlen=max_length, padding='post')

modelpredictions = model.predict(padded_docs2)
print(modelpredictions)
'''
'''
vocab = list(string.digits + string.ascii_lowercase + string.punctuation + string.whitespace)
parts=(("word",[1,2,3]),("smth",[4,5,6]))
seps=["//"]
for s in zip(parts, seps) :
    print (s)
#strings = ''.join(sum([(s[0][0], s[1]) for s in zip(parts, seps)], ()))
#print(strings)

'''
'''
def vocab_lookup(characters: str) -> (int, np.ndarray):
    """
    Converts a string into a list of vocab indices
    :param characters: the string to convert
    :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return len(characters), np.array(result, dtype=np.int64)


x=vocab_lookup("11")
print(x[0])
'''


'''
data={
    "name" :"omar" ,
    "age" : 20 ,
    "nationality" : "egyptian"
}


name=tf.train.BytesList(value=[data["name"].encode()])
age=tf.train.Int64List(value=[data["age"]])
nation=tf.train.BytesList(value=[data["nationality"].encode()])

names=tf.train.Feature(bytes_list=name)
ages=tf.train.Feature(int64_list=age)
nations=tf.train.Feature(bytes_list=nation)

dic={
 "names" : names ,
 "ages"  : ages ,
 "nations" : nations
 }

everything=tf.train.Features(feature=dic)

example=tf.train.Example(features=everything)
print(example)

tf_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

with tf.python_io.TFRecordWriter('omarahmadtester',options=tf_options) as writer:
  writer.write(example.SerializeToString())
'''

