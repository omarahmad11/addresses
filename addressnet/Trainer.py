import addressnet.model as model
import tensorflow as tf
from addressnet.dataset import dataset

'''
x=dataset("sample_record_output")
print(x)
'''

filenames=0
classifier = tf.estimator.Estimator(model_fn= model.model_fn)  # Path to where checkpoints etc are stored

classifier.train(input_fn= dataset(["sample_record_output"]))
