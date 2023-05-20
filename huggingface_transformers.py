import tensorflow as tf
from tensorflow.keras import backend as K
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report
import numpy as np
import time

# Sets the default float/posit type. ['float16' , 'float32', 'float64', 'posit160']
# Load the model converting weights to posti<16,0> dataype
K.set_floatx('posit160')

#
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
# https://huggingface.co/datasets/imdb
#

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
#print(model.get_weights())

# Load the test subset o)f IMDB dataset
data = load_dataset('imdb', split="test")
test_data = data[:16]

# Set batch size for tokenization and prediction
batch_size = 16

num_samples = len(test_data['text'])
num_batches = (num_samples + batch_size - 1) // batch_size

seconds = time.time()

# Tokenize the texts and make predictions in batches
predictions = []
for i in range(num_batches):
    batch_texts = test_data['text'][i*batch_size:(i+1)*batch_size]
    inputs = tokenizer(list(batch_texts), padding=True, truncation=True, return_tensors="tf")
    
    # Cast the input in posti160
    # inputs['input_ids'] = tf.cast(inputs['input_ids'], dtype=tf.posit160)
    # inputs['attention_mask'] = tf.cast(inputs['attention_mask'], dtype=tf.posit160)
    
    outputs = model(inputs)
    batch_predictions = np.argmax(outputs.logits, axis=-1)
    
    predictions.append(batch_predictions)

predictions = np.concatenate(predictions)

# Calculate metrics
print("Execution time =", time.time() - seconds)
print(classification_report(predictions, test_data['label']))