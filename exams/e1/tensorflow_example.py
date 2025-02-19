import tensorflow as tf
import numpy as np
import os

'''Load dataset: tiny_shakespeare'''
filePath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filePath, 'rb').read().decode(encoding='utf-8')

#Print first 250 characters as check
print(text[:250])

'''Create character-level vocabulary of chars to integers'''
vocab = sorted(set(text))

#map unique characters -> indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

#Text -> integer sequence
textToInt = np.array([char2idx[c] for c in text])

'''Data preparation'''

#Define maximum input sentence length:
seqLength = 100
examples_per_epoch = len(text) // (seqLength + 1)

#Create training targets
charDataset = tf.data.Dataset.from_tensor_slices(textToInt)

sequences = charDataset.batch(seqLength + 1, drop_remainder=True)

#Define targets

def splitInputTarget(chunk):
    inputText = chunk[:-1]
    targetText = chunk[1:]
    return inputText, targetText

#Define training dataset

dataset = sequences.map(splitInputTarget)

#Define batch size
BATCH_SIZE = 64

#Define buffer size for shuffling dataset
BUFFER_SIZE = 10000

#Shuffle dataset
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

'''Define and build a Recurrent Neural Network model'''

#Define vocabulary length in chars
vocab_size = len(vocab)

#Define embedding dimension (the mapping of words to vectors and the length of those vectors)
embedding_dim = 256

#Define number of RNN units (the number of vectors per word)
rnn_units = 1024

#Define model structure
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)

    ])
    return model

#Build model
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

'''Model compilation and training'''

#Define a loss function for the model (loss functions act as a performance measure)
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

#Checkpoint save directory
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

'''Text generation'''

#Restore latest checkpoint
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#Define text generation function
def generate_text(model, start_string, num_generate=100):
    #Convert start string to numbers (vectorize)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    #Empty string for result storage
    text_generated = []

    #Temperature setting: lower = more predictable, higher = more suprising
    temperature = 1.0

    #Set batch size = 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        #Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        #Use a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        #Pass the predicted character and previous hidden state as next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return(start_string + ''.join(text_generated))

#Generate text
generated_text = generate_text(model, start_string=u"JULIET: ", num_generate=100)
print(generated_text)

'''Post-processing to check for Iambic Pentameter integrity'''
import pronouncing
#natural language library

def is_iambic_pentameter(line):
    words = line.split()
    syllables = []
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            syllables.append(pronouncing.syllable_count(phones[0]))
        else:
            return False
    total_syllables = sum(syllables)
    return total_syllables == 10
#Iambic pentameter consits of 10 syllables and 5 stress points
#But i'm not sure how to evaluate for syllable stress yet.

#Example usage
if is_iambic_pentameter(generated_text):
    print("The generated text contains exactly 10 syllables.")
else:
    print("The generated text does not contain exactly 10 syllables.")
