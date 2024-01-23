import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.layers import Bidirectional, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load train data
with open("../data/train.pkl", "rb") as fp:   #Pickling
    chest_wrist_segementation_data = pickle.load(fp)

baseline = chest_wrist_segementation_data['chest+wrist_segementation_baseline']           #(33422, 120, 6)
amusement = chest_wrist_segementation_data['chest+wrist_segementation_amusement']         #(9358, 120, 6)
stress = chest_wrist_segementation_data['chest+wrist_segementation_stress']               #(18133, 120, 6)

#signal index: ACC: 0 1 2; ECG: 3; EMG: 4; EDA: 5; Resp: 6
#signal index: ACC: 7 8 9; BVP: 10; EDA: 11
x_baseline = baseline
x_amusement = amusement
x_stress = stress

all_data = np.vstack((x_baseline, x_amusement, x_stress))       

# Generate labels
baseline_labels = np.zeros(len(x_baseline)).reshape(-1,1)            # baseline->0
amusement_labels = np.ones(len(x_amusement)).reshape(-1,1)          # amusement->1
stress_labels = (np.ones(len(x_stress))*2).reshape(-1,1)               # stress->2

all_labels = np.squeeze(np.vstack((baseline_labels, amusement_labels, stress_labels)))   


# Load test data
with open("../data/test.pkl", "rb") as fp:   #Pickling
    chest_wrist_segementation_data_test = pickle.load(fp)

baseline_test = chest_wrist_segementation_data_test['chest+wrist_segementation_baseline_test']           #(33422, 120, 6)
amusement_test = chest_wrist_segementation_data_test['chest+wrist_segementation_amusement_test']         #(9358, 120, 6)
stress_test = chest_wrist_segementation_data_test['chest+wrist_segementation_stress_test']               #(18133, 120, 6)

y_baseline = baseline_test
y_amusement = amusement_test
y_stress = stress_test

test_data = np.vstack((y_baseline, y_amusement, y_stress))

# Generate labels
baseline_labels_test = np.zeros(len(y_baseline)).reshape(-1,1)            # baseline->0
amusement_labels_test = np.ones(len(y_amusement)).reshape(-1,1)          # amusement->1
stress_labels_test = (np.ones(len(y_stress))*2).reshape(-1,1)               # stress->2

test_labels = np.squeeze(np.vstack((baseline_labels_test, amusement_labels_test, stress_labels_test))) 

#shauffling
shuffle_ix_te = np.random.permutation(np.arange(len(test_data)))
test_data = test_data[shuffle_ix_te,:]
test_labels = test_labels[shuffle_ix_te]


# Split into training data and validation data
train_data, validation_data, train_labels, validation_labels = train_test_split(
    all_data, all_labels, test_size=0.2, random_state=21)


# Normalization       
scaler = MinMaxScaler()
tem_train = scaler.fit_transform(train_data.reshape(-1, train_data.shape[2]))                       
train_data = tem_train.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])       

tem_validation = scaler.transform(validation_data.reshape(-1, validation_data.shape[2]))                       
validation_data = tem_validation.reshape(validation_data.shape[0], validation_data.shape[1], validation_data.shape[2])  

tem_test = scaler.transform(test_data.reshape(-1, test_data.shape[2]))                              
test_data = tem_test.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2])            
    

train_data = tf.cast(train_data, tf.float32)
validation_data = tf.cast(validation_data, tf.float32)            
test_data = tf.cast(test_data, tf.float32)              

# One Hot Encoding
train_labels = keras.utils.to_categorical(train_labels)
validation_labels = keras.utils.to_categorical(validation_labels)
test_labels = keras.utils.to_categorical(test_labels)    
# data.shape[0] = samples, data.shape[1] = seq_length, data.shape[2] = n_features


seq_length = train_data.shape[1]     # data.shape[1]
n_features = train_data.shape[2]      ## data.shape[2] 


# Model
RNN_CELL_SIZE = 32

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # Perform addition to calculate fractions
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, seq_length, 1)
        # get 1 at the last axis because the score is applied to self.V
        # the shape of the tensor before applying self.V is (batch_size, seq_length, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape == (batch_size, seq_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    

x_input = Input(shape=(seq_length, n_features))

lstm = Bidirectional(LSTM(RNN_CELL_SIZE, return_sequences = True), name="bi_lstm_0")(x_input)
lstm = Bidirectional(LSTM(RNN_CELL_SIZE), name="bi_lstm_1")(lstm)

r1 = Reshape(target_shape=(64,1))(lstm)
c1 = Conv1D(filters=64, kernel_size=3, padding="same", strides=2, activation="relu")(r1)
d1 = Dropout(rate=0.2)(c1)
b1 = BatchNormalization()(d1)
c2 = Conv1D(filters=32, kernel_size=3, padding="same", strides=2, activation="relu")(b1)
d2 = Dropout(rate=0.2)(c2)
b2 = BatchNormalization()(d2)
m1 = MaxPooling1D(pool_size=3, strides=2)(b2)   
f1 = Flatten()(m1)
d3 = Dropout(rate=0.2)(f1)                          
output = Dense(3, activation="softmax")(d3)


model = keras.Model(inputs=x_input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()


checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_data, train_labels,
          epochs=10,
          batch_size=128,
          validation_data=(validation_data, validation_labels),
          callbacks=[model_checkpoint_callback],
          shuffle=True, verbose=2)


# load the weights from the checkpoint
model.load_weights(checkpoint_filepath)


# plot training history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.legend()


# Evaluation
def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))
  print("F1 = {}".format(f1_score(labels, predictions)))

start = time.time()  
pred = model.predict(test_data)

print ('Time for prediction is {} sec'.format(time.time()-start))
print(classification_report(np.argmax(test_labels, 1),
                            np.argmax(pred, 1),
      target_names = [f'Class-{i}' for i in range(0, 3)],digits=4))
