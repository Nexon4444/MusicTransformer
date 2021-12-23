from model import MusicTransformer
from custom.layers import *
from custom import callback
from tensorflow.python import keras
# import params as par
import midi_processor.processor as sequence
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
# tf.executing_eagerly()
#%%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
