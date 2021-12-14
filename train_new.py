import os
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint

from model_new import MusicTransformer
from custom.layers import *
from custom import callback
import params as par
from tensorflow.keras.optimizers import Adam
from data import DataNew
import utils
import argparse
import datetime

import sys

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, help='학습률', type=float)
parser.add_argument('--batch_size', default=2, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='music', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--epochs', default=100, help='에폭 수', type=int)
parser.add_argument('--load_path', default=None, help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default="result/dec0722", help='모델 저장 경로')
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=1, type=int)

args = parser.parse_args()


# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers


def get_current_datetime():
    from datetime import datetime
    now = datetime.now()
    dt_name = now.strftime("%m_%d_%Y__%H_%M_%S")
    return dt_name


# if IS_ON_GOOGLE_COLAB:
#     FOLDER_ROOT = "/content/drive/MyDrive/magisterka/SheetMusicGenerator2"
# else:
#     FOLDER_ROOT = "."

TEST_RUN = True
NORMALIZE_NOTES = True
USE_COMPUTED_VALUES = True
USE_SAVE_POINT = False

NORMALIZATION_BOUNDARIES = [3, 4]
EPOCHS = 250
LATENT_VECTOR_DIM = 2
BATCH_SIZE = 256
SEQUENCE_LENGTH = 32

FOLDER_ROOT = "."
# COMPUTED_INT_TO_NOTE_PATH = "/content/drive/MyDrive/magisterka/SheetMusicGenerator2/AUTOENCODER/data/dicts/int_to_note_08_19_2021__17_25_44"
# COMPUTED_INT_TO_DURATION_PATH = "/content/drive/MyDrive/magisterka/SheetMusicGenerator2/AUTOENCODER/data/dicts/int_to_duration_08_19_2021__17_25_44"
# COMPUTED_NOTES_PATH = "/content/drive/MyDrive/magisterka/SheetMusicGenerator2/AUTOENCODER/data/notes/notes_08_19_2021__17_25_44"
# COMPUTED_DURATIONS_PATH = "/content/drive/MyDrive/magisterka/SheetMusicGenerator2/AUTOENCODER/data/durations/durations_08_19_2021__17_25_44"
COMPUTED_DATA_PATH = "AUTOENCODER/data/data_file_12_06_2021__19_53_42"

SAVE_POINT = "AUTOENCODER/checkpoints/08_19_2021__18_34_10/epoch=014-loss=383.5284-acc=0.0000.hdf5"
AUTOENCODER = "TRANSFORMER"

MODEL_NAME = AUTOENCODER
MODEL_FOLDER_ROOT = os.path.join(FOLDER_ROOT, MODEL_NAME)
CURR_DT = get_current_datetime()
MODEL_DIR_PATH = os.path.join(MODEL_FOLDER_ROOT, "generated_models")
OCCURENCES = os.path.join(MODEL_FOLDER_ROOT, "data", "occurences")

DATA_DIR = os.path.join(MODEL_FOLDER_ROOT, "data")
DATA_NOTES_DIR = os.path.join(DATA_DIR, "notes")
DATA_DURATIONS_DIR = os.path.join(DATA_DIR, "durations")

DATA_FILE_PATH = os.path.join(DATA_DIR, "data_file_" + str(CURR_DT))

DATA_DICTS_DIR = os.path.join(DATA_DIR, "dicts")
DATA_INT_TO_NOTE_PATH = os.path.join(DATA_DICTS_DIR, "int_to_note_" + str(CURR_DT))
DATA_INT_TO_DURATION_PATH = os.path.join(DATA_DICTS_DIR, "int_to_duration_" + str(CURR_DT))
DATA_NOTES_PATH = os.path.join(DATA_NOTES_DIR, "notes_" + str(CURR_DT))

DATA_DURATIONS_PATH = os.path.join(DATA_DURATIONS_DIR, "durations_" + str(CURR_DT))
# MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs")
MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs_smaller")
# MIDI_SONGS_DIR = os.path.join(FOLDER_ROOT, "midi_songs_medium")
MIDI_GENERATED_DIR = os.path.join(MODEL_FOLDER_ROOT, "midi_generated")
MIDI_SONGS_REGEX = os.path.join(MIDI_SONGS_DIR, "*.mid")
CHECKPOINTS_DIR = os.path.join(MODEL_FOLDER_ROOT, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINTS_DIR, str(CURR_DT))
LOGS_DIR = os.path.join(MODEL_FOLDER_ROOT, "logs")

LOG = os.path.join(LOGS_DIR, str(CURR_DT))
all_paths = [MODEL_DIR_PATH, OCCURENCES, DATA_NOTES_DIR, DATA_DURATIONS_DIR, DATA_DICTS_DIR,
             MIDI_GENERATED_DIR, CHECKPOINTS_DIR, CHECKPOINT, LOGS_DIR, LOG]

for path in all_paths:
    Path(path).mkdir(parents=True, exist_ok=True)
# load data
dataset = DataNew('midi_processed', max_seq, batch_size)
# print(dataset)


# load model
curr_dt = get_current_datetime()
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
filepath = CHECKPOINT + str(curr_dt) + "/" + "epoch:{epoch:02d}-loss:{loss:.4f}" #-acc:{binary_accuracy:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='max'
)
#
# checkpoint = ModelCheckpoint(
#     filepath,
#     monitor='binary_accuracy',
#     verbose=0,
#     save_best_only=True,
#     mode='max'
# )
log = tf.keras.callbacks.TensorBoard(log_dir=LOG + curr_dt),

callbacks_list = [checkpoint, log]
# define model
mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)
mt.run_eagerly = True
# batch = (dataset.generators_dict["train"][0])
# print(type(dataset.generators_dict["train"]))
# mt.train_on_batch(batch[0], batch[1])
# print(len(dataset.generators_dict["train"][0]))
# print(len(dataset.generators_dict["train"][1]))
# print(len(dataset.generators_dict["train"][2]))
# ds = tf.data.Dataset.from_generator(dataset.generators_dict["train"].__iter__(), output_types=tf.float32)
mt.fit(x=dataset.generators_dict["train"][0], y=dataset.generators_dict["train"][1], epochs=EPOCHS, callbacks=callbacks_list)
# mt.fit(dataset.generators_dict["train"], epochs=EPOCHS, callbacks=callbacks_list)
# mt.fit(dataset.generators_dict["train"], epochs=EPOCHS, callbacks=callbacks_list)

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)
EPOCHS=2

