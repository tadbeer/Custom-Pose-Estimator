
import keras
from keras.layers import Input, Concatenate, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import generic_utils
from keras import backend as K
from pose_by3_modellers import Resnet, SplitBy3, ReduceToColumns, EnqeueColumns, SharedLearning,\
    RearrangeLateral, Get6

from pose_by3_data import fetch_data, Batchify
import numpy as np
import random
import time

k = 22
breadth = 8 * 6 * k
hieght = 8 * 4 * k

image_shape = (hieght, breadth, 3)

batch_size = 4
epochs = 3
val_batch_size = 4


def build_model(image_shape):
    image = Input(shape=image_shape)
    # print(image.shape)

    resnet_features = Resnet(image)
    # print(resnet_features.shape)

    left_features = SplitBy3(resnet_features, 1)
    central_features = SplitBy3(resnet_features, 2)
    right_features = SplitBy3(resnet_features, 3)
    # print(left_features.shape)

    left_columns_list = ReduceToColumns(left_features, 1)
    central_columns_list = ReduceToColumns(central_features, 2)
    right_columns_list = ReduceToColumns(right_features, 3)
    # print(left_columns.shape)

    portion_columns_list = [left_columns_list, central_columns_list, right_columns_list]
    enqeued_columns = EnqeueColumns(portion_columns_list)
    # print(enqeued_columns.shape)

    one_d_condense = SharedLearning(enqeued_columns)
    # print(one_d_condense.shape)

    laterally_arranged = RearrangeLateral(one_d_condense)
    # print(laterally_arranged.shape)

    locations = Get6(laterally_arranged)
    # print(locations.shape)

    """
    left_points = Get2(left_column, 1)
    central_points = Get2(central_column, 2)
    right_points = Get2(right_column, 3)
    # print(left_points.shape)

    location_list = [left_points, central_points, right_points]
    locations = Concatenate(axis=-1)(location_list)

    # locations = Lambda(Concat)(location_list)
    print(locations.shape)
	"""
    pose_by3 = Model(inputs=image, outputs=locations)

    return(pose_by3)


model = build_model(image_shape)

adam = Adam()
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

train_path = 'temp_train'
val_path = 'temp_test'

train_images, train_points = fetch_data(data_path=train_path, resize_dimensions=image_shape, mode='train')
val_images, val_points = fetch_data(data_path=val_path, resize_dimensions=image_shape, mode='val')

print(len(train_images), len(train_points))
print(len(val_images), len(val_points))

print(batch_size, val_batch_size)

train_images, train_points, train_steps = Batchify(X=train_images, y=train_points, batch_size=batch_size)
val_images, val_points, val_steps = Batchify(X=val_images, y=val_points, batch_size=val_batch_size)

print(len(train_images), len(train_points), train_steps)
print(len(val_images), len(val_points), val_steps)

"""
ckpt_loss = ModelCheckpoint('pose_by3_loss-{loss:.4f}_{epoch:03d}.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
ckpt_val_loss = ModelCheckpoint('pose_by3_val_loss-{val_loss:.4f}_{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
photo = TensorBoard(log_dir='logs')
callbacks = [ckpt_loss, ckpt_val_loss, photo]

model.fit(x=train_images, y=train_points,
          batch_size=batch_size, epochs=epochs,
          verbose=1, callbacks=callbacks,
          validation_data=(val_images, val_points), validation_steps=val_steps,
          shuffle=True, initial_epoch=0)
"""

metrics = {}
for metric in ['loss', 'acc', 'val_loss', 'val_acc']:
    metrics[metric] = np.zeros(train_steps)

metrics_log = []

start_time = time.time()

for epoch in range(epochs):

	print()

    filler = list(zip(train_images, train_points))
    random.shuffle(filler)
    train_images[:], train_points[:] = zip(*filler)

    progbar = generic_utils.Progbar(epochs)
    val_counter = 0

    for step, (batch, labels) in enumerate(zip(train_images, train_points)):
        batch, labels = np.array(batch), np.array(labels)
        # batch = np.expand_dims(batch,axis=0)
        # labels = np.expand_dims(labels,axis=0)

        val_batch = val_images[val_counter]
        val_labels = val_points[val_counter]
        val_batch, val_labels = np.array(val_batch), np.array(val_labels)
        # val_batch = np.expand_dims(val_batch, axis=0)
        # val_labels = np.expand_dims(val_labels, axis=0)
        val_counter = val_counter + 1 if val_counter < val_steps else 0

        training_loss = model.train_on_batch(x=batch, y=labels)
        val_loss = model.test_on_batch(x=val_batch, y=val_labels)

        metrics['loss'][step] = training_loss[0]
        metrics['acc'][step] = training_loss[1]
        metrics['val_loss'][step] = val_loss[0]
        metrics['val_acc'][step] = val_loss[1]
        progbar.update(step, [('training_loss', np.mean(metrics['loss'][:step])),
                              ('\t training_acc', np.mean(metrics['acc'][:step])),
                              ('\t val_loss', np.mean(metrics['val_loss'][:step])),
                              ('\t val_acc', np.mean(metrics['val_acc'][:step]))])

        if step == train_steps:
            current_loss = np.mean(metrics['loss'])
            current_accuracy = np.mean(metrics['acc'])
            current_val_loss = np.mean(metrics['val_loss'])
            current_val_accuracy = np.mean(metrics['val_acc'])

            print('Current Training Loss : {}'.format(current_loss))
            print('Current training Accuracy : {}'.format(current_accuracy))
            print('Current Validation Loss: {}'.format(current_val_loss))
            print('Current Validaton Accuracy : {}'.format(current_val_accuracy))
            print('Time Elapsed: {}'.format(time.time() - start_time))

            loggy = [current_loss, current_accuracy, current_val_loss, current_val_accuracy]
            metrics_log.append(loggy)
            np.save('log.npy', np.array(metrics_log))

            start_time = time.time()
