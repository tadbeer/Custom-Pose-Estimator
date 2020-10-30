from keras.layers import Conv2D, Lambda, Concatenate, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Lambda, Reshape, Conv1D
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K

from pose_by3_lambda_overdrive import Split1, Split2, Split3,\
    GetChannel1, GetChannel2, GetChannel3, GetChannel4, GetChannel5, GetChannel6,\
    GetChannel7, GetChannel8, GetChannel9, GetChannel10, GetChannel11,\
    GetChannel12, GetChannel13, GetChannel14, GetChannel15, GetChannel16,\
    GetPortion1, GetPortion2, GetPortion3

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def Concat(feature_list):
    return(K.concatenate(feature_list, axis=-1))


def AtrousFeatures(input_features):
    rate_1 = Conv2D(16, (3, 3), padding='same', dilation_rate=(1, 1), name='block1_conv1_dilate1')(input_features)
    # rate_2 = Conv2D(8, (3, 3), padding='same', dilation_rate=(2, 2), name='block1_conv1_dilate2')(input_features)
    rate_3 = Conv2D(8, (3, 3), padding='same', dilation_rate=(3, 3), name='block1_conv1_dilate3')(input_features)
    # rate_4 = Conv2D(8, (3, 3), padding='same', dilation_rate=(4, 4), name='block1_conv1_dilate4')(input_features)
    rate_5 = Conv2D(8, (3, 3), padding='same', dilation_rate=(5, 5), name='block1_conv1_dilate5')(input_features)
    # rate_6 = Conv2D(8, (3, 3), padding='same', dilation_rate=(6, 6), name='block1_conv1_dilate6')(input_features)
    rate_7 = Conv2D(4, (3, 3), padding='same', dilation_rate=(7, 7), name='block1_conv1_dilate7')(input_features)
    # rate_8 = Conv2D(8, (3, 3), padding='same', dilation_rate=(8, 8), name='block1_conv1_dilate8')(input_features)
    rate_9 = Conv2D(4, (3, 3), padding='same', dilation_rate=(9, 9), name='block1_conv1_dilate9')(input_features)
    # rate_10 = Conv2D(8, (3, 3), padding='same', dilation_rate=(10, 10), name='block1_conv1_dilate10')(input_features)
    rate_11 = Conv2D(4, (3, 3), padding='same', dilation_rate=(11, 11), name='block1_conv1_dilate11')(input_features)
    rate_13 = Conv2D(2, (3, 3), padding='same', dilation_rate=(13, 13), name='block1_conv1_dilate13')(input_features)
    rate_15 = Conv2D(2, (3, 3), padding='same', dilation_rate=(15, 15), name='block1_conv1_dilate15')(input_features)

    feature_list = [rate_1, rate_3, rate_5, rate_7, rate_9, rate_11, rate_13, rate_15]
    result = Concatenate(axis=-1)(feature_list)
    #result = Lambda(Concat)(feature_list)

    return(result)


def _shortcut(inp, residual):
    input_shape = K.int_shape(inp)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = inp

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inp)
    # print(shortcut.shape)
    z = add([shortcut, residual])
    return z


def Resnet(input_features):

    # BLOCK_1
    x = AtrousFeatures(input_features)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block1_BN1')(x)
    x = Activation('relu')(x)
    y = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='block1_pool1')(x)

    # BLOCK_2

    # branch1
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch1_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(16, (1, 1), strides=(2, 2), padding='same', name='block2_branch1_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch1_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='block2_branch1_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch1_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block2_branch1_conv3')(x)
    y = _shortcut(y, x)

    # branch2
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch2_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='block2_branch2_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch2_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='block2_branch2_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch2_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block2_branch2_conv3')(x)
    y = _shortcut(y, x)

    # branch3
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch3_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='block2_branch3_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch3_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(616, (3, 3), strides=(1, 1), padding='same', name='block2_branch3_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block2_branch3_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block2_branch3_conv3')(x)
    y = _shortcut(y, x)

    # BLOCK_3

    # branch1
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch1_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 1), strides=(2, 2), padding='same', name='block3_branch1_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch1_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block3_branch1_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch1_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='block3_branch1_conv3')(x)
    y = _shortcut(y, x)

    # branch2
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch2_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='block3_branch2_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch2_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block3_branch2_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch2_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='block3_branch2_conv3')(x)
    y = _shortcut(y, x)

    # branch3
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch3_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='block3_branch3_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch3_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block3_branch3_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch3_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='block3_branch3_conv3')(x)
    y = _shortcut(y, x)

    # branch4
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch4_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), padding='same', name='block3_branch4_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch4_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='block3_branch4_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block3_branch4_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='block3_branch4_conv3')(x)
    y = _shortcut(y, x)

    # BLOCK_4

    # branch1
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch1_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch1_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch1_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch1_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch1_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch1_conv3')(x)
    y = _shortcut(y, x)

    # branch2
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch2_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch2_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch2_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch2_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch2_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch2_conv3')(x)
    y = _shortcut(y, x)

    # branch3
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch3_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch3_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch3_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch3_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch3_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch3_conv3')(x)
    y = _shortcut(y, x)

    # branch4
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch4_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch4_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch4_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch4_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch4_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch4_conv3')(x)
    y = _shortcut(y, x)

    # branch5
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch5_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch5_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch5_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch5_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch5_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch5_conv3')(x)
    y = _shortcut(y, x)

    # branch6
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch6_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='block4_branch6_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch6_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='block4_branch6_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block4_branch6_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='block4_branch6_conv3')(x)
    y = _shortcut(y, x)

    # BLOCK_5_ATROUS

    # branch1
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch1_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch1_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch1_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='block5_branch1_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch1_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch1_conv3')(x)
    y = _shortcut(y, x)

    # branch2
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch2_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch2_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch2_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='block5_branch2_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch2_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch2_conv3')(x)
    y = _shortcut(y, x)

    # branch3
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch3_BN1')(y)
    x = Activation('relu')(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch3_conv1')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch3_BN2')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='block5_branch3_conv2')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='block5_branch3_BN3')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='block5_branch3_conv3')(x)
    y = _shortcut(y, x)

    return y


def SplitBy3(input_features, split_portion):

    #breadth = int(input_features.shape[2])
    #breadth = K.int_shape(input_features)[2]
    #split_side_length = int(breadth / 3)
    #split_begin = (split_portion - 1) * split_side_length
    #split_end = split_begin + split_side_length
    #print(breadth, split_begin, split_end)

    #split_features = input_features[:, :, split_begin: split_end, :]

    if split_portion == 1:
        split_features = Lambda(Split1)(input_features)
    if split_portion == 2:
        split_features = Lambda(Split2)(input_features)
    if split_portion == 3:
        split_features = Lambda(Split3)(input_features)

    return(split_features)


def ReduceToColumns(input_features, split_portion):
    num = split_portion
    # layer_num = 1
    # x = input_features
    # while(x.shape)
    # if
    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', name='split{}_red2col_conv1'.format(num))(input_features)
    # print(x.shape)
    x = Conv2D(128, (3, 3), strides=(1, 2), padding='same', name='split{}_red2col_conv2'.format(num))(x)
    # print(x.shape)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='split{}_red2col_conv3'.format(num))(x)
    # print(x.shape)
    x = Conv2D(32, (3, 3), strides=(1, 2), padding='same', name='split{}_red2col_conv4'.format(num))(x)
    # print(x.shape)
    x = Conv2D(16, (3, 3), strides=(2, 1), padding='valid', name='split{}_red2col_conv5'.format(num))(x)
    # print(x.shape)
    #x = Conv2D(4, (3, 3), strides=(1, 2), padding='same', name='split{}_red2col_conv6'.format(num))(x)
    # print(x.shape)
    #x = Conv2D(2, (2, 2), strides=(2, 1), padding='same', name='split{}_red2col_conv7'.format(num))(x)
    # print(x.shape)
    #x = Conv2D(1, (2, 1), strides=(2, 1), padding='same', name='split{}_red2col_conv8'.format(num))(x)
    # print(x.shape)
    #x = Flatten()(x)
    # return(x)

    num_channels = K.int_shape(x)[-1]
    # print(num_channels)
    #column_channels_list = []
    # for channel_index in range(num_channels):
    #channel = x[:, :, :, channel_index:channel_index + 1]
    # column_channels_list.append(channel)
    channel1 = Lambda(GetChannel1)(x)
    channel2 = Lambda(GetChannel2)(x)
    channel3 = Lambda(GetChannel3)(x)
    channel4 = Lambda(GetChannel4)(x)
    channel5 = Lambda(GetChannel5)(x)
    channel6 = Lambda(GetChannel6)(x)
    channel7 = Lambda(GetChannel7)(x)
    channel8 = Lambda(GetChannel8)(x)
    channel9 = Lambda(GetChannel9)(x)
    channel10 = Lambda(GetChannel10)(x)
    channel11 = Lambda(GetChannel11)(x)
    channel12 = Lambda(GetChannel12)(x)
    channel13 = Lambda(GetChannel13)(x)
    channel14 = Lambda(GetChannel14)(x)
    channel15 = Lambda(GetChannel15)(x)
    channel16 = Lambda(GetChannel16)(x)

    column_channels_list = [channel1, channel2, channel3, channel4, channel5, channel6,
                            channel7, channel8, channel9, channel10, channel11, channel12,
                            channel13, channel14, channel15, channel16]

    return(column_channels_list)


def EnqeueColumns(list_of_lists_of_columns):

    portion_columns_list = list_of_lists_of_columns

    num_channels = len(list_of_lists_of_columns[0])
    enqeued_columns_list = []

    for channel_index in range(num_channels):
        channels_list = [portion_columns[channel_index] for portion_columns in portion_columns_list]
        channels = Concatenate(axis=1)(channels_list)
        num_elements = K.int_shape(channels)[1]
        channels = Reshape((num_elements, 1))(channels)
        # print(channels.shape)
        enqeued_columns_list.append(channels)

    enqeued_columns = Concatenate(axis=-1)(enqeued_columns_list)
    # print(columns.shape)

    return (enqeued_columns)


def SharedLearning(enqeued_columns):

    portion_column_length = int(K.int_shape(enqeued_columns)[1] / 3)
    one_d_condense = Conv1D(8, portion_column_length, strides=portion_column_length, padding='same')(enqeued_columns)

    return(one_d_condense)


def RearrangeLateral(input_features):

    column_list = []
    #print(input_features[:, 0:1, :].shape)

    # for portion in range(3):
    #    column_list.append(input_features[:, portion:portion + 1, :])

    portion1 = Lambda(GetPortion1)(input_features)
    portion2 = Lambda(GetPortion2)(input_features)
    portion3 = Lambda(GetPortion3)(input_features)

    column_list = [portion1, portion2, portion3]

    rearranged = Concatenate(axis=-1)(column_list)

    num_elements = K.int_shape(rearranged)[-1]
    rearranged = Reshape((num_elements, 1))(rearranged)
    rearranged = Flatten()(rearranged)

    return(rearranged)


def Get6(input_features):

    x = Dense(12)(input_features)
    x = Dense(6)(x)

    return(x)


def Get2(input_features, split_portion):
    num = split_portion
    x = Dense(6, name='split{}_dense6'.format(num))(input_features)
    x = Dense(2, name='split{}_dense2'.format(num))(x)

    return(x)
