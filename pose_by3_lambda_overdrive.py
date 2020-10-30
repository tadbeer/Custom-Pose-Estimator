from keras import backend as K


def Split1(input_features):
    breadth = K.int_shape(input_features)[2]
    split_side_length = int(breadth / 3)
    split_begin = 0
    split_end = split_begin + split_side_length
    #print(breadth, split_begin, split_end)

    split_features = input_features[:, :, split_begin: split_end, :]

    return(split_features)


def Split2(input_features):
    breadth = K.int_shape(input_features)[2]
    split_side_length = int(breadth / 3)
    split_begin = split_side_length
    split_end = split_begin + split_side_length
    #print(breadth, split_begin, split_end)

    split_features = input_features[:, :, split_begin: split_end, :]

    return(split_features)


def Split3(input_features):

    breadth = K.int_shape(input_features)[2]
    split_side_length = int(breadth / 3)
    split_begin = split_side_length * 2
    split_end = split_begin + split_side_length
    #print(breadth, split_begin, split_end)

    split_features = input_features[:, :, split_begin: split_end, :]

    return(split_features)


def GetChannel1(input_features):
    channel = input_features[:, :, :, 0:1]
    return(channel)


def GetChannel2(input_features):
    channel = input_features[:, :, :, 1:2]
    return(channel)


def GetChannel3(input_features):
    channel = input_features[:, :, :, 2:3]
    return(channel)


def GetChannel4(input_features):
    channel = input_features[:, :, :, 3:4]
    return(channel)


def GetChannel5(input_features):
    channel = input_features[:, :, :, 4:5]
    return(channel)


def GetChannel6(input_features):
    channel = input_features[:, :, :, 5:6]
    return(channel)


def GetChannel7(input_features):
    channel = input_features[:, :, :, 6:7]
    return(channel)


def GetChannel8(input_features):
    channel = input_features[:, :, :, 7:8]
    return(channel)


def GetChannel9(input_features):
    channel = input_features[:, :, :, 8:9]
    return(channel)


def GetChannel10(input_features):
    channel = input_features[:, :, :, 9:10]
    return(channel)


def GetChannel11(input_features):
    channel = input_features[:, :, :, 10:11]
    return(channel)


def GetChannel12(input_features):
    channel = input_features[:, :, :, 11:12]
    return(channel)


def GetChannel13(input_features):
    channel = input_features[:, :, :, 12:13]
    return(channel)


def GetChannel14(input_features):
    channel = input_features[:, :, :, 13:14]
    return(channel)


def GetChannel15(input_features):
    channel = input_features[:, :, :, 14:15]
    return(channel)


def GetChannel16(input_features):
    channel = input_features[:, :, :, 15:16]
    return(channel)


def GetPortion1(input_features):
    portion1 = input_features[:, 0:1, :]
    return(portion1)


def GetPortion2(input_features):
    portion1 = input_features[:, 1:2, :]
    return(portion1)


def GetPortion3(input_features):
    portion1 = input_features[:, 2:3, :]
    return(portion1)
