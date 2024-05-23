
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation
from keras import backend as K

def jacard(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (intersection+1)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection+1)

def dynamic_model(n_classes=3, patch_size=128, num_bands=8, filt1=3,filt2=3,filt3=3,filt4=3, filt5=3, filt6=3, filt7=3, filt8=3, filt9=3,
                drop1=0.1, drop2=0.1, drop3=0.1, drop4=0.1, drop5=0.1,
                norm1=1, norm2=1, norm3=1, norm4=1, norm5=1, norm6=1, norm7=1, norm8=1, norm9=1):

    inputs=Input((patch_size,patch_size,num_bands))
    s=inputs

    #contracting path
    c1=Conv2D(16, (filt1,filt1), activation=None, kernel_initializer='he_normal', padding='same')(s)
    if(norm1==1):
        c1 = BatchNormalization()(c1)
    c1=Activation("relu")(c1)
    c1 = Dropout(drop1)(c1)
    c1=Conv2D(16, (filt1,filt1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1=MaxPooling2D((2,2))(c1)

    c2 = Conv2D(32, (filt2, filt2), activation=None, kernel_initializer='he_normal', padding='same')(p1)
    if(norm2==1):
        c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    c2 = Dropout(drop1)(c2)
    c2 = Conv2D(32, (filt2, filt2), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (filt3, filt3), activation=None, kernel_initializer='he_normal', padding='same')(p2)
    if(norm3==1):
        c3 = BatchNormalization()(c3)
    c3 = Activation("relu")(c3)
    c3 = Dropout(drop2)(c3)
    c3 = Conv2D(64, (filt3, filt3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (filt4, filt4), activation=None, kernel_initializer='he_normal', padding='same')(p3)
    if(norm4==1):
        c4 = BatchNormalization()(c4)
    c4 = Activation("relu")(c4)
    c4 = Dropout(drop2)(c4)
    c4 = Conv2D(128, (filt4, filt4), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (filt5, filt5), activation=None, kernel_initializer='he_normal', padding='same')(p4)
    if(norm5==1):
        c5 = BatchNormalization()(c5)
    c5 = Activation("relu")(c5)
    c5 = Dropout(drop3)(c5)
    c5 = Conv2D(256, (filt5, filt5), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = Dropout(drop3)(c5)
    c5 = Conv2D(256, (filt5, filt5), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #expanding path
    u6=Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6=concatenate([u6,c4])
    c6=Conv2D(128, (filt6,filt6), activation=None, kernel_initializer='he_normal', padding='same')(u6)
    if(norm6==1):
        c6 = BatchNormalization()(c6)
    c6 = Activation("relu")(c6)
    c6 = Dropout(drop4)(c6)
    c6=Conv2D(128, (filt6,filt6), activation='relu', kernel_initializer='he_normal', padding='same')(c6)



    u7=Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7=concatenate([u7,c3])
    c7=Conv2D(64, (filt7,filt7), activation=None, kernel_initializer='he_normal', padding='same')(u7)
    if(norm7==1):
        c7 = BatchNormalization()(c7)
    c7 = Activation("relu")(c7)
    c7 = Dropout(drop4)(c7)
    c7=Conv2D(64, (filt7,filt7), activation='relu', kernel_initializer='he_normal', padding='same')(c7)


    u8=Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8=concatenate([u8,c2])
    c8=Conv2D(32, (filt8,filt8), activation=None, kernel_initializer='he_normal', padding='same')(u8)
    if(norm8==1):
        c8 = BatchNormalization()(c8)
    c8 = Activation("relu")(c8)
    c8 = Dropout(drop5)(c8)
    c8=Conv2D(32, (filt8,filt8), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (filt9, filt9), activation=None, kernel_initializer='he_normal', padding='same')(u9)
    if(norm9==1):
        c9 = BatchNormalization()(c9)
    c9 = Activation("relu")(c9)
    c9 = Dropout(drop5)(c9)
    c9 = Conv2D(16, (filt9, filt9), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    #prob for each class
    outputs= Conv2D(n_classes,(1,1), activation='softmax')(c9)

    model=Model(inputs=[inputs], outputs=[outputs])
    return model
