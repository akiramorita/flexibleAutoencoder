from src.utils_tailHandler import log_to
import os, random, cv2
from collections.abc import Callable
import pandas as pd
import numpy as np
from keras import regularizers, callbacks, activations, initializers, constraints
from keras.layers import Input, Dense, Flatten, Lambda, Reshape, Dropout, BatchNormalization, Activation, Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard


@log_to('tail')
class AutoencoderTrain(Callable):
    '''
    Given autoencoder,optimizer, loss, metrics, filename, train autoencoder and record scores under certain settings.
    '''
    def __init__(self,
                 train_generator = None,
                 valid_generator = None,
                 train_epochs = None,
                 patience = None,
                 verbose = 1,
                 period = 100,
                 save_dir = None,
                 AutoencoderEvals = None):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.train_epochs = train_epochs
        self.patience = patience
        self.verbose = verbose
        self.period = period
        self.save_dir = save_dir
        self.AutoencoderEvals = AutoencoderEvals
    
    def __call__(self,autoencoder,optimizer,encoderLayerName, loss = 'binary_crossentropy', metrics = ['accuracy'], AEfilename = 'ae_weights.h5',ENfilename = 'en_weights.h5'):
        self.tail.debug("autoencoder {0} optimizer {1} loss {2} metrics {3} AEfilename {4} encoderLayerName {5} ENfilename {6} attr {7}".format(autoencoder, optimizer, loss, metrics, AEfilename, encoderLayerName, ENfilename, self.__dict__))
        estop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=self.verbose, mode='auto',restore_best_weights=True)
        checkpoint = callbacks.ModelCheckpoint(self.save_dir + '/' + 'ep{epoch:03d}-loss{loss:.7f}-val_loss{val_loss:.7f}.h5',monitor='val_loss', save_weights_only=False, save_best_only=True, period=self.period)
        autoencoder.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        autoencoder.fit_generator(generator=self.train_generator,validation_data=self.valid_generator,use_multiprocessing=True,workers=4, epochs=self.train_epochs,callbacks=[estop,checkpoint],verbose=self.verbose,steps_per_epoch=len(self.train_generator),validation_steps=len(self.valid_generator))
        autoencoder.save(self.save_dir + '/' + AEfilename)
        encoder = Model(autoencoder.input, autoencoder.get_layer(encoderLayerName).output)
        encoder.save(self.save_dir + '/' + ENfilename)
 
        dic1 = {}
        for ev in self.AutoencoderEvals:
            sourceCategory,score = ev(autoencoder,self.save_dir)
            dic1[sourceCategory + '_score'] = score
        return dic1
            

@log_to('tail')
class AutoencoderEval(Callable):
    '''
    Given autoencoder,save_dir, provide autoencoder scores for specified datasets under certain settings.
    '''
    def __init__(self,
                 sourceCategory='valid',
                 valid_generator=None,
                 sample_X_valid=None,
                 seed=1,
                 n_sample=10):
        self.sourceCategory = sourceCategory
        self.valid_generator = valid_generator
        self.sample_X_valid = sample_X_valid
        self.seed = seed
        self.n_sample = n_sample
    
    def __call__(self,autoencoder,save_dir):
        self.tail.debug("attr {0}".format(self.__dict__))
        score = autoencoder.evaluate_generator(generator=self.valid_generator, workers=1, use_multiprocessing=True, verbose=0,steps=len(self.valid_generator))
        pred = autoencoder.predict(self.sample_X_valid)
        image1, image2 = None, None
        for i in range(0,len(pred)):
            image1 = np.multiply(pred[i], 255).astype(int)
            image2 = np.multiply(self.sample_X_valid[i], 255).astype(int)
            cv2.imwrite(save_dir + '/' + self.sourceCategory + '_eval_pred_image_' + str(i) + '.png',image1)
            cv2.imwrite(save_dir + '/' + self.sourceCategory + '_eval_orig_image_' + str(i) + '.png',image2)
    
        with open(save_dir + '/setting.txt','a') as f:
            f.write(self.sourceCategory + '_eval_score\t\t' + str(score) + '\n')
        self.tail.debug("sourceCategory {0} score {1}".format(self.sourceCategory,score))
        return self.sourceCategory,score


def sample_X_valid(imageShape,valid_dir,seed=1,n_sample=10):
    '''
    Provide smapled images as np.array format (n_smaple,imageShape[0],imageShape[1],1) from given directory
    '''
    random.seed(seed)
    d = valid_dir + '/class_0'
    X = np.empty((n_sample,*imageShape, 1))
    for i, img in enumerate([np.expand_dims(cv2.imread(os.path.join(d,file),0),axis=-1) for file in random.sample(os.listdir(d),n_sample)]):
        X[i,] = np.multiply(img, 1./255)
    return X


@log_to('tail')
class Generator(Callable):
    ''' Define generator input, output and data conversion function ''' 
    def __init__(self, func, sourceType='directory', datagen=None, shuffle=True, save_dir=None,save_prefix=None,cropShape=(1200,600),batch_size=32):
        self.func = func
        self.sourceType = sourceType
        self.datagen = datagen(func)
        self.shuffle = shuffle
        self.save_dir = save_dir
        if self.save_dir is not None and not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
        self.save_prefix = save_prefix
        self.cropShape = cropShape
        self.batch_size = batch_size
    
    def __call__(self,sourceDir=None, dataframe=None, x_col='pairPath',y_col='class',color_mode="grayscale"):
        if self.func == "argumentation":
            class_mode = 'categorical'
        elif self.func == "autoencoder":
            class_mode = 'input'
        elif self.func == "triplet":
            class_mode = 'binary'
        elif self.func == "distance":
            class_mode = 'raw'
        else:
            print('Wrong arguments to Datagen')
            raise Exception
            
        self.tail.debug("sourceDir {0} class_mode {1} attr {2}".format(sourceDir,class_mode,self.__dict__))
        if self.sourceType == 'directory':
            return self.datagen.flow_from_directory(
                    sourceDir,
                    target_size = self.cropShape,
                    batch_size = self.batch_size,
                    color_mode = color_mode,
                    class_mode = class_mode,
                    shuffle = self.shuffle,
                    seed = 1,
                    save_to_dir = self.save_dir,
                    save_prefix = self.save_prefix)
        elif self.sourceType == 'dataframe':
            return self.datagen.flow_from_dataframe(
                    dataframe = dataframe,
                    x_col = x_col,
                    y_col = y_col,
                    shuffle = self.shuffle,
                    target_size = self.cropShape,
                    batch_size = self.batch_size,
                    color_mode = color_mode,
                    class_mode = class_mode,
                    save_to_dir = self.save_dir,
                    save_prefix = self.save_prefix)


@log_to('tail')
class Datagen(Callable):
    ''' Given func, provide ImageDataGenerator function under certain settings. '''
    def __init__(self,rescale=1./255,preprocessing_function=None,width_shift_range=0.04,rotation_range=90,horizontal_flip=True,vertical_flip=True):
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.width_shift_range = width_shift_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        
    def __call__(self,func):
        self.tail.debug("func {0} attr {1}".format(func,self.__dict__))
        if func == "argumentation":
            return ImageDataGenerator(
                    width_shift_range=0.04,
                    vertical_flip=True,
                    preprocessing_function=self.preprocessing_function)
        elif func == "autoencoder":
            return ImageDataGenerator(
                    horizontal_flip = self.horizontal_flip,
                    vertical_flip = self.vertical_flip,
                    rescale=self.rescale)
        elif func == "triplet" or func == "distance":
            return ImageDataGenerator(rescale=self.rescale)
        else:
            print('Wrong arguments to Datagen')
            raise Exception


@log_to('tail')
class AutoencoderConv2D(Callable):
    '''
    With given inpu_img and dims, provide autoencoder model, encoder model and flattenLength under certain settings.
    '''
    def __init__(self,
                 paras=None,
                 input_shape=None,
                 paras_cnn=None,
                 last_activation=None,
                 Conv2D_BN=None,
                 AutoencoderMLP=None,
                 AutoencoderMLP_switch=None,
                 dropRate=None):
        self.input_shape=input_shape
        self.filters=paras_cnn['filters']
        self.pooling=paras_cnn['pooling']
        self.kernel_size=paras_cnn['kernel_size']
        self.strides=paras_cnn['strides']
        self.last_activation=last_activation
        self.dropRate=dropRate 
        self.AutoencoderMLP_switch=AutoencoderMLP_switch
        self.Conv2D_BN = Conv2D_BN
        self.AutoencoderMLP = AutoencoderMLP
        
    def __call__(self,input_img,dims):
        self.tail.debug("input_img.shape {0} dims{1} attr {2}".format(input_img.shape,dims,self.__dict__))
        x = input_img
        ps = []
        for i in range(len(self.filters)):
            x, p = self.Conv2D_BN(x,i,UpSampling=False)
            ps += [p]
        
        x = Flatten()(x)
        x = Dropout(self.dropRate)(x)

        y = self.input_shape[0]
        for i in range(len(self.filters)):
            y = y//(self.strides[i]*ps[i])
        
        flattenLength = int(y*y*self.filters[-1])   # assuming img shape is square
        print('flattenLength: ',flattenLength)

        encoded = x
        if self.AutoencoderMLP_switch:
            x, encoded = self.AutoencoderMLP(dims,flattenLength,x,last_activation=self.last_activation)
        
        x, _ = poolingSelection(x,self.pooling[-1],UpSampling=True)
        x = Reshape((int(y), int(y), self.filters[-1]))(x)      # assuming img shape is square

        for i in range(len(self.filters)-2,-1,-1):
            x, _ = self.Conv2D_BN(x,i,UpSampling=True)

        x = Conv2DTranspose(self.input_shape[2], kernel_size=self.kernel_size[0], strides=self.strides[0],activation=None, padding='same', name='deconv_end')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        self.tail.debug("decoded.shape {0} encoded.shape {1} flattenLength {2} y {0}".format(decoded.shape,encoded.shape,flattenLength,y))
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder'), flattenLength


@log_to('tail')
class Conv2D_BN(Callable):
    '''
    Given input_img,i,UpSampling, provide Conv2D processed np.array and p (flattenLength element)
    with BatchNormalization and activation under certain settings.
    '''
    def __init__(self,
                 input_shape=None,
                 paras_cnn=None,
                 activation=None):
        self.input_shape=input_shape
        self.filters=paras_cnn['filters']
        self.pooling=paras_cnn['pooling']
        self.kernel_size=paras_cnn['kernel_size']
        self.strides=paras_cnn['strides']
        self.kernel_initializer = paras_cnn['kernel_initializer']
        self.kernel_regularizer = paras_cnn['kernel_regularizer']
        self.bias_initializer = paras_cnn['bias_initializer']
        self.activation=activation
    
    def __call__(self,input_img,i,UpSampling):
        self.tail.debug("input_img.shape {0} i {1} UpSampling {2} attr {3}".format(input_img.shape,i,UpSampling,self.__dict__))
        x, p = None, None
        if self.input_shape[0] % 8 == 0:
            pad3 = 'same'
        else:
            pad3 = 'valid'
        
        if UpSampling:
            x, p = poolingSelection(input_img,self.pooling[i],UpSampling=UpSampling)

            if i == len(self.filters) - 2:
                x = Conv2DTranspose(self.filters[i], kernel_size=self.kernel_size[i+1], strides=self.strides[i+1], padding=pad3, activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='deconv_%d' % i)(x)
            else:
                x = Conv2DTranspose(self.filters[i], kernel_size=self.kernel_size[i+1], strides=self.strides[i+1], padding='same', activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='deconv_%d' % i)(x)

            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)
            
        else:
            if i == len(self.filters) - 1 and i == 0:
                x = Conv2D(self.filters[i], kernel_size=self.kernel_size[i], strides=self.strides[i], padding=pad3, activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='conv_%d' % i, input_shape=self.input_shape)(input_img)
            elif i == 0:
                x = Conv2D(self.filters[i], kernel_size=self.kernel_size[i], strides=self.strides[i], padding='same', activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='conv_%d' % i, input_shape=self.input_shape)(input_img)
            elif i == len(self.filters) - 1:
                x = Conv2D(self.filters[i], kernel_size=self.kernel_size[i], strides=self.strides[i], padding=pad3, activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='conv_%d' % i)(input_img)
            else:
                x = Conv2D(self.filters[i], kernel_size=self.kernel_size[i], strides=self.strides[i], padding='same', activation=None, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, bias_initializer=self.bias_initializer, name='conv_%d' % i)(input_img)

            x = BatchNormalization()(x)
            x = Activation(self.activation)(x)

            x, p = poolingSelection(x,self.pooling[i],UpSampling=UpSampling)
        self.tail.debug("x.shape {0} p {1} i {2}".format(x.shape,p,i))
        return x, p
        

# Depending poolingSelection, return pooling processed x and p (flattenLength element).
def poolingSelection(x,poolingSelection, UpSampling=False):
    """
    >>> poolingSelection(Input(shape=(576,576,1)),'no',False)
    (<tf.Tensor 'input_1:0' shape=(?, 576, 576, 1) dtype=float32>, 1)
    >>> poolingSelection(Input(shape=(576,576,1)),'avg',False)
    (<tf.Tensor 'average_pooling2d_1/AvgPool:0' shape=(?, 288, 288, 1) dtype=float32>, 2)
    >>> poolingSelection(Input(shape=(576,576,1)),'max',False)
    (<tf.Tensor 'max_pooling2d_1/MaxPool:0' shape=(?, 288, 288, 1) dtype=float32>, 2)
    >>> poolingSelection(Input(shape=(576,576,1)),'no',True)
    (<tf.Tensor 'input_4:0' shape=(?, 576, 576, 1) dtype=float32>, 1)
    >>> poolingSelection(Input(shape=(576,576,1)),'avg',True)
    (<tf.Tensor 'up_sampling2d_1/ResizeNearestNeighbor:0' shape=(?, 1152, 1152, 1) dtype=float32>, 2)
    >>> poolingSelection(Input(shape=(576,576,1)),'max',False)
    (<tf.Tensor 'max_pooling2d_2/MaxPool:0' shape=(?, 288, 288, 1) dtype=float32>, 2)
    """
    if UpSampling:
        if poolingSelection == 'no':
            return x, 1
        elif poolingSelection == 'max' or poolingSelection == 'avg':
            return UpSampling2D((2,2))(x), 2
        else:
            print('no UpSampling is selected.')
            return x, 1
    else:
        if poolingSelection == 'no':
            return x, 1
        elif poolingSelection == 'max':
            return MaxPooling2D(pool_size=(2,2))(x), 2
        elif poolingSelection == 'avg':
            return AveragePooling2D(pool_size=(2,2))(x), 2
        else:
            print('no pooling is selected.')
            return x, 1


@log_to('tail')
class AutoencoderMLP(Callable):
    '''
    Given dims,flattenLength,x,last_activation, provide autoencodered np.array and encoded np.array under certain settings.
    '''
    def __init__(self,
                 dropRate=None,
                 MLP=None,
                 Dense_=None):
        self.dropRate = dropRate
        self.MLP = MLP
        self.Dense_ = Dense_
    
    def __call__(self,dims,flattenLength,x,last_activation):
        self.tail.debug("dims {0} flattenLength {1} x {2} attr {3}".format(dims,flattenLength,x,self.__dict__))

        # internal layers in encoder
        encoded = self.MLP(dims,x,last_activation=last_activation,prefix_name='encoder',dropRate=self.dropRate)
        x = encoded
        
        # internal layers in decoder
        dims.reverse()
        print("dims.reverse() {0}",dims)
        self.tail.debug("dims.reverse() {0}".format(dims))
        if len(dims) == 1:
            x = self.MLP(dims,x,last_activation=last_activation,prefix_name='decoder',dropRate=0)
        else:
            x = self.MLP(dims[1:],x,last_activation=last_activation,prefix_name='decoder',dropRate=0)

        # output
        x = self.Dense_(flattenLength,activation=last_activation, name='decoder_%d' % len(dims))(x)

        return x, encoded


@log_to('tail')
class MLP(Callable):
    '''
    Given dims,x,last_activation,prefix_name,dropRate, provide
    multilayered-NN-processed np.array under certain settings.
    '''
    def __init__(self,
                 paras=None,
                 Dense_BN=None):
        self.paras=paras
        self.Dense_BN=Dense_BN

    def __call__(self,dims,x,last_activation,prefix_name,dropRate):
        n_layer = len(dims) - 1
        self.tail.debug("n_layer {0} dims {1} x {2} last_activation {3} dropRate {4} attr {5}".format(n_layer,dims,x,last_activation,dropRate,self.__dict__))
        for i in range(n_layer):
            x = self.Dense_BN(dims[i],x,prefix_name + '_%d' % i,dropRate)
        print('dims: ',dims)
        if len(dims) == 1:
            x = Dense(dims[-1],activation=last_activation,kernel_regularizer=self.paras['kernel_regularizer'],kernel_initializer=self.paras['kernel_initializer'],name=prefix_name + '_%d' % n_layer)(x)
        else:
            x = Dense(dims[-1],activation=last_activation,kernel_initializer=self.paras['kernel_initializer'],name=prefix_name + '_%d' % n_layer)(x)
        return x
        

@log_to('tail')
class Dense_BN(Callable):
    '''
    Given units,x,name,dropRate, provide np.array with
    BantchNormalization and activation under certain settings.
    '''
    def __init__(self,
                 activation=None,
                 Dense_=None):
        self.activation = activation
        self.Dense_ = Dense_
        
    def __call__(self,units,x,name,dropRate):
        self.tail.debug("units {0} x {1} name {2} attr {3}".format(units,x,name,self.__dict__))
        x = self.Dense_(units=units,activation=None,name=name)(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = Dropout(dropRate)(x)
        return x


@log_to('tail')
class Dense_(Callable):
    '''
    Given units,activation,name, provide Keras.layers.Dence function under paras settings.
    '''
    def __init__(self,paras=None):
        self.paras = paras
        
    def __call__(self,units,activation,name):
        self.tail.debug("units {0} activation {1} name {2} attr {3}".format(units,activation,name,self.__dict__))
        return Dense(units=units,
             activation=activations.get(activation),
             name = name,
             use_bias = self.paras['use_bias'],
             kernel_initializer = initializers.get(self.paras['kernel_initializer']),
             bias_initializer = initializers.get(self.paras['bias_initializer']),
             kernel_regularizer = regularizers.get(self.paras['kernel_regularizer']),
             bias_regularizer = regularizers.get(self.paras['bias_regularizer']),
             activity_regularizer = regularizers.get(self.paras['activity_regularizer']),
             kernel_constraint = constraints.get(self.paras['kernel_constraint']),
             bias_constraint = constraints.get(self.paras['bias_constraint']))


def create_callbacks(save_dir,patience,period,monitor='val_loss'):
    ''' Create callbacks as list to add to .fit '''
    estop = callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto',restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(save_dir + '/' + 'ep{epoch:03d}-loss{loss:.15f}-val_loss{val_loss:.15f}.h5',monitor=monitor, save_weights_only=False, save_best_only=True, period=period)
    tensorboard_callback = TensorBoard(log_dir='./Everglades/logs/' + os.path.basename(save_dir), histogram_freq=1, batch_size=512,write_graph=True, write_grads=False)

    return [estop,checkpoint,tensorboard_callback]


