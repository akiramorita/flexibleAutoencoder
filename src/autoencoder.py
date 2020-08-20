import sys,os, datetime, fire, csv
import logging, atexit, yaml
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

from keras import regularizers
from src.utils_tailHandler import log_to, tail
from src.utils_common import recordSetting
from src.utils_CNN_AE import AutoencoderConv2D, Generator, sample_X_valid, Conv2D_BN, AutoencoderMLP, MLP, Dense_, Dense_BN, Datagen, AutoencoderTrain, AutoencoderEval
from keras.layers import Input
from keras.regularizers import l2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@log_to('tail')
class Autoencoder:

        '''
        Create autoencoder model, and evaluate its performance with sample images
        Adjustable autoencoder structure with input parameters for better performance
        Inputs: images, CNN structure, MPL structure, and keras.layers.Dence parameters
        Outputs: autoencoder model, original and reproduced image samples
        '''
        def __init__(self,
                 preTrain_verbose=1,                            # verbose on/off
                 imageShape=[256,256],                          # image shape
                 train_epochs=100,                              # number of train epoch
                 batch_size=32,                                 # batch size
                 filters=[32,64,128,256],                       # CNN filter structure
                 pooling=['avg','avg','no','no'],               # CNN pooling structure
                 kernel_size=[5,5,3],                           # CNN kernel structure
                 strides=[2,2,2],                               # CNN stride structure
                 activation='elu',                              # CNN and MPL activation 
                 last_activation='selu',                        # CNN last layer activation
                 optimizer='adadelta',                          # optimizer
                 dropRate=0.5,                                  # CNN and MPL dropout rate
                 dims=[1000],                                   # MPL node and layer structure 
                 patience=100,                                  # epoch count for early stopping
                 period=100,                                    # epoch frequency for weight saving
                 use_bias=True,                                 # parameter for MPL keras.layers.Dense
                 kernel_initializer = 'glorot_uniform',         # parameter for MPL keras.layers.Dense
                 bias_initializer='zeros',                      # parameter for MPL keras.layers.Dense
                 kernel_regularizer=None,                       # parameter for MPL keras.layers.Dense
                 bias_regularizer=None,                         # parameter for MPL keras.layers.Dense
                 activity_regularizer=None,                     # parameter for MPL keras.layers.Dense
                 kernel_constraint=None,                        # parameter for MPL keras.layers.Dense
                 bias_constraint=None,                          # parameter for MPL keras.layers.Dense
                 cnn_kernel_initializer = 'glorot_uniform',     # parameter for CNN keras.layers.Dense
                 cnn_kernel_regularizer= None,                  # parameter for CNN keras.layers.Dense
                 cnn_bias_initializer = 'zeros',                # parameter for CNN keras.layers.Dense
                 AutoencoderMLP_switch=True,                    # MPL on/off
                 save_dir=None,                                 # output save directory
                 train_dir='data/train',                        # train dataset selection
                 valid_dir='data/valid',                        # valid dataset selection
                 save_prefix='AE'):                             # autoencoder name prefix
                self.startTime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                self.preTrain_verbose = preTrain_verbose
                self.imageShape = tuple(imageShape)
                self.train_epochs = train_epochs
                self.batch_size = batch_size
                self.filters=filters
                self.pooling=pooling
                self.kernel_size=kernel_size
                self.strides=strides
                self.cnn_kernel_initializer = cnn_kernel_initializer
                self.cnn_kernel_regularizer = cnn_kernel_regularizer
                self.cnn_bias_initializer = cnn_bias_initializer
                self.activation = activation
                self.last_activation = last_activation
                self.optimizer = optimizer
                self.dropRate=dropRate
                self.dims = dims
                self.patience = patience
                self.period = period
                self.AutoencoderMLP_switch = AutoencoderMLP_switch
                self.train_dir = train_dir
                self.valid_dir = valid_dir
                self.save_dir = save_dir
                self.save_prefix = save_prefix
                self.input_shape = (imageShape[0],imageShape[1],1)
                self.adjust()

                if self.save_dir is None:
                        self.save_dir = 'logs/ae_' + self.startTime
                if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)

                self.encoderLayerName = 'dropout_1'
                if self.AutoencoderMLP_switch: self.encoderLayerName = 'encoder_' + str(len(dims)-1)

                self.paras= {'use_bias':use_bias,
                        'kernel_initializer':kernel_initializer,
                        'bias_initializer':bias_initializer,
                        'kernel_regularizer':kernel_regularizer,
                        'bias_regularizer':bias_regularizer,
                        'activity_regularizer':activity_regularizer,
                        'kernel_constraint':kernel_constraint,
                        'bias_constraint':bias_constraint}
            
        def adjust(self):
                ''' Adjust the length of pooling, kernel_size and strides according to filter length '''
                for i in [self.pooling,self.kernel_size,self.strides]:
                        n =len(self.filters) - len(i)
                        if n > 0: i += [i[-1]]*n
                        else: i = i[:len(self.filters)]
            
                self.paras_cnn = {'filters':self.filters,
                          'pooling':self.pooling,
                          'kernel_size':self.kernel_size,
                          'strides':self.strides,
                          'kernel_initializer':self.cnn_kernel_initializer,
                          'kernel_regularizer':self.cnn_kernel_regularizer,
                          'bias_initializer': self.cnn_bias_initializer}
                self.tail.debug("paras_cnn {0}".format(self.paras_cnn))
    
        def run(self):
                ''' 
                Construct CNN and MPL structure in the following section
                according to filters, pooling, kernel_size, strides, and dims.
                '''
                Dense__ = Dense_(paras=self.paras)
                Dense_BN_ = Dense_BN(
                        activation = self.activation,
                        Dense_  = Dense__)
                MLP_ = MLP(
                        paras=self.paras,
                        Dense_BN = Dense_BN_)
                AutoencoderMLP_ = AutoencoderMLP(
                        dropRate=self.dropRate,
                        MLP = MLP_,
                        Dense_ = Dense__)
                Conv2D_BN_ = Conv2D_BN(
                        input_shape=self.input_shape,
                        paras_cnn=self.paras_cnn,
                        activation=self.activation)
                AutoencoderConv2D_ = AutoencoderConv2D(
                        paras=self.paras,
                        input_shape=self.input_shape,
                        paras_cnn=self.paras_cnn,
                        last_activation=self.last_activation,
                        Conv2D_BN=Conv2D_BN_,
                        AutoencoderMLP=AutoencoderMLP_,
                        AutoencoderMLP_switch=self.AutoencoderMLP_switch,
                        dropRate=self.dropRate)

                input_img = Input(shape=self.input_shape)
                self.autoencoder, self.encoder, self.flattenLength = AutoencoderConv2D_(input_img,self.dims)
                print(self.autoencoder.summary())
                print(self.encoder.summary())

                ''' Prepare generator to feed data to model: no need to load all of data once '''
                Datagen_ = Datagen()
                Generator_ = Generator(
                        'autoencoder',
                        datagen=Datagen_,
                        save_dir=None,
                        save_prefix=None,
                        cropShape=self.imageShape,
                        batch_size=self.batch_size)
                train_generator = Generator_(self.train_dir)
                valid_generator = Generator_(self.valid_dir)
                
                ''' Prepare sample data for performance evaluation '''
                X_train = sample_X_valid(self.imageShape,self.train_dir,seed=1,n_sample=10)
                X_valid = sample_X_valid(self.imageShape,self.valid_dir,seed=1,n_sample=10)

                ''' Construct autoencoder for sample data '''
                AutoencoderEvals = [AutoencoderEval('train', valid_generator=train_generator,sample_X_valid=X_train),
                                AutoencoderEval('valid', valid_generator=valid_generator,sample_X_valid=X_valid)]

                ''' Construct autoencoder for training '''
                AutoencoderTrain_ = AutoencoderTrain(
                        train_generator = train_generator,
                        valid_generator = valid_generator,
                        train_epochs = self.train_epochs,
                        patience = self.patience,
                        verbose = self.preTrain_verbose,
                        period = self.period,
                        save_dir = self.save_dir,
                        AutoencoderEvals = AutoencoderEvals)
                
                ''' Train autoencoder '''
                dic1 = AutoencoderTrain_(
                        autoencoder=self.autoencoder,
                        optimizer=self.optimizer,
                        encoderLayerName=self.encoderLayerName)

                with open('logs/aescore.csv','a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.datetime.now().strftime("%Y%m%d_%H%M"),os.path.basename(self.save_dir),dic1['train_score'][0],dic1['valid_score'][0],dic1['train_score'][1],dic1['valid_score'][1]])

                self.endTime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                recordSetting(self,self.save_dir)
                return dic1


if __name__ == "__main__":
    logging.config.dictConfig(yaml.load(tail))
    atexit.register(logging.shutdown)
    try:
        fire.Fire(Autoencoder)
    except Exception as e:
        logging.exception( e )
        status= 2

