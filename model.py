## Import necessary libraries
import numpy as np
import csv
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Import sci-kit and keras libraries and utilities
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


## Define flag variables:
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define flags to use, including flags to tweak hyper-parameters
flags.DEFINE_string('data_folder_path', './data/', "Location of data folder")
flags.DEFINE_string('pretrained_model', '', "Loading pretrained model")
flags.DEFINE_integer('epochs', 10, "Number of epochs")
flags.DEFINE_integer('batch_size', 36, "Size of each batch")


# Method to Load data ( both image paths and steering measurements)
def load_data():
    # lines in csv file
    lines = []
    with open( FLAGS.data_folder_path + 'driving_log.csv' ) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append( line )
        
        print( '>>> Number of data in file: {}' .format( len(lines) ))

    # Shuffle data and divide it into training set and validation set (80:20)
    shuffle( lines )
    lines_train, lines_valid = train_test_split( lines, test_size=0.2 )

    print( '===> Completed loading data.'); print( '>>> Size of Training set: ', len(lines_train), '\n>>> Size of Validation set:', len(lines_valid) );   
    return lines_train, lines_valid

# Generator for training. Creates data in batches as needed by train().
# Example source: https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
def data_generator(input_data, batch_size):

    # redefine batch size to account for augmented data
    batch_size = int(FLAGS.batch_size/3.0/2.0)  # 3 for three cameras, and 2 for flips. 

    while True:
        for start_idx in range(0, len( input_data ) , batch_size):
            lines = input_data[ start_idx : (start_idx + batch_size) ]

            images, measurements = [], []
            for line in lines:
                for i in range(3):
                    source_path = FLAGS.data_folder_path + 'IMG/' + line[i].split('/')[-1]
                    image = cv2.imread( source_path )
                    images.append(image); 
                    # plt.figure(); plt.imshow( image, cmap='gray' ); plt.savefig( './examples/image' + str(i) + '.png', transparent=True, bbox_inches='tight', pad_inches=0 )
                    
                    # Adding views from all camera ( taking correction factor into account )
                    correction_factor = 0.05
                    if ( i == 0 ):
                        measurement = float(line[3])
                    elif ( i == 1 ):
                        measurement = float(line[3]) + correction_factor
                    elif ( i == 2 ):
                        measurement = float(line[3]) - correction_factor
                    measurements.append(measurement)
                    
                    # Adding flipped version of image to overcome veering to one side
                    augmented_images, augmented_measurements = [], []
                    for image, measurement in zip(images, measurements):
                        augmented_images.append(image)
                        augmented_measurements.append(measurement)
                        augmented_images.append(cv2.flip(image,1))
                        augmented_measurements.append(measurement * -1.0)
                        
            # Convert to numpy array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield X_train, y_train
             

# Create a Model
# Source Nvidia model as presented in classroom
def create_model():
    drop_rate = 0.25

    model = Sequential()
    # Normalilzing image between -1.0 to +1.0
    model.add( Lambda( lambda x: x/127.5 - 1.0, input_shape=(160, 320, 3) ) )
    # Cropping image
    model.add( Cropping2D( cropping=((70,25), (0,0)) ))
    # LeNet based CNN with dropouts
    model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation="relu", init="normal" ) )
    model.add( Dropout( drop_rate ) )
    model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation="relu", init='normal' ) )
    model.add( Dropout( drop_rate ) )
    model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation="relu", init='normal' ) )
    model.add( Dropout( drop_rate ) )
    model.add( Convolution2D( 64, 3, 3, activation="relu", init='normal' ))
    model.add( Dropout( drop_rate ) )
    model.add( Convolution2D( 64, 3, 3, activation="relu", init='normal' ))
    model.add( Dropout( drop_rate ) )
    model.add( Flatten() )
    model.add( Dense(100, init='normal' ) )
    model.add( Dropout( drop_rate ) )
    model.add( Dense(50, init='normal' ) )
    model.add( Dropout( drop_rate ) )
    model.add( Dense(10, init='normal' ) )
    model.add( Dropout( drop_rate ) )
    model.add( Dense(1, init='normal') )

    return model


# Training method
def train():
    # Compile and train the model using the generator function
    generator_data_train = data_generator( data_train, batch_size=FLAGS.batch_size )
    generator_data_valid = data_generator( data_valid, batch_size=FLAGS.batch_size )

    # Create a new model or load a pretrained model
    if FLAGS.pretrained_model: 
        model = load_model(FLAGS.pretrained_model)
        print( '===> Loaded pretrained model from {} \n' .format(FLAGS.pretrained_model) )
    else:
        model = create_model()
        print( '===> Model created \n' )

    # Define training operation and its optimizers
    samples_per_batch = int(len(data_train)/FLAGS.batch_size)*FLAGS.batch_size
    model.compile( optimizer='adam', loss='mae' )
    model.fit_generator( generator_data_train, samples_per_epoch=samples_per_batch, validation_data=generator_data_valid, nb_val_samples=len(data_valid), nb_epoch=FLAGS.epochs )
    
    # Save the model
    saved_model_name = 'model.h5'
    model.save(saved_model_name)
    print( '===> Model saved as ' + saved_model_name + ' \n' )


# Load data and strat training process:
data_train, data_valid = load_data()
train()
