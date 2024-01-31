# Lung_Histopathology_Detection

Early detection of lung histopathology has become crucial and essential for humans. Rapid recognition gives many patients the greatest chance of re-covery. Histopathological graphics of biopsy samples tissue from possibly infected areas of the lungs are used by doctors to best solution. The multiple type of lung disease is frequently misdiagnosed and prolonged to detect. The characteristics used to detect Lung Histopathology are extracted from Computed Tomography (CT scan) images. DL is a novel method that ena-bles us to improve result's precision.  In this journal, we create DL model to determine the type of lung cancers from Computed Tomography images. Convolutional Neural Networks (CNN) which recognize & categorize lung-cancer type within improved efficiency & less amount of time, that is crucial to deciding on the best treatment approach for clients and their risk of mor-tality. This study proposes, a tri-category classification was applied to imag-es of lung cancer (BENIGN, ADENOCARCINOMA & SQUAMOUS CELL CARCINOMA) is performed utilising VL, VGG-16, and Le-Net to process an image of lung tissue and obtain functionalities effective for diagnostic techniques. Furthermore, one of the finest perspectives we have collaborat-ed on in this analysis is extracting handcrafted characteristics from raw pic-tures after various processing. For better classification, we are planning to implement the CNN Architecture. Eventually, the Python framework is uti-lized to deploy the model (Django). The goal of using this technique is to ob-tain some characteristics that are mainly accountable for lung histopatholo-gy forecasting.

PRETRAINED MODELS:
Using the lung histopathology dataset, we used the considerations models for lung cancer detection and classification.

VGG-16
    The VGG model, which exists for the Visual Geometry Group, is a well-known CNN architecture that demonstrated that it was possible to achieve efficient image recognition using a deep network and small convolutional filters. It is regarded as one of the finest vi-sion model architectures available today. The most distinctive aspect of VGG16 is that it focused upon using convolution layers of a 3x3 filter with stride 1 instead of a lot of hyper-parameters and consistently uses the same padding & maxpooling layer of a 2x2 filter with stride 2. The architecture keeps the layers of convolutional and max pooling in this ar-rangement. Two FC as well as a softmax are included as its final features. VGG16 contains 16 weighted layers, as indicated by the number 16. The network has 138 million (estimated) parameters, making is fairly huge.

Step by step implementation of VGG-16:

Step 1: START

Step 2: import all libraries

             INPUT:
             
                 Data Augmentation: ImageDataGenerator ()
                 
                 Parameter: rescale-1. /255, shear_range-0.2, zoom_range-0.2, 
                 horizontal_flip- True
                 
Step 3: Dataset

                 training_set ← train_datagen
                 
                 test_set ← test_datagen
                 
              Parameter: target_size(224,244), batch_size-32, class_mode-categorical
              
Step 4: to building a sequential model,

             model←Sequential()
             
             Layers: C1: Conv1_2, 64 depth, 3x3
             
                                 pool_size:2x2
                                 
                          C2: Conv2_2, 128 depth, 3x3
                          
                                 pool_size:2x2
                                 
                          C3: Conv3_3, 256 depth, 3x3
                          
                                 pool_size:2x2
                                 
                          C4: Conv4_3, 512 depth, 3x3
                          
                                 pool_size:2x2
                                 
                          C5: Conv5_3, 512 depth, 3x3
                          
                                 pool_size:2x2
                                 
             Add ReLu activation to every layer to prevent all negative values from being
              passed onto subsequent layers.
              
Step 5:  send data to the fully connected layer in order to flatten the vector,

                             F1: 256→relu
                             
                             F2: 128→relu
                             
                             F3: 3→sigmoid
                             
Step 6: to compile the model,

               Parameter: optimizer→SGD, loss→categorical_crossentropy, 
                                  metrics→accuracy, loss, precision, recall
                                  
Step 7: to create a ModelCheckpoint(),

                   mc←ModelCheckpoint()
                   
               Parameter: monitor, verbose
               
Step 8: to fit the model,

                   history← model.fit(steps_per_epoch-training_set.samples, 
                                    validation_steps-test_set.samples)
                                    
Step 9: observe train/test Accuracy, Loss, Precision & Recall using matplotlib

Step 10: OUTPUT

           end


LENET:

LeNet is an abbreviation for LeNet-5, which is a simple CNN that can respond to a subset of the neighbouring cells in the coverage areas and accomplish well in large-scale image analysis. Lenet-5 was one of the first pre-trained models, and its popularity stemmed from its simple and straightforward architectural style. It is an image classification multi-layer convolution neural network. The network is named Lenet-5 because it has five layers with learnable parameters. The LeNet-5 CNN architecture has seven layers. Three convolutional layers, two subsampling layers, and two fully connected layers make up the layer composi-tion.

Step by step implementation of LENET:

Step 1: START

Step 2: import all libraries

             INPUT:
             
                 Data Augmentation: ImageDataGenerator ()
                 
                 Parameter: rescale-1. /255, shear_range-0.2, zoom_range-0.2, horizontal_flip- True
                 
Step 3: Dataset

                 training_set ← train_datagen
                 
                 test_set ← test_datagen
                 
              Parameter: target_size(224,244), batch_size-32, class_mode-categorical
              
Step 4: to building a sequential model,

             Classifier←Sequential()
             
             Layers: C1: Conv1_1, 32 depth, 3x3
             
                                 pool_size:2x2
                                 
                          C2: Conv2_1, 128 depth, 3x3
                          
                                 pool_size:2x2
             Add ReLu activation to every layer to prevent all negative values from being passed onto subsequent layers.
             
Step 5:  send data to the fully connected layer in order to flatten the vector,

                             F1: 256→relu
                             
                             F2: 3→softmax
                             
Step 6: to compile the model,

               Parameter: optimizer→Adam,	loss→categorical_crossentropy, 
                                     metrics→accuracy, loss, precision, recall
                                     
                              Classifier←summary ()
                              
Step 7: to create a ModelCheckpoint(),

                   callbacks←ModelCheckpoint()
                   
               Parameter: model_path, monitor, verbose
               
Step 8: to fit the model,

                   history← Classifier.fit(steps_per_epoch-training_set.samples, 
                                    validation_steps-test_set.samples)
                                    
Step 9: observe train/test Accuracy, Loss, Precision & Recall using matplotlib

Step 10: OUTPUT

              End


Implementation of VL architecture:
      The proposed model is built on the Jupyter Notebook platform, and it makes use of the Python libraries Tensorflow and Keras. Implementation of proposed architecture Im-age_details function is used as data analysis that contains image_count, min_width, min_height, max_width, max_height. All these arguments show the max width & height, min width & height and also total number of images in the dataset. Plot_image is used to plot the images which is related to the Image_details function. Image data generator includes all possible orientation of the image i.e., rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True.
   The manual architecture is summarized as follows:
The classifier is the name of the architecture and here the model is a sequential model which allows us to create models layer by layer in sequential order. The first convolution layer with kernel_size=3, fliter=32, input_shape=250, channels=3, acyivaion=relu, max-pooling=(2,2), batch_size=32, Flatten is used to flatten the dimensions of the image ob-tained after convolving it. Dense=3, activation=softmax. To compile the model the argu-ments are optimizer=adam, loss=categorical_crossentropy.

Step by step implementation of VL architecture:

Step 1: START

Step 2: import all libraries

             INPUT:
             
                 def Images_details_print_data ()
                 
              Printing each key & values
              
                     For k, v in data.items(): 
                     
                     Print (k,v)
                     
Step 3: def Images_details(path)

                 files← take all images store in var 
                 
                 data={} // empty dictionary 
                 
            which contains image_count, max_width, max_height, min_width, min_height
            
                 for f in files
                 
                    im←to open the image
                    
Step 4: def plot_images() 

                    //prints 10 images of each category along with image_count, min_width, max_width, min_height, max_height
                    
Step 5:  Data Augmentation: ImageDataGenerator ()

               Parameter: rescale-1. /255, shear_range-0.2, zoom_range-0.2, horizontal_flip- True
               
Step 6: Dataset

                 training_set ← train_datagen
                 
                 test_set ← test_datagen
                 
              Parameter: target_size(224,244), batch_size-32, class_mode-categorical
              
Step 7: to building a sequential model,

             Classifier←Sequential()
             
             Layers: C1: Conv1_1, 32 depth, 3x3
             
                                 pool_size:2x2
                                 
             Add ReLu activation to the conv layer to prevent all negative values from being passed onto subsequent layers.
             
Step 8:  send data to the fully connected layer in order to flatten the vector,

                             F1: 38→relu
                             
                             F2: 3→softmax
                             
Step 9: to compile the model,

               Parameter: optimizer→Adam, loss→categorical_crossentropy, 
               
                                  metrics→accuracy, loss, precision, recall
                                  
                     Classifier←summary()
Step 10: to create a ModelCheckpoint(),

                      mc←ModelCheckpoint()
                      
               Parameter: model_path, monitor, verbose
               
Step 11: to fit the model,

                   history← Classifier.fit(steps_per_epoch-training_set.samples, 
                                    validation_steps-test_set.samples)
                                    
Step 12: observe train/test Accuracy, Loss, Precision & Recall using matplotlib

Step 13: OUTPUT

           end

DISCUSSION ON FINDINGS:

tensorflow: TensorFlow is a Google accessible library primarily designed for applications in deep learning. TensorFlow was originally designed for large numerical calculations rather than deep learning. Even so, it was also very helpful for deep learning development. Tensors are multi-dimensional arrays of extra dimensionality that TensorFlow accepts. 
syntax:pip install tensorflow==2.6.0

keras: An advanced deep learning API that incorporating NN is Google's Keras. It was created by utilising Python to write minimal NN framework. Moreover, it entails permitting the computing of several backend neural network.
syntax:pip install keras==2.6.0

matplotlib:	A popular Python package for visualisation methods is called Matplotlib. It is a platform-agnostic tool for 2D plotting array data. Python is used to create Matplotlib, which makes use of NumPy, Python's extensions for numerical mathematics.

CONCLUSION:
It concentrated on how an image from a testing set (trained dataset) and a previous data set were used to estimate the sequence of Lung Histopathology using a CNN model. This provides some of the succeeding insights into the prediction of Lung Histopathology. The ability to automatically classify images is the primary advantage of the CNN classification framework. In future, will deploy the cloud platform and also create a mobile based appli-cation. The Lung Histopathology primarily contributes to face misshape and is frequently irreversible because patients are diagnosed with the diseases too late. We have discussed an overview of methodologies for detecting abnormalities in Lung Histopathology images in this study, which includes the collection of Lung Histopathology image data sets, pre-processing methodologies, feature extraction techniques, and classification schemes. In future, will deploy the cloud platform and also create a mobile based application. 
