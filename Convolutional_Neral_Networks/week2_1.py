# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """
    
    
    #Returns:
    #model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    """
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32,(7,7),strides = (1,1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2),name = 'mas_pool')(X)
    
    X = Flatten()(X)
    X = Dense(1,activation = 'sigmoid', name = 'fc')(X)
    model = Model(inputs = X_input, outputs = X, name = "Happymodel")
    
    ### END CODE HERE ###
    
    return model


### START CODE HERE ### (1 line)
model = HappyModel(X_train.shape[1:])
### END CODE HERE ###

### START CODE HERE ### (1 line)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
### END CODE HERE ###

### START CODE HERE ### (1 line)
model.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 128)
### END CODE HERE ###

### START CODE HERE ### (1 line)
preds = model.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))