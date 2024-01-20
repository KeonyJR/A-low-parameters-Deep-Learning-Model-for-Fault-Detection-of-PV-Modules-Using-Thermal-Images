from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Concatenate, MaxPooling2D, \
                                    UpSampling2D, SpatialDropout2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

class HDUnet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.spatialdroput=0.15
        self.droput=0.1
        self.alpha=0.1
        self.padding='same'
        self.activation_conv='relu'
        self.activation_dense='softmax' # 'softmax'(11 and 12 classes) /'sigmoid' (2 classes problem)
        self.neurons_dense_layer=32
        self.filters_resolution_level_1=32
        self.filters_resolution_level_2=48
        self.filters_resolution_level_3=64
        self.filters_downsampling_level_2=14
        self.filters_downsampling_level_3=18
        self.filters_upsampling_level_1=10
        self.filters_upsampling_level_2=14
        self.filters_final_upsampling=62
        self.filters_output=16
        self.layers_dense_block=6
        self.growth_rate_resolution_level_1=6
        self.growth_rate_resolution_level_2=9
        self.growth_rate_resolution_level_3=12
        self.strides_downsampling_one_level=2
        self.strides_downsampling_two_levels=4
        self.strides_input=1
        self.maxpooling_pool_size=(2, 2)
        self.upsampling_size_up_two_levels=(4, 4)
        self.upsampling_size_up_one_level=(2, 2)
        self.kernel_size_input_layer=(3, 3)
        self.kernel_size_dense_layer_part_1=(1, 1)
        self.kernel_size_dense_layer_part_2=(3, 3)
        self.kernel_size_downsampling_layer=(3, 3)
        self.kernel_size_upsampling_layer=(1, 1)
        self.kernel_size_output_layer=(1, 1)

    def dense_layer(self, inputs, growth_rate):
        x = Conv2D(growth_rate, kernel_size=self.kernel_size_dense_layer_part_1, activation= self.activation_conv, 
                   padding=self.padding)(inputs)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)
        y = Conv2D(growth_rate, kernel_size=self.kernel_size_dense_layer_part_2, activation= self.activation_conv, 
                   padding=self.padding)(x)
        y = LeakyReLU(alpha=self.alpha)(y)
        y = BatchNormalization()(y)
        z = Concatenate()([x, y])
        return z

    def input_block(self, input_tensor, filters ):
        x = Conv2D(filters=filters, activation= self.activation_conv, kernel_size=self.kernel_size_input_layer, 
                   strides=self.strides_input, padding=self.padding)(input_tensor)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)
        return x

    def downsampling(self, input_tensor, filters, strides):
        x = Conv2D(filters, kernel_size=self.kernel_size_downsampling_layer, strides=strides, 
                   activation= self.activation_conv, padding=self.padding)(input_tensor)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)
        return x

    def upsampling(self, input_tensor, filters, size):
        x = UpSampling2D(size=size)(input_tensor)
        x = Conv2D(filters, kernel_size=self.kernel_size_upsampling_layer, activation= self.activation_conv, 
                   padding=self.padding)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)
        return x

    def output_block(self, input_tensor):
        x = Conv2D(filters=self.filters_output, kernel_size=self.kernel_size_output_layer, 
                   activation= self.activation_conv, padding=self.padding)(input_tensor)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(self.neurons_dense_layer, activation= self.activation_conv)(x)
        x = Dropout(self.droput)(x)
        output = Dense(self.num_classes, activation=self.activation_dense)(x)
        return output

    def dense_block(self,input_tensor, initial_filters, growth_rate,layers):
        x_c = self.dense_layer(input_tensor, initial_filters)
        for _ in range(layers-1):
            x = self.dense_layer(x_c, growth_rate)
            x_c=Concatenate()([x, x_c])
        return x_c

    def hdunet(self):
        # ------------Encoder-------------
        # Encoder Part 1
        inputs = Input(shape=self.input_shape)
        rl1 = self.input_block(inputs, self.filters_resolution_level_1)
        rl1  = self.input_block(rl1 , self.filters_resolution_level_1)
        # Dense Block- Resolution level 1 # part 1
        rl1_db1=self.dense_block(rl1 , self.filters_resolution_level_1, self.growth_rate_resolution_level_1,
                                 self.layers_dense_block)
        rl1_db1 = SpatialDropout2D(self.spatialdroput)(rl1_db1)
        # Encoder Part 2
        rl2 = MaxPooling2D(self.maxpooling_pool_size)(rl1)
        rl2 = self.dense_layer(rl2,self.filters_resolution_level_2)
        rl2 = self.dense_layer(rl2,self.filters_resolution_level_2)  #64 
        # Dense Block- Resolution level 2 # part 1
        rl2_db1=self.dense_block(rl2, self.filters_resolution_level_2, self.growth_rate_resolution_level_2,
                                 self.layers_dense_block)
        rl2_db1 = SpatialDropout2D(self.spatialdroput)(rl2_db1)
        # Encoder Part 3
        rl3 = MaxPooling2D(self.maxpooling_pool_size)(rl2)
        rl3 = self.dense_layer(rl3,self.filters_resolution_level_3)
        rl3 = self.dense_layer(rl3,self.filters_resolution_level_3)  #128 
        # Dense Block- Resolution level 3 # part 1
        rl3_db1=self.dense_block(rl3,self.filters_resolution_level_3, self.growth_rate_resolution_level_3,
                                 self.layers_dense_block)
        rl3_db1 = SpatialDropout2D(self.spatialdroput)(rl3_db1)
        # Dense Block- Resolution level 3 # part 2
        dp23=self.downsampling(rl2_db1,self.filters_downsampling_level_3,self.strides_downsampling_one_level)
        dp13=self.downsampling(rl1_db1,self.filters_downsampling_level_3,self.strides_downsampling_two_levels)
        c1=Concatenate()([rl3_db1,dp13,dp23])
        rl3_db2=self.dense_block(c1, self.filters_resolution_level_3,self.growth_rate_resolution_level_3,
                                 self.layers_dense_block)
        rl3_db2 = SpatialDropout2D(self.spatialdroput)(rl3_db2)
        # Dense Block- Resolution level 2 # part 2
        dp12=self.downsampling(rl1_db1,self.filters_downsampling_level_2, self.strides_downsampling_one_level)
        up32= self.upsampling(rl3_db1, self.filters_upsampling_level_2, self.upsampling_size_up_one_level)
        c2=Concatenate()([rl2_db1, up32 , dp12])
        rl2_db2=self.dense_block(c2, self.filters_resolution_level_2,self.growth_rate_resolution_level_2,
                                 self.layers_dense_block)
        rl2_db2 = SpatialDropout2D(self.spatialdroput)(rl2_db2)
        # Dense Block- Resolution level 1 # part 2
        up31= self.upsampling(rl3_db1, self.filters_upsampling_level_1, self.upsampling_size_up_two_levels)
        up21= self.upsampling(rl2_db1, self.filters_upsampling_level_1, self.upsampling_size_up_one_level)
        c3=Concatenate()([rl1_db1,up21,up31])
        rl1_db2=self.dense_block(c3, self.filters_resolution_level_1,self.growth_rate_resolution_level_1,
                                 self.layers_dense_block)
        rl1_db2 = SpatialDropout2D(self.spatialdroput)(rl1_db2)

        # ------------Decoder------------
        Up1= self.upsampling(rl2_db2, self.filters_final_upsampling, self.upsampling_size_up_one_level)
        Up2= self.upsampling(rl3_db2, self.filters_final_upsampling, self.upsampling_size_up_two_levels)
        c_4 = Concatenate()([rl1_db2, Up1,Up2])
        rl1 = self.dense_layer(c_4,self.filters_resolution_level_1)
        rl1 = self.dense_layer(rl1,self.filters_resolution_level_1)  

        # Output
        output = self.output_block(rl1)
        
        return Model(inputs=inputs, outputs=output)

# Example of usage
input_shape = (40, 24, 1)  # Adjust dimensions 
num_classes = 11  # Adjust the number of classes
model_instance = HDUnet(input_shape, num_classes)
dunet_model = model_instance.hdunet()
dunet_model.summary()