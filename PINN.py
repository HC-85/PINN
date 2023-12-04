import tensorflow as tf
import numpy as np

def identity_like(shape):
    M = min(shape)
    N = max(shape)
    den = N-M+1
    A = np.triu(np.ones(shape = (M, N)))
    B = np.tril(np.ones(shape = (M, N)), k = N-M)
    K = np.multiply(A, B)/den
    if shape[0]>shape[1]:
        K = np.transpose(K)
    return tf.Variable(K, dtype=tf.float32)


class IdentityLike(tf.keras.initializers.Initializer):    
    def __call__(self, shape, dtype=None):
        return identity_like(shape)


class OuterCoder(tf.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = [] 
        for layer_size in layer_sizes[:-1]:
            self.layers.append(tf.keras.layers.Dense(layer_size, 
                activation = tf.nn.relu, 
                kernel_initializer='he_uniform', 
                kernel_regularizer=tf.keras.regularizers.L2))
                
        self.layers.append(tf.keras.layers.Dense(layer_size, 
                activation = None, 
                kernel_initializer = 'he_uniform', 
                kernel_regularizer=tf.keras.regularizers.L2))


    def __call__(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
        y = x + self.layers[-1](y)
        self.out = y
        return y


class InnerCoder(tf.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = []
        identity_like_initializer = IdentityLike()
        for i in range(len(layer_sizes)):
            self.layers.append(tf.keras.layers.Dense(layer_sizes[i], 
                activation = None, 
                kernel_initializer = identity_like_initializer,
                kernel_regularizer=tf.keras.regularizers.L2))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x


class Koopman(tf.Module):
    def __init__(self, size):
        super().__init__()
        self.op = tf.Variable(tf.linalg.diag(tf.ones((size,))))
        
    def __call__(self, x, steps):
        evolution = tf.TensorArray(dtype=tf.float32, size = steps+1, dynamic_size=False)
        evolution = evolution.write(0, x)
        for i in range(steps):
            x = tf.linalg.matmul(x, self.op)
            evolution = evolution.write(i, x)
        evolution = evolution.stack()
        self.out = evolution
        return x


class PINN(tf.Module):
    def __init__(self, oe_ls, ie_ls, ks, id_ls, od_ls):
        super().__init__()
        self.OuterEncoder = OuterCoder(oe_ls)
        self.InnerEncoder = InnerCoder(ie_ls)
        self.KoopmanOp = Koopman(ks)
        self.InnerDecoder = InnerCoder(id_ls)
        self.OuterDecoder = OuterCoder(od_ls)
    
    def __call__(self, burger_wave, steps):
        # Cole-Hopf
        heat_wave = self.OuterEncoder(burger_wave)
        # FT
        trans_heat_wave = self.InnerEncoder(heat_wave)
        # Evolution
        evolved_trans_heat_wave = self.KoopmanOp(trans_heat_wave, steps = steps)
        # IFT
        evolved_heat_wave = self.InnerDecoder(evolved_trans_heat_wave)
        # Inverse Cole-Hopf
        evolved_burger_wave = self.OuterDecoder(evolved_heat_wave)
        
        
        return evolved_burger_wave


    def calculate_loss(self, states, targets):    
        reshaped_states = tf.reshape(states, shape = (-1, states.shape[-1]))
        reshaped_target = tf.reshape(targets, shape = (-1, targets.shape[-1]))

        oe_states = self.OuterEncoder(reshaped_states)
        ie_oe_states = self.InnerEncoder(oe_states)
        k_ie_oe_states = self.KoopmanOp(ie_oe_states, 1)
        id_k_ie_oe_states = self.InnerDecoder(k_ie_oe_states)
        od_id_k_ie_oe_states = self.OuterDecoder(id_k_ie_oe_states)
        id_ie_oe_states = self.InnerDecoder(ie_oe_states)
        od_oe_states = self.OuterDecoder(oe_states)
        od_id_ie_oe_states = self.OuterDecoder(id_ie_oe_states)
        
        oe_target = self.OuterEncoder(reshaped_target)
        ie_oe_target = self.InnerEncoder(oe_target)

        ae_loss = tf.reduce_sum(tf.square(reshaped_states - od_id_ie_oe_states))/tf.reduce_sum(tf.square(reshaped_states))            
        oae_loss = tf.reduce_sum(tf.square(reshaped_states - od_oe_states))/tf.reduce_sum(tf.square(reshaped_states))
        iae_loss = tf.reduce_sum(tf.square(oe_states - id_ie_oe_states))/tf.reduce_sum(tf.square(oe_states))
        p_loss = tf.reduce_sum(tf.square(reshaped_target - od_id_k_ie_oe_states))/tf.reduce_sum(tf.square(reshaped_target))
        l_loss = tf.reduce_sum(tf.square(ie_oe_target - k_ie_oe_states))/tf.reduce_sum(tf.square(ie_oe_target))

        total_loss = ae_loss + oae_loss + iae_loss + p_loss + l_loss

        return total_loss/len(states)

    def evaluate(self, test_set):
        heat_wave = self.OuterEncoder(burger_wave)
        trans_heat_wave = self.InnerEncoder(heat_wave)
        evolved_trans_heat_wave = self.KoopmanOp(trans_heat_wave, steps = steps)
        evolved_heat_wave = self.InnerDecoder(evolved_trans_heat_wave)
        evolved_burger_wave = self.OuterDecoder(evolved_heat_wave)
        
        return evolved_burger_wave
    

@tf.function
def training_loop(model, epochs, dataset, batch_size, optimizer):
    print('Graph traced')
    size = int((len(dataset)//batch_size)*epochs)
    loss_hist = tf.TensorArray(dtype=tf.float32, size = size, dynamic_size=False)
    for _epoch in tf.range(epochs, dtype = tf.int64):
        epoch_loss = 0.0
        batched_dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        for i, (batch_states, batch_targets) in enumerate(batched_dataset):
            with tf.GradientTape() as tape:
                loss = model.calculate_loss(batch_states, batch_targets)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))            
            
            epoch_loss += loss
            loss_hist = loss_hist.write(tf.cast(len(dataset)*_epoch + i, tf.int32), loss)

    return loss_hist.stack()