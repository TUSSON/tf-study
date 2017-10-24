
import numpy as np
import tensorflow as tf
import random

MAX_REPLAY_MEMORY = 50000
MINI_BATCH_SIZE = 32
GAMMA = 0.99
OBSERVE = 1000

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 80, 80, 4])

    # Convolutional Layer #1
    # Computes 32 features using a 8x8 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 80, 80, 4]
    # Output Tensor Shape: [batch_size, 20, 20, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[8, 8],
        strides=(4, 4),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 64 features using a 4,4 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 20, 20, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 64]
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[4, 4],
        strides=(2, 2),
        activation=tf.nn.relu)


    # Convolutional Layer #3
    # Computes 64 features using a 3,3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 9, 9, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        activation=tf.nn.relu)

    conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 512]
    dense = tf.layers.dense(inputs=conv3_flat, units=512, activation=tf.nn.relu)

    # Logits layer
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dense, units=2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=logits)

    action = tf.reshape(features["a"], [-1,2])
    qa = tf.reduce_sum(logits * action, axis=1)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(tf.square(labels - qa))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=qa)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_predict_input_fn(datasets):
    return tf.estimator.inputs.numpy_input_fn(
            x={"x": datasets},
            num_epochs=1,
            shuffle=False)

def get_train_input_fn(datasets, actions, labels, num_epochs=1, shuffle=False):
    return tf.estimator.inputs.numpy_input_fn(
            x={"x": datasets, "a": actions},
            y=labels,
            num_epochs=num_epochs,
            shuffle=shuffle)

def trainDeepQNet(frameStep):

    cnn_net = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                     model_dir="./saved_model")  
    
    status, reward, terminal =  frameStep(0)
    status = np.stack((status, status, status, status), axis=2)
    status = status.astype(np.float32)

    epsilon = 0.05
    t = 0

    D = []

    while True:
        input_fn = get_predict_input_fn(status[np.newaxis,:])
        try:
            predictions = cnn_net.predict(input_fn=input_fn)
            q_value = [p[1] for p in enumerate(predictions)][0]
        except ValueError:
            print('-------first train model-------')
            actions=np.zeros((1, 2), dtype=np.float32)
            labels=np.zeros((1,1), dtype=np.float32)
            train_input_fn = get_train_input_fn(status[np.newaxis,:], actions, labels)
            cnn_net.train(input_fn=train_input_fn, steps=1)
            q_value = labels

        #select an action
        #    with probability epsilon select a random action
        #    otherwise select Q.argmax(Q) 
        if random.random() < epsilon:
            action = random.randrange(2)
            print('--------random action----------')
        else:
            action = q_value.argmax()
            
        #carry out action
        new_status, reward, terminal = frameStep(action)
        new_status = new_status.astype(np.float16)
        new_status = new_status[:,:,np.newaxis]
        new_status = np.append(new_status, status[:,:,:3], axis=2)

        a = np.zeros(2, dtype=np.float32)
        a[action] = 1
        #store experience <status, action, reward, new_status> in replay memory D
        D.append((status, a, reward, new_status, terminal))

        if len(D) > MAX_REPLAY_MEMORY:
            D.pop(0)
        
        if t > OBSERVE:
            #sample random transitions from replay memory D
            exp_batch = random.sample(D, MINI_BATCH_SIZE)
            s_batch = np.asarray([exp[0] for exp in exp_batch])
            a_batch = np.asarray([exp[1] for exp in exp_batch])
            r_batch = np.asarray([exp[2] for exp in exp_batch], dtype=np.float32)
            news_batch = np.asarray([exp[3] for exp in exp_batch])
            t_batch = np.asarray([exp[4] for exp in exp_batch])

            #calculate target for each minibatch transition
            input_fn = get_predict_input_fn(news_batch)
            newq_batch = cnn_net.predict(input_fn=input_fn)
            newq_batch = np.asarray([p[1] for p in enumerate(newq_batch)])
            q_batch = np.empty(len(newq_batch), dtype=np.float32)

            for i in range(len(newq_batch)):
                terminal = t_batch[i]
                reward = r_batch[i]
                #if new_status is terminal state then tt=rr
                if terminal:
                    q_batch[i] = reward
                #otherwise tt=rr + y*max(Q(new_status))
                else:
                    q_batch[i] = reward + GAMMA*np.max(newq_batch[i])
            train_input_fn = get_train_input_fn(s_batch, a_batch, q_batch)
            cnn_net.train(input_fn=train_input_fn, steps=1)

        status = new_status 
        t += 1

        print(t, 'epsilon:', epsilon, 'action:',
              action, 'reward:', reward,
              'qmax:', np.max(q_value))
