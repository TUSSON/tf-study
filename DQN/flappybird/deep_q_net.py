
import numpy as np
import tensorflow as tf
import random

MAX_REPLAY_MEMORY = 50000
MINI_BATCH_SIZE = 32
GAMMA = 0.99
OBSERVE = 1000

def inference():
    """Model function for CNN."""
    s = tf.placeholder(tf.float32, shape=(None, 80, 80, 4))

    # Convolutional Layer #1
    # Computes 32 features using a 8x8 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 80, 80, 4]
    # Output Tensor Shape: [batch_size, 20, 20, 32]
    conv1 = tf.layers.conv2d(
        inputs=s,
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

    return s, logits

def netloss(logits):
    a = tf.placeholder(tf.float32, shape=(None, 2))
    q = tf.placeholder(tf.float32, shape=(None))
    qa = tf.reduce_sum(logits * a, axis=1)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(tf.square(q - qa))
    return a, q, loss

def training(loss):
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return train_op

def trainDeepQNet(frameStep):
    s, logits = inference()
    a, q, loss = netloss(logits)
    train_op = training(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    status, reward, terminal =  frameStep(0)
    status = np.stack((status, status, status, status), axis=2)
    status = status.astype(np.float32)

    epsilon = 0.5
    t = 0

    D = []

    while True:
        q_value = sess.run(logits, feed_dict={s: [status]})[0]

        #select an action
        #    with probability epsilon select a random action
        #    otherwise select Q.argmax(Q) 
        if random.random() < epsilon:
            action = random.randrange(2)
            print('--------random action----------')
        else:
            action = q_value.argmax()

        if epsilon > 0.001:
            if t % 1000 == 0:
                epsilon = epsilon * 0.9
            
        #carry out action
        new_status, reward, terminal = frameStep(action)
        new_status = new_status.astype(np.float16)
        new_status = new_status[:,:,np.newaxis]
        new_status = np.append(new_status, status[:,:,:3], axis=2)

        a_t = np.zeros([2], dtype=np.float32)
        a_t[action] = 1
        #store experience <status, action, reward, new_status> in replay memory D
        D.append((status, a_t, reward, new_status, terminal))

        if len(D) > MAX_REPLAY_MEMORY:
            D.pop(0)
        
        if t > OBSERVE:
            #sample random transitions from replay memory D
            exp_batch = random.sample(D, MINI_BATCH_SIZE)
            s_batch = [exp[0] for exp in exp_batch]
            a_batch = [exp[1] for exp in exp_batch]
            r_batch = [exp[2] for exp in exp_batch]
            news_batch = [exp[3] for exp in exp_batch]
            t_batch = [exp[4] for exp in exp_batch]

            #calculate target for each minibatch transition
            newq_batch = sess.run(logits, feed_dict={s: news_batch})
            q_batch = []

            for i in range(len(newq_batch)):
                terminal = t_batch[i]
                reward = r_batch[i]
                #if new_status is terminal state then tt=rr
                if terminal:
                    q_batch.append(reward)
                #otherwise tt=rr + y*max(Q(new_status))
                else:
                    q_batch.append(reward + GAMMA*np.max(newq_batch[i]))
            sess.run(train_op, feed_dict={s: s_batch, a:a_batch, q:q_batch})

        status = new_status 
        t += 1

        if t % 1000 == 0:
            saver.save(sess, 'saved_model/bird-dqn', global_step = t)

        print(t, 'epsilon:', epsilon, 'action:',
              action, 'reward:', reward,
              'qmax:', np.max(q_value))
