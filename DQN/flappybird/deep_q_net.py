
import numpy as np
import tensorflow as tf
import random

MAX_REPLAY_MEMORY = 50000
MINI_BATCH_SIZE = 32
GAMMA = 0.99
OBSERVE = 10000

def inference():
    """Model function for CNN."""
    s = tf.placeholder(tf.float32, shape=(None, 40, 40, 4))

    # Convolutional Layer #1
    # Computes 32 features using a 4x4 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 40, 40, 4]
    # Output Tensor Shape: [batch_size, 20, 20, 32]
    conv1 = tf.layers.conv2d(
        inputs=s,
        filters=32,
        kernel_size=[4, 4],
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 64 features using a 4,4 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 20, 20, 32]
    # Output Tensor Shape: [batch_size, 9, 9, 32]
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[4, 4],
        strides=(2, 2),
        activation=tf.nn.relu)

    conv3_flat = tf.reshape(conv2, [-1, 9 * 9 * 32])

    # Convolutional Layer #3
    # Computes 64 features using a 3,3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 9, 9, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    #conv3 = tf.layers.conv2d(
    #    inputs=conv2,
    #    filters=64,
    #    kernel_size=[3, 3],
    #    activation=tf.nn.relu)

    #conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 64])

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
    optimizer = tf.train.AdamOptimizer(1e-4)
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

    checkpoint = tf.train.get_checkpoint_state("saved_model")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    status, reward, terminal =  frameStep(0)
    status = status.astype(np.float32)
    status = np.stack((status, status, status, status), axis=2)

    epsilon = 0.001
    t = 0

    D = []
    reset_status = False

    try_times = 0

    while True:
        q_value = sess.run(logits, feed_dict={s: [status]})[0]

        #select an action
        #    with probability epsilon select a random action
        #    otherwise select Q.argmax(Q) 
        if random.random() < epsilon:
            action = 0
            if random.random() < 0.1:
                action = 1
            print('--------random action----------')
        else:
            action = q_value.argmax()

        #carry out action
        new_status, reward, terminal = frameStep(action)
        new_status = new_status.astype(np.float32)
        if reset_status:
            new_status = np.stack((new_status, new_status, new_status, new_status), axis=2)
        else:
            new_status = new_status[:,:,np.newaxis]
            new_status = np.append(new_status, status[:,:,:3], axis=2)

        reset_status = terminal

        if terminal:
            try_times += 1

        print(t, 'epsilon:', epsilon, 'action:',
              action, 'reward:', reward,
              'q:', q_value[0], q_value[1])

        a_t = np.zeros([2], dtype=np.float32)
        a_t[action] = 1
        #store experience <status, action, reward, new_status> in replay memory D
        D.append((status, a_t, reward, new_status, terminal, q_value[action]))

        if len(D) > MAX_REPLAY_MEMORY:
            D.pop(0)
        
        if t > OBSERVE and terminal and try_times > 10:
            try_times = 0
            for train_times in range(200):
                print("train times:", train_times)
                #sample random transitions from replay memory D
                exp_batch = random.sample(D, MINI_BATCH_SIZE)
                s_batch = [exp[0] for exp in exp_batch]
                a_batch = [exp[1] for exp in exp_batch]
                r_batch = [exp[2] for exp in exp_batch]
                news_batch = [exp[3] for exp in exp_batch]
                t_batch = [exp[4] for exp in exp_batch]
                q_batch = [exp[5] for exp in exp_batch]

                #calculate target for each minibatch transition
                newq_batch = sess.run(logits, feed_dict={s: news_batch})

                for i in range(len(newq_batch)):
                    #if new_status is terminal state then tt=rr
                    if t_batch[i]:
                        q_batch[i] = r_batch[i]
                    #otherwise tt=rr + y*max(Q(new_status))
                    else:
                        q_batch[i] = (r_batch[i] + GAMMA*np.max(newq_batch[i]))
                sess.run(train_op, feed_dict={s: s_batch, a:a_batch, q:q_batch})

        status = new_status 
        t += 1

        if epsilon > 0.001:
            if t % 10000 == 0:
                epsilon = epsilon * 0.8

        if t % 5000 == 0:
            saver.save(sess, 'saved_model/bird-dqn', global_step = t)
