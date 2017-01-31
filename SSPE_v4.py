#coding=utf8
import collections
import numpy as np
from nltk import ngrams
import tensorflow as tf
import os, sys, math, random, datetime

data_index, sent_index, buffer = 0, 0, collections.deque(maxlen=3)
def load_data(filename, min_len=0):
    '''
    return a list with each element one row of the input file
    '''
    result= []
    docs_str = open(filename, 'r').read().decode('utf-8','ignore').encode('utf-8','ignore')
    u_list = [line.split() for line in docs_str.split('\n')[:400000] if len(line.split()) >= min_len]
    return u_list

def save_embeddings(embeddings, dictionary, vocabulary_size, filename):
    fout = open(filename, 'wb')
    for i in range(vocabulary_size):
        fout.write(dictionary[i] + '\t' + ' '.join(map(str, embeddings[i])) + '\n')

def build_dataset(data_list):
    count = [['UNK', -1]]
    count.extend([item for item in collections.Counter([word for doc in data_list for word in doc]).most_common() if item[1] > 1])
    count.extend([['voc_FILL_WORD_voc', 0]])
    sents_len_list = [len(doc) for doc in data_list]
    max_sent_len, vocabulary_size = max(sents_len_list), len(count)
    print 'Token Num:%d\tVocabulary Size:%d'% (sum(sents_len_list), vocabulary_size)
    dictionary = dict(zip([word[0] for word in count], range(vocabulary_size)))
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for i, doc in enumerate(data_list):
        for j, word in enumerate(doc):
            if word in dictionary: data_list[i][j] = dictionary[word]
            else:data_list[i][j] = 0; count[0][1]+=1
        if sents_len_list[i] < max_sent_len: data_list[i].extend([vocabulary_size - 1] * (max_sent_len - sents_len_list[i]))
    data_list, sents_len_list = np.array(data_list), np.array(sents_len_list)
    return data_list, sents_len_list, reverse_dictionary, vocabulary_size, max_sent_len
    
def generate_batch(batch_size, num_skips, skip_window, data_list, sents_len_list):
    global data_index, sent_index, buffer
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    sents = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * skip_window + 1
    for i in range(batch_size // num_skips):
        if len(buffer)==0:
            while not (data_index <= (sents_len_list[sent_index] - span)):
                sent_index, data_index = (sent_index+1) % len(data_list), 0
            for _ in range(span):
                buffer.append(data_list[sent_index][data_index])
                data_index += 1
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            sents[i * num_skips + j] = sent_index
        if data_index < sents_len_list[sent_index]:
            buffer.append(data_list[sent_index][data_index])
            data_index += 1
        elif data_index == sents_len_list[sent_index]:
            data_index, sent_index = 0, (sent_index+1) % len(data_list)
            buffer.clear()
    return batch, labels, sents.tolist()

def start_demo():
    print "Readding data"
    pos_list, neg_list = load_data('data' + os.sep + 'pos_fenci', min_len=6), load_data('data' + os.sep + 'neg_fenci', min_len=6)
    dataset, sents_label = pos_list + neg_list , np.array([[0,1]] * len(pos_list) + [[1,0]] * len(neg_list))
    print "End Readding"
    data_list, sents_len_list, reverse_dictionary, vocabulary_size, max_sent_len = build_dataset(dataset)
    del pos_list, neg_list, dataset
    sample_num = data_list.shape[0]
    print 'sample_num:', sample_num
    batch_size, embedding_size, skip_window, num_skips, num_sampled, num_steps, alpha = 3008, 50, 1, 2, 1504, 100001, 0.3
    indices, stop = np.random.permutation(sample_num), sample_num - 10000
    data_list, sents_label, sents_len_list, test_list, test_labels, sents_len_test = data_list[indices[:stop]], sents_label[indices[:stop]], \
        sents_len_list[indices[:stop]], data_list[indices[stop:]], sents_label[indices[stop:]], sents_len_list[indices[stop:]]
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        sents_inputs = tf.placeholder(tf.int32, shape=[None, max_sent_len])
        sents_labels = tf.placeholder(tf.float32, shape=[None, 2])
        sents_len_weight = tf.placeholder(tf.float32, shape=[None])

        embeddings = tf.Variable(tf.concat(0, [tf.random_uniform([vocabulary_size-1, embedding_size], -1.0, 1.0), tf.zeros([1, embedding_size])]))
        nce_weights = tf.Variable(tf.concat(0, [tf.truncated_normal([vocabulary_size-1, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), \
                                                tf.zeros([1, embedding_size])]))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
        word_embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        embedding_input = tf.nn.embedding_lookup(embeddings, sents_inputs)
        sent_embed = tf.matmul(tf.diag(1/sents_len_weight), tf.reduce_sum(embedding_input, 1))
        
        sent_weights = tf.Variable(tf.truncated_normal([embedding_size, 2], stddev=1.0 / math.sqrt(embedding_size)))
        sent_biases = tf.Variable(tf.zeros([2]))
        sent_y = tf.nn.softmax(tf.matmul(sent_embed, sent_weights) + sent_biases)

        embed_new = tf.concat(0, [embeddings[:vocabulary_size-1], tf.zeros([1, embedding_size])])
        embed_update = tf.assign(embeddings, embed_new)
        
        correct_prediction = tf.equal(tf.argmax(sent_y, 1), tf.argmax(sents_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        syntactic_loss = alpha * tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, word_embed, train_labels, num_sampled, vocabulary_size))
        sentiment_loss = (1 - alpha) * (-tf.reduce_sum(sents_labels * tf.log(sent_y + 1e-5)))
        loss = syntactic_loss + sentiment_loss
        optimizer = tf.train.AdagradOptimizer(0.3).minimize(loss)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        tf.scalar_summary('accuracy', accuracy)
        tf.scalar_summary('loss', loss)
        tf.scalar_summary('sentiment_loss', sentiment_loss)
        tf.scalar_summary('syntactic_loss', syntactic_loss)
        merged_summary = tf.merge_all_summaries()
        
    with tf.Session(graph=graph) as session:
        session.run(init)
        #saver.restore(session, 'model' + os.sep + str(13000) + ".ckpt")
        print "Initialized"
        model_path,graph_path = 'model', 'graph'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)
        writer = tf.train.SummaryWriter(graph_path, session.graph)
        average_loss, avg_sen_loss, avg_syn_loss, avg_train_acc = 0, 0, 0, 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels, sents_index = generate_batch(batch_size, num_skips, skip_window, data_list, sents_len_list)
            batch_sents_inputs, batch_sents_labels, batch_sents_len_weight = data_list[sents_index], sents_label[sents_index], sents_len_list[sents_index]
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, sents_inputs:batch_sents_inputs, \
                         sents_labels:batch_sents_labels, sents_len_weight: batch_sents_len_weight}
            session.run([optimizer], feed_dict=feed_dict)
            session.run(embed_update)
            sen_loss_val, syn_loss_val, loss_val, train_acc, summary = session.run([sentiment_loss, syntactic_loss, \
                                                                                       loss, accuracy, merged_summary], feed_dict=feed_dict)
            average_loss += loss_val; avg_sen_loss += sen_loss_val; avg_syn_loss += syn_loss_val; avg_train_acc += train_acc
            writer.add_summary(summary, step)
            if step % 200 == 0:
                save_path = saver.save(session, model_path + os.sep + str(step) + ".ckpt")
                print("Save to path: ", save_path)
                test_sents_len_weight = sents_len_test
                feed_dict = {sents_inputs:test_list, sents_labels:test_labels, sents_len_weight: test_sents_len_weight}
                if step > 0: average_loss /=200; avg_sen_loss /=200; avg_syn_loss /=200; avg_train_acc/=200
                print"Average loss at step ", step, ": ", average_loss,'sent_loss:', avg_sen_loss,'sys_loss:', avg_syn_loss, \
                     'train_acc:', avg_train_acc, 'test_acc:', session.run(accuracy, feed_dict=feed_dict)
                average_loss, avg_sen_loss, avg_syn_loss, avg_train_acc = 0, 0, 0, 0
                final_embeddings = session.run(embeddings)
                save_embeddings(final_embeddings, reverse_dictionary, vocabulary_size, 'embedding')
if __name__=='__main__':
    start_time = datetime.datetime.now()
    start_demo()
    end_time = datetime.datetime.now()
    print 'Done!', '\nSeconds cost:', (end_time - start_time).seconds
