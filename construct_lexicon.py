#coding=utf8
import os, datetime, math
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
embed_index = 0
class KNN:
    def __init__(self):
        self.threshold_pos = 3
        self.threshold_neg = 6
        self.threshold_neu = 1
        
    def train(self, X, y, model):
        self.knn_clf = KNeighborsClassifier()
        self.knn_clf.fit(X, y)
        joblib.dump(self.knn_clf, model)
        print 'acc:',self.knn_clf.score(X, y)
        return self.knn_clf
    
    def predict(self, X, model=None):
        if model:
            self.knn_clf = joblib.load(model)
        labels = self.knn_clf.predict(X)
        return labels
    
    def ud_predict(self, sim_sent_list):
        result = []
        for i,labels in enumerate(sim_sent_list):
            final_target = -2
            num_neg, num_neu, num_pos = [labels.count(i) for i in range(-1, 2)]
            if (num_pos > num_neg + self.threshold_pos) and (num_pos > num_neu + self.threshold_pos):
                final_target = 1
            elif (num_neu > num_pos + self.threshold_neu) and (num_neu > num_neg + self.threshold_neu):
                final_target = 0
            elif (num_neg > num_pos + self.threshold_neg) and (num_neg > num_neu + self.threshold_neg):
                final_target = -1
            result.append(final_target)
        return result

class Softmax():
    def __init__(self, class_num, batch_size, embed_size = 50):
        self.class_num = class_num
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.label_dic = {-1:[1,0,0], 0:[0,1,0], 1:[0,0,1]}
        
    def next_batch(self, datalist, labels):
        global embed_index
        data, targets = [], []
        for i in range(self.batch_size):
            sample = datalist[embed_index]
            label = self.label_dic[labels[embed_index]]
            data.append(sample)
            targets.append(label)
            embed_index = (embed_index + 1) % len(datalist)
        return data, targets
    
    def run(self, model, graph_path, data_list, labels, test):
        class_num, embed_size = self.class_num, self.embed_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            embed_inputs = tf.placeholder(tf.float32, shape=[None, embed_size])
            embed_labels = tf.placeholder(tf.float32, shape=[None, class_num])
            
            embed_weights = tf.Variable(tf.truncated_normal([embed_size, class_num], stddev=1.0 / math.sqrt(embed_size)))
            embed_biases = tf.Variable(tf.zeros([class_num]))
            #norm2 = tf.reduce_sum(tf.pow(embed_weights, 2))
            embed_y = tf.nn.softmax(tf.matmul(embed_inputs, embed_weights) + embed_biases)
            correct_prediction = tf.equal(tf.argmax(embed_y, 1), tf.argmax(embed_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            loss = - tf.reduce_sum(embed_labels * tf.log(embed_y + 1e-5))
            optimizer = tf.train.AdagradOptimizer(0.2).minimize(loss)
            saver = tf.train.Saver()
            tf.scalar_summary('accuracy', accuracy)
            tf.scalar_summary('loss', loss)
            merged_summary = tf.merge_all_summaries()
            
        with tf.Session(graph=self.graph) as session:
            init = tf.initialize_all_variables()
            session.run(init)
            #saver.restore(session, 'model' + os.sep + str(84000) + ".ckpt")
            print("Initialized")
            if not os.path.exists(model):
                os.mkdir(model)
            if not os.path.exists(graph_path):
                os.mkdir(graph_path)
            writer = tf.train.SummaryWriter(graph_path, session.graph)
            average_loss, avg_train_acc = 0, 0
            for step in xrange(5001):
                batch_inputs, batch_labels = self.next_batch(data_list, labels)
                feed_dict = {embed_inputs: batch_inputs, embed_labels: batch_labels}
                _, loss_val, train_acc, summary = session.run([optimizer, loss, accuracy, merged_summary], feed_dict=feed_dict)
                average_loss += loss_val; avg_train_acc += train_acc
                if step % 500 == 0:
                    save_path = saver.save(session, os.path.join(model, str(step)) + ".ckpt")
                    print("Save to path: ", save_path)
                    if step >= 400: average_loss /= 400; avg_train_acc/=400
                    print"Average loss at step ", step, ": ", average_loss, 'train_acc:', avg_train_acc
                    average_loss, avg_train_acc = 0, 0
                    writer.add_summary(summary, step)
            feed_dict = {embed_inputs: test}
            self.result = session.run(embed_y, feed_dict=feed_dict)
    def save_lexicon(self, words, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        class_set = ['negative','neutral','positive']
        for class_id in range(self.class_num):
            class_label = class_set[class_id]
            lex_list = [(words[i],str(self.result[i][class_id])) for i,item in enumerate(self.result)\
                                 if self.result[i][class_id]>0.5]
            if class_label=='negative':
                order_list=['\t-'.join(item) for item in sorted(lex_list, key=lambda x:float(x[1]), reverse=True)]
            else:
                order_list=['\t'.join(item) for item in sorted(lex_list, key=lambda x:float(x[1]), reverse=True)]
            doc_str = '\n'.join(order_list)
            fobj = open(save_dir + os.sep + class_label, 'w')
            fobj.write(doc_str)
            fobj.close()

def walk_dir(data_dir):
    filelist = []
    for root, dir, files in os.walk(data_dir):
        for f in files:
            fullpath = os.path.join(root, f)
            filelist.append(fullpath)
    return filelist

def load_embeddings(filename):
    embedings_dic = {}
    for index, line in enumerate(open(filename, 'r').readlines()):
        line = line.strip('\n')
        key, value = line.split()[0], line.split()[1:]
        embedings_dic[key] = map(float, value)
    return embedings_dic

def load_seeds(fnamelist, embedings_dic, label_dict={'negative':-1, 'neutral':0, 'positive':1}):
    data, target, items = [], [], []
    for filename in fnamelist:
        fname = os.path.split(filename)[1]
        data_list = open(filename, 'r').read().split('\n')
        data_items = [(item, embedings_dic[item]) for item in \
                      data_list if embedings_dic.has_key(item)]
        items, data_cur = zip(*data_items)
        label_cur = [label_dict[fname]] * len(data_cur)
        data.extend(data_cur); target.extend(label_cur)
    return data, target, items

def load_UBDictory(fnamelist, embedings_dic):
    sim_dic = {}
    for filename in fnamelist:
        data_list = [[word.replace(" ", "<w-w>") for word in line.split('\t')]\
                     for line in open(filename, 'r').read().split('\n')]
        for item in data_list:
            key, value = item[0], [w for w in item[1:] if embedings_dic.has_key(w)]
            if embedings_dic.has_key(key) and len(value) > 0:
                if not sim_dic.has_key(key):
                    sim_dic[key]= [word for word in value if embedings_dic.has_key(word)]
                else:
                    cand_words = [word for word in value if embedings_dic.has_key(word)]
                    sim_dic[key] = list(set(sim_dic[key] + cand_words))
    return sim_dic

def build_dataset(sim_dic, embd_sent_dic):
    dic = {}
    for key in sim_dic.keys():
        dic[key] = [embd_sent_dic[word] for word in sim_dic[key]]
    words, sents_list = zip(*sorted(dic.items(), key=lambda x:x[-1]))
    return words, sents_list

def build_extend_lexicon():
    fnamelist, ud_fnamelist, model = walk_dir(os.path.join('data', 'seeds')), \
    walk_dir(os.path.join('data', 'ud')), os.path.join('model', 'knn.model')
    embedings_dic  =  load_embeddings(r'embeddings' + os.sep + 'embedding')
    ud_sim_dic = load_UBDictory(ud_fnamelist, embedings_dic)
    knn = KNN()
    seeds_data, seeds_target = load_seeds(fnamelist, embedings_dic)[:2]
    knn.train(seeds_data, seeds_target, model)
    vocabulary, embeddings = zip(*sorted(embedings_dic.items(), key=lambda x:x[-1]))
    train_targets = knn.predict(embeddings)
    embd_sent_dic = dict(zip(vocabulary, train_targets))
    ud_words, ud_sim_sent_list = build_dataset(ud_sim_dic, embd_sent_dic)
    ud_words_target = knn.ud_predict(ud_sim_sent_list)
    
    ud_neg_str = '\n'.join([word for i, word in enumerate(ud_words)\
                            if ud_words_target[i]==-1])
    ud_neu_str = '\n'.join([word for i, word in enumerate(ud_words)\
                            if ud_words_target[i]==0])
    ud_pos_str = '\n'.join([word for i, word in enumerate(ud_words)\
                            if ud_words_target[i]==1])
    open('result' + os.sep + 'negative','w').write(ud_neg_str)
    open('result' + os.sep + 'neutral','w').write(ud_neu_str)
    open('result' + os.sep + 'positive','w').write(ud_pos_str)
    
def build_lexicon():
    fnamelist = walk_dir('result')
    embedding_fname = r'embeddings' + os.sep + 'embedding'
    embedings_dic = load_embeddings(embedding_fname)
    data, target = load_seeds(fnamelist, embedings_dic)[:2]
    vocabulary, embeddings = zip(*embedings_dic.items())
    softmax = Softmax(3, 2000)
    softmax.run('model','graph', data, target, embeddings)
    softmax.save_lexicon(vocabulary, 'final')
    
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    build_extend_lexicon()
    build_lexicon()
    end_time = datetime.datetime.now()
    print 'Done!', '\nSeconds cost:', (end_time - start_time).seconds