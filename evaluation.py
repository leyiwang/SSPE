#coding=utf8
import os, datetime
from tool import performance
from tool import pytc
from tool.lbsa import LBSA

FNAME_LIST = ['negative', 'positive']
def load_data(fname_list, class_list):
    doc_str_list = []
    doc_class_list = []
    for doc_fname, class_fname in zip(fname_list, class_list):
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
        doc_class_list.extend([class_fname] * len(doc_str_list_one_class))
    return doc_str_list, doc_class_list

def build_lex_samps(doc_class_list, doc_uni_token):
    samp_dict_list,samp_class_list = [], []
    test = LBSA()
    for k in range(len(doc_class_list)):
        samp_class = doc_class_list[k]
        samp_class_list.append(samp_class)
        samp_dict, res = {}, []
        res.extend(test.build_sample(doc_uni_token[k]))
        samp_dict = dict(zip(range(1, len(res)+1), res))
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def start_demo(data_dir, result_dir):
    fname_samp_train = result_dir + os.sep + 'train.samp'
    fname_samp_test = result_dir + os.sep + 'test.samp'
    fname_model = result_dir + os.sep +'liblinear.model'
    fname_output= result_dir+ os.sep + 'test.out'
    class_list = range(1, len(FNAME_LIST) + 1)
    doc_str_list_train, doc_class_list_train = load_data([data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST], class_list)
    doc_str_list_test, doc_class_list_test = load_data([data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], class_list)
    doc_uni_token_train = pytc.get_doc_unis_list(doc_str_list_train)
    doc_uni_token_test =  pytc.get_doc_unis_list(doc_str_list_test)

    print "building samps......"
    samp_list_train, class_list_train = build_lex_samps(doc_class_list_train, doc_uni_token_train)
    samp_list_test, class_list_test = build_lex_samps(doc_class_list_test, doc_uni_token_test)
    print "saving samps......"
    pytc.save_samps(samp_list_train, class_list_train, fname_samp_train)
    pytc.save_samps(samp_list_test, class_list_test, fname_samp_test)
    
    print 'start training......'
    pytc.liblinear_exe(fname_samp_train, fname_samp_test, fname_model, fname_output)
    samp_class_list_linear = [int(x.split()[0]) for x in open(fname_output).readlines()[1:]]
    class_dict = dict(zip(class_list, ['neg', 'pos']))
    print 'evaluation...'
    result_dict = performance.demo_performance(samp_class_list_linear, doc_class_list_test, class_dict)
    res = ''
    for key in ['p_neg','r_neg','p_pos','r_pos','macro_f1','acc']:
        res += key + ':' + str(round(result_dict[key]*100, 4))+'%\t'
    print res.rstrip('\t')

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    data_dir, result_dir = 'data'+os.sep+'semeval2013', 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    start_demo(data_dir, result_dir)
    end_time = datetime.datetime.now()
    print 'Done!', '\nSeconds cost:', (end_time - start_time).seconds