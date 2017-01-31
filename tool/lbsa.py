#coding=utf8
import os
DICT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'data', 'dict'))
def load_lexicon(fname, convert_func):
    data_list = [x.strip().split('\t') for x in open(fname).readlines()]
    word_dict = {}
    for key, val in data_list:
        word_dict[key] = convert_func(val)
    return word_dict
class LBSA:
    '''
    Lexcion based sentiment analysis
    '''
    sspe_lexicon = load_lexicon(os.path.join(DICT_DIR, 'SSPE'), float)
    def feature_template_sspe(self, doc):
        '''
        get eight features of a sample by the sentiment lexicon
        max_score, sum_score, num of sentiment words, the last sentiment word
        '''
        pos_list, neg_list, result=[], [], [0]*8
        for term in set(doc):
            if term in self.lexicon:
                if self.lexicon[term]>0:
                    pos_list.append(self.lexicon[term])
                elif self.lexicon[term]<0:
                    neg_list.append(self.lexicon[term])
            if len(pos_list) > 0:
                result[:4]=[sum(pos_list), max(pos_list), len(pos_list), pos_list[-1]]
            if len(neg_list) > 0:
                result[4:]=[sum(neg_list), min(neg_list), len(neg_list), neg_list[-1]]
        return result
    
    def build_sample(self, doc, lexicon='sspe'):
        res = []
        if lexicon=='sspe':
            self.lexicon = self.sspe_lexicon
            res = self.feature_template_sspe(doc)
        return res