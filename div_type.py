import pickle
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


class subtopic:
    def __init__(self, subtopic_id, subtopic):
        self.subtopic_id = subtopic_id
        self.subtopic = subtopic


class div_query:
    def __init__(self, qid, query, subtopic_id_list, subtopic_list):
        '''
        object for diversity query
        alpha = 0.5 by default
        doc_list: the inital document ranking derived from indri
        doc_score_list: the normalized relevance score list of documents
        best_metric: the best metric of the query
        stand_alpha_DCG: stand alpha-DCG (from DSSA) used for normalization
        '''
        self.qid = qid
        self.query = query
        self.subtopic_id_list = subtopic_id_list
        self.subtopic_list = []
        self.doc_list = []
        self.doc_score_list = []
        self.best_metric = 0
        self.stand_alpha_DCG = 0

        for index in range(len(subtopic_id_list)):
            t = subtopic(subtopic_id_list[index], subtopic_list[index])
            self.subtopic_list.append(t)

    def set_std_metric(self, m):
        self.stand_alpha_DCG = m

    def add_docs(self, doc_list):
        self.doc_list = doc_list
        self.DOC_NUM = len(self.doc_list)
        init_data = np.zeros((len(doc_list), len(self.subtopic_list)), dtype=int)
        self.subtopic_df = pd.DataFrame(init_data, columns=self.subtopic_id_list, index=doc_list)

    def add_query_suggestion(self, query_suggestion):
        self.query_suggestion = query_suggestion

    def add_docs_rel_score(self, doc_score_list):
        self.doc_score_list = doc_score_list

    def get_test_alpha_nDCG(self, docs_rank):
        '''
        get the alpha_nDCG@20 for the input document list (for testing).
        '''
        temp_data = np.zeros((len(docs_rank), len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.best_subtopic_df)
        metrics = []
        p = 0.5
        real_num = min(20, len(docs_rank))
        best_docs_index = []
        for index in range(real_num):
            result_index = self.best_docs_rank.index(docs_rank[index])
            best_docs_index.append(result_index)
            temp_data[index, :] = temp_array[result_index, :]
            if index == 0:
                score = np.sum(temp_data[index, :])
                metrics.append(score)
            else:
                r_ik = np.array([np.sum(temp_data[:index, s]) for s in range(temp_data.shape[1])], dtype=np.int64)
                t = np.power(p, r_ik)
                score = np.dot(temp_data[index, :], t) / np.log2(2 + index)
                metrics.append(score)
        ''' normalized by the stand alpha DCG '''
        if hasattr(self, 'stand_alpha_DCG') and self.stand_alpha_DCG > 0:
            try:
                alpha_nDCG = np.sum(metrics) / self.stand_alpha_DCG
            except:
                print('except np.sum =', np.sum(metrics), 'self.global_best_metric = ', self.global_best_metric)
        else:
            print('error! qid =', self.qid)
            alpha_nDCG = 0
        return alpha_nDCG

    def get_alpha_DCG(self, docs_rank, print_flag=False):
        '''
        get the alpha-DCG for the input document list (for generating training samples)
        '''
        temp_data = np.zeros((len(docs_rank), len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.best_subtopic_df)
        metrics = []
        p = 0.5
        for index in range(len(docs_rank)):
            result_index = self.best_docs_rank.index(docs_rank[index])
            temp_data[index, :] = temp_array[result_index, :]
            if index == 0:
                score = np.sum(temp_data[index, :])
                metrics.append(score)
            else:
                r_ik = np.array([np.sum(temp_data[:index, s]) for s in range(temp_data.shape[1])], dtype=np.int64)
                t = np.power(p, r_ik)
                score = np.dot(temp_data[index, :], t) / np.log2(2 + index)
                metrics.append(score)
        if print_flag:
            print('self.best_gain = ', self.best_gain, 'sum(best_gain) = ', np.sum(self.best_gain), 'best_metric = ',
                  self.best_metric)
            print('test metrics = ', metrics, 'sum(metrics) = ', np.sum(metrics))
        '''get the total gain for the input document list'''
        alpha_nDCG = np.sum(metrics)
        return alpha_nDCG

    def get_best_rank(self, top_n=None, alpha=0.5):
        '''
        get the best diversity document ranking based on greedy strategy
        '''
        p = 1.0 - alpha
        if top_n == None:
            top_n = self.DOC_NUM
        real_num = int(min(top_n, self.DOC_NUM))
        temp_data = np.zeros((real_num, len(self.subtopic_list)), dtype=int)
        temp_array = np.array(self.subtopic_df)
        best_docs_rank = []
        best_docs_rank_rel_score = []
        best_gain = []
        ''' greedy document selection '''
        for step in range(real_num):
            scores = []
            if step == 0:
                for index in range(real_num):
                    temp_score = np.sum(temp_array[index, :])
                    scores.append(temp_score)
                result_index = np.argsort(scores)[-1]
                gain = scores[result_index]
                docid = self.doc_list[result_index]
                doc_rel_score = self.doc_score_list[result_index]
                best_docs_rank.append(docid)
                best_docs_rank_rel_score.append(doc_rel_score)
                best_gain.append(scores[result_index])
                temp_data[0, :] = temp_array[result_index, :]
            else:
                for index in range(real_num):
                    if self.doc_list[index] not in best_docs_rank:
                        r_ik = np.array([np.sum(temp_data[:step, s]) for s in range(temp_array.shape[1])],
                                        dtype=np.int64)
                        t = np.power(p, r_ik)
                        temp_score = np.dot(temp_array[index, :], t)
                        scores.append(temp_score)
                    else:
                        scores.append(-1.0)
                result_index = np.argsort(scores)[-1]
                gain = scores[result_index]
                docid = self.doc_list[result_index]
                doc_rel_score = self.doc_score_list[result_index]
                if docid not in best_docs_rank:
                    best_docs_rank.append(docid)
                    best_docs_rank_rel_score.append(doc_rel_score)
                else:
                    print('document already added!')
                best_gain.append(scores[result_index] / np.log2(2 + step))
                temp_data[step, :] = temp_array[result_index, :]
        self.best_docs_rank = best_docs_rank
        self.best_docs_rank_rel_score = best_docs_rank_rel_score
        self.best_gain = best_gain
        self.best_subtopic_df = pd.DataFrame(temp_data, columns=self.subtopic_id_list, index=self.best_docs_rank)
        self.best_metric = np.sum(self.best_gain)

