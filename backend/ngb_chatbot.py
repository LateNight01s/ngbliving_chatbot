import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter
import tarfile
import nltk
import spacy
# from sentence_transformers import SentenceTransformer

nlp = ''
model = ''
stop_words = ''
df_data = ''
sent_emb = ''
responses_np = ''
# pdata = ''
# DF = ''
# vocab = ''
# tf_idf = ''
# sent_vec = ''


def to_lower(corpus):
    for i,doc in enumerate(corpus):
        corpus[i][1] = (doc[1]).lower()

    return corpus

def remove_stopwords(corpus):
    for i,doc in enumerate(corpus) :
        text = ''
        for token in nlp(doc[1]):
            word = token.text
            if word not in stop_words and len(word)>1:
                text = text + ' ' + word
        corpus[i][1] = text.strip()

    return corpus

def remove_punctuations(corpus):
    for i,doc in enumerate(corpus):
        sent = doc[1]
        sent = re.sub(r'[^\w\s]', ' ', sent)
        sent = re.sub('\s*\\n+', ' ', sent)
        sent = re.sub('ngb\s*living', 'ngbliving', sent)
        corpus[i][1] = sent.strip()

    return corpus

def lemmatize(corpus):
    for i,doc in enumerate(corpus):
        tokens = nlp(doc[1])
        text = ''
        for token in tokens:
            if (token.text).isspace() or len(token.text)<3: continue
            text += token.lemma_ + ' '
        corpus[i][1] = text.strip()

    return corpus

def preprocess(corpus):
    corpus = to_lower(corpus)
    # corpus = remove_stopwords(corpus)
    corpus = remove_punctuations(corpus)
    corpus = lemmatize(corpus)

    return corpus

def build_df(docs):
    DF = {}
    for i, doc in enumerate(docs):
        for word in doc:
            try:
                DF[word].add(i)
            except:
                DF[word] = {i}

    for i in DF: DF[i] = len(DF[i])

    vocab = [w for w in DF]

    return DF, vocab

def build_tfidf(docs):
    tf_idf = {}
    N = len(docs)
    for i, doc in enumerate(docs):
        counter = Counter(doc)
        for term in set(doc):
            tf = counter[term]/len(doc)
            df = DF[term]
            idf = np.log(N/(df+1))
            tf_idf[term] = tf * idf

    return tf_idf

def tfidf_vectorization():
    docs_vector = np.zeros((N, len(vocab)))
    for score in tf_idf:
        idx = vocab.index(score)
        docs_vector[score[0]][idx] = tf_idf[score]

    return docs_vector

def getWeightedVec(sent, i=0, q=False, tfidf=0):
    weights, vectors = [], []
    doc = nlp(sent)
    for token in doc:
        if token.has_vector:
            term = token.text
            if len(term) < 3: continue
            if  q is False:
                weight = tf_idf[i, term]
            else:
                weight = tfidf[term]
            weights.append(weight)
            vectors.append(token.vector)

    try:
        doc_vec = np.average(vectors, weights=weights, axis=0)
        # doc_vec = np.average(vectors, axis=0)
    except:
        return doc.vector

    return doc_vec

def getSentVectors(data):
    vectors = []
    for i, sent in enumerate(data):
        vector = getWeightedVec(sent, i)
        vectors.append(vector)

    return vectors

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def get_query_tfidf(pquery):
    N = len(sent_emb)
    tokens = [token.text for token in nlp(pquery) if not (token.text).isspace()]
    counter = Counter(tokens)
    tf_idf = {}
    for term in set(tokens):
        tf = counter[term]/len(tokens)
        try: df = DF[term]
        except: df = 0
        idf = np.log((N+1)/(df + 1))
        tf_idf[term] = tf * idf

    return tf_idf

def get_query_vector(pquery):
    q_vector = np.zeros((len(vocab)))
    tf_idf = get_query_tfidf(pquery)

    try:
        idx = vocab.index(term)
        q_vector[idx] = tf * idf
    except: pass

    return q_vector

def get_top_responses(sent_emb, q_vector, k=5):
    if k > len(sent_emb): k = len(sent_emb)
    scores = []

    for i, d_vec in enumerate(sent_emb):
        # cos_score = cosine_sim(q_vector, d_vec[1][0])
        cos_score = cosine_sim(q_vector, d_vec[1])
        scores.append((cos_score, d_vec[0]))


    try:
        scores.sort(reverse=True)
    except Exception as err:
        print(err)
        return None

    return [scores[i] for i in range(k)]

def handle_query(query):
    pquery = [[-1, ''.join(query)]]
    pquery = preprocess(pquery)
    q_tfidf = get_query_tfidf(pquery[0][1])
    q_vector = getWeightedVec(sent=pquery[0][1], i=0, tfidf=q_tfidf, q=True)
    # q_vector = model.encode([pquery[0][1]])
    # responses = get_top_responses(sent_emb, q_vector[0])
    responses = get_top_responses(sent_emb, q_vector)
    if responses is None:
        print('Sorry! I could not resolve that query, try again.')
        return None, None
    top_doc_id = responses[0][1]
    top_conf = responses[0][0] * 100

    return responses, responses_np[top_doc_id], top_conf

def main(spacyModel, data, sentEmb, bertTokens=None, query=''):
    global nlp, model
    global stop_words
    global df_data
    global sent_emb, responses_np
    # global pdata
    # global DF, vocab, tf_idf
    # global sent_vec


    nlp = spacyModel
    model = bertTokens
    stop_words = nlp.Defaults.stop_words
    df_data = data
    sent_emb = sentEmb
    # ques = df_data['Questions'].to_numpy()
    responses_np = df_data['Responses'].to_numpy()

    # data = [unicodedata.normalize("NFKD", doc.lower()) for doc in ques]
    # ques_map = []
    # for i,ques in enumerate(data):
    #     for q in ques.split('\n'):
    #         if not len(q): continue
    #         ques_map.append([i, q])

    # pdata = ques_map[:]

    # pdata = preprocess(pdata)
    # docs = [[token.text for token in nlp(doc) if not (token.text).isspace()] for doc in pdata]
    # DF, vocab = build_df(docs)
    # tf_idf = build_tfidf(docs)
    # sent_vec = getSentVectors(pdata)

    if __name__=='__main__':
        while (1):
            print('Please input a query> ', end='')
            query = input()
            responses, ans, conf = handle_query(query)
            if ans is None: continue
            if(conf < 80.0):
                ans = '''Sorry, I could not find any relevant information. Kindly contact our office or try again with a different query.'''
            print(f'Answer> {ans} | conf: {conf}\n')
            # print(responses)
    else:
        responses, ans, conf = handle_query(query)
        if(conf < 80.0):
            ans = '''Sorry, I could not find any relevant information. Kindly contact our office or try again with a different query.'''
        return ans

if __name__=='__main__':
    nlp = spacy.load('./en_core_web_md-2.3.1')
    # model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    df_data = pd.read_csv(f'./ngb_data.csv', encoding='utf8')
    sent_emb = np.load(f'./sent_emb_gloveTFIDF.npy', allow_pickle=True)
    # sent_emb = np.load(f'./sent_emb.npy', allow_pickle=True)

    main(nlp, df_data, sent_emb)
