import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter
import tarfile
import nltk
import spacy
# import wmd

nlp = ''
stop_words = ''
df_numpy = ''
pdata = ''
DF = ''
vocab = ''
tf_idf = ''
sent_vec = ''


def to_lower(corpus):
    for i,doc in enumerate(corpus):
        corpus[i] = doc.lower()

    return corpus

def remove_stopwords(corpus):
    for i,doc in enumerate(corpus) :
        text = ''
        for token in nlp(doc):
            word = token.text
            if word not in stop_words and len(word)>1:
                text = text + ' ' + word
        corpus[i] = text.strip()

    return corpus

def remove_punctuations(corpus):
    # symbols = "!\"#$%&()*+-./:;,<=>?@[\]^_`{|}~\n"
    # table = str.maketrans('', '', symbols)
    for i, doc in enumerate(corpus):
        sent = doc
        sent = re.sub(r'[^\w\s]', ' ', sent)
        sent = re.sub('\s*\\n+', ' ', sent)
        sent = re.sub('ngb\s*living', 'ngbliving', sent)
        corpus[i] = sent.strip()
        # corpus[i] = doc.translate(table)

    return corpus

def lemmatize(corpus):
    for i, doc in enumerate(corpus):
        tokens = nlp(doc)
        text = ''
        for token in tokens:
            if (token.text).isspace() or len(token.text)<3: continue
            text += token.lemma_ + ' '
        corpus[i] = text.strip()

    return corpus

def preprocess(corpus):
    corpus = to_lower(corpus)
    corpus = remove_stopwords(corpus)
    corpus = remove_punctuations(corpus)
    corpus = lemmatize(corpus)

    return corpus

def get_vocab(data):
    wc = {}
    for doc in data:
        for token in nlp(doc):
            word = token.text
            try: wc[word] += 1
            except: wc[word] = 1

    return wc

def get_oov(data, wc):
    oov = {}
    for doc in data:
        for token in nlp(doc):
            if not token.has_vector and token.text not in oov:
                oov[token.text] =  wc[token.text]

    return oov

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
            tf_idf[i, term] = tf * idf

    return tf_idf

def tfidf_vectorization():
    docs_vector = np.zeros((N, len(vocab)))
    for score in tf_idf:
        idx = vocab.index(score[1])
        docs_vector[score[0]][idx] = tf_idf[score]

    return docs_vector

def getWeightedVec(sent, i=0, q=False, tfidf=0):
    weights, vectors = [], []
    doc = nlp(sent)
    for token in doc:
        if token.has_vector:
            term = token.text
            if len(term) < 3: continue
            # if  q is False:
            #     weight = tf_idf[i, term]
            # else:
            #     weight = tfidf
            # weights.append(weight)
            vectors.append(token.vector)

    # try: doc_vec = np.average(vectors, weights=weights, axis=0)
    try: doc_vec = np.average(vectors, axis=0)
    except: return doc.vector

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
    N = len(pdata)
    tokens = [token.text for token in nlp(pquery[0]) if not (token.text).isspace()]
    counter = Counter(tokens)
    for term in set(tokens):
        tf = counter[term]/len(tokens)
        try: df = DF[term]
        except: df = 0
        idf = np.log((N+1)/(df + 1))

    return tf * idf

def get_query_vector(pquery):
    q_vector = np.zeros((len(vocab)))
    tf_idf = get_query_tfidf(pquery)

    try:
        idx = vocab.index(term)
        q_vector[idx] = tf * idf
    except: pass

    return q_vector

def get_top_responses(q_vector, k=5):
    if k > len(pdata): k = len(pdata)
    scores = []
    for i, d_vec in enumerate(sent_vec):
        cos_score = cosine_sim(q_vector, d_vec)
        scores.append((cos_score, i))

    try:
        scores.sort(reverse=True)
    except:
        return None

    return [scores[i] for i in range(k)]

def handle_query(query):
    pquery = preprocess([''.join(query)])
    # print('pquery:', pquery)
    q_tfidf = get_query_tfidf(pquery)
    q_vector = getWeightedVec(sent=pquery[0], i=0, tfidf=q_tfidf, q=True)
    responses = get_top_responses(q_vector)
    if responses is None:
        print('Sorry! I could not resolve that query, try again.')
        return None, None
    top_doc_id = responses[0][1]

    return responses, df_numpy[top_doc_id][0]

def main(query=''):
    global nlp
    global stop_words
    global df_numpy
    global pdata
    global DF, vocab, tf_idf
    global sent_vec

    nlp = spacy.load('./en_core_web_lg-2.3.1')
    # nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
    stop_words = nlp.Defaults.stop_words

    df = pd.read_csv(f'./ngb_data.csv', encoding='utf8')

    df_numpy = df.to_numpy()
    data = [unicodedata.normalize("NFKD", str(doc[0]).lower()) for doc in df_numpy]

    pdata = data[:]
    pdata = preprocess(pdata)

    # wc = get_vocab(pdata)
    # oov = get_oov(pdata, wc)
    # wc_top = sorted(wc.items(), key=lambda x: x[1])[::-1]
    # oov_top = sorted(oov.items(), key=lambda x: x[1])[::-1]

    docs = [[token.text for token in nlp(doc) if not (token.text).isspace()] for doc in pdata]

    DF, vocab = build_df(docs)
    tf_idf = build_tfidf(docs)

    sent_vec = getSentVectors(pdata)

    if __name__=='__main__':
        while (1):
            print('Please input a query> ', end='')
            query = input()
            responses, ans = handle_query(query)
            if ans is None: continue
            print(f'Answer> {ans}\n')
            # print(responses)
    else:
        responses, ans = handle_query(query)
        return ans

if __name__=='__main__':
    main()
