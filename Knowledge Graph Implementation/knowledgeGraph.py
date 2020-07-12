import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
from kgGetEntities import getEntities
from kgGetRelation import getRelation
#from kgMix import processSubjectObjectPairs
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)
#%matplotlib inline
df = pd.read_csv("dhoni.csv")
#df.shape
#print(df['sentence'].sample(5))
entityPairs = []

for i in tqdm(df["sentence"]):
  entityPairs.append(getEntities(i))
print(entityPairs[10:20])
relations = [getRelation(i) for i in tqdm(df['sentence'])]
print(pd.Series(relations).value_counts()[:50])


src= [i[0] for i in entityPairs]
trgt=[i[1] for i in entityPairs]
kg=pd.DataFrame({'source':src, 'target':trgt, 'edge':relations})

G=nx.from_pandas_edgelist(kg, "source", "target",edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()