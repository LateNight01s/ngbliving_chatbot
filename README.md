# NGB Living ChatBot

## About ChatBots

### Types

- Retrieval-based approach (goal-oriented, narrow, predefined-responses)
- Generative model (chit-chat, general, commonsense)

Generative models are not yet Turing complete, require large amount of data. The SOTA generative models are very large (GPT3-175B, Meena-2.1B) and are for general purpose not for specific domain.

Retrieval-based models are goal-oriented, require domain specific data. There are many approaches involved, i.e, similarity functions with TF-IDF, Dual encoder LSTM, classifier models, Knowledge graphs.

For a website like NGB Living that offers their services to customers, a hybrid approach using both of these two options would work the best.

## Knowledge Graph

KG as the name suggests is a graph based structured data with _entities_ as nodes and their relationship with other entities defined by an edge in the graph.

![Example of a KG](https://miro.medium.com/max/1446/1*yhtuMXi91btQDLXR1ldtcQ.png)

> **triple**: (Leonard Nimoy, played, Spock), (Spock, character in, Star Trek)

KG can be constructed from unstructured text using various NLP methods like Named Entity Recognition (NER), Keyword Extraction, Sentence Segmentation, etc.

KG are widely used in NLP based system like intelligent chatbots, cognitive search system, QA application, etc. [Google Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_Graph) is the knowledge base that Google uses to enhance it's search algorithm thats how Google Assistance seems so intelligent.

### References

- [Production Ready Chatbots: Generate if not Retrieve](https://arxiv.org/pdf/1711.09684.pdf)
