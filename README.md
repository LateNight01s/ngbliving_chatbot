# NGB Living ChatBot

## About ChatBots

### Types

- Retrieval-based approach (goal-oriented, narrow, predefined-responses)
- Generative model (chit-chat, general, commonsense)

Generative models are not yet Turing complete, require large amount of data. The SOTA generative models are very large (GPT3-175B, Meena-2.1B) and are for general purpose not for specific domain.

Retrieval-based models are goal-oriented, require domain specific data. There are many approaches involved, i.e, similarity functions with TF-IDF, Dual encoder LSTM, classifier models, Knowledge graphs.

For a website like NGB Living that offers their services to customers, a hybrid approach using both of these two options would work the best.

### References

- [Production Ready Chatbots: Generate if not Retrieve](https://arxiv.org/pdf/1711.09684.pdf)
