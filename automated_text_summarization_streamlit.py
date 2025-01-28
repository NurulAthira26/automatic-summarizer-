#!/usr/bin/env python
# coding: utf-8

# In[111]:


text = """history of artificial intelligence (AI), a survey of important events and people in the field of artificial intelligence (AI) from the early work of British logician Alan Turing in the 1930s to advancements at the turn of the 21st century. AI is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings. The term is frequently applied to the project of developing systems endowed with the intellectual processes characteristic of humans, such as the ability to reason, discover meaning, generalize, or learn from past experience. For modern developments in AI, see artificial intelligence.
Alan Turing and the beginning of AI of Theoretical work.The earliest substantial work in the field of artificial intelligence was done in the mid-20th century by the British logician and computer pioneer Alan Mathison Turing. In 1935 Turing described an abstract computing machine consisting of a limitless memory and a scanner that moves back and forth through the memory, symbol by symbol, reading what it finds and writing further symbols. The actions of the scanner are dictated by a program of instructions that also is stored in the memory in the form of symbols. This is Turing’s stored-program concept, and implicit in it is the possibility of the machine operating on, and so modifying or improving, its own program. Turing’s conception is now known simply as the universal Turing machine. All modern computers are in essence universal Turing machines.
During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945. Nevertheless, during the war he gave considerable thought to the issue of machine intelligence. One of Turing’s colleagues at Bletchley Park, Donald Michie (who later founded the Department of Machine Intelligence and Perception at the University of Edinburgh), later recalled that Turing often discussed how computers could learn from experience as well as solve new problems through the use of guiding principles—a process now known as heuristic problem solving.
Turing gave quite possibly the earliest public lecture (London, 1947) to mention computer intelligence, saying, “What we want is a machine that can learn from experience,” and that the “possibility of letting the machine alter its own instructions provides the mechanism for this.” In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery.” However, Turing did not publish this paper, and many of his ideas were later reinvented by others. For instance, one of Turing’s original ideas was to train a network of artificial neurons to perform specific tasks, an approach described in the section Connectionism."""


# In[112]:


len(text)


# In[113]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


# In[114]:


nlp = spacy.load('en_core_web_sm')


# In[115]:


doc = nlp(text)


# In[116]:


tokens = [token.text.lower() for token in doc
          if not token.is_stop and
          not token.is_punct and
          token.text !='\n']


# In[117]:


tokens


# In[118]:


tokens1=[]
stopwords = list(STOP_WORDS)
allowed_pos = ['ADJ','PROPN','VERB','NOUN']
for token in doc:
    if token.text in stopwords or token.text in punctuation:
        continue
    if token.pos_ in allowed_pos:
        tokens1.append(token.text)


# In[119]:


tokens1


# In[120]:


from collections import Counter


# In[121]:


word_freq = Counter(tokens)


# In[122]:


word_freq


# In[123]:


max_freq = max(word_freq.values())


# In[124]:


max_freq


# In[125]:


for word in word_freq.keys():
    word_freq[word] = word_freq[word]/max_freq


# In[126]:


word_freq


# In[127]:


sent_token = [sent.text for sent in doc.sents]


# In[128]:


sent_token


# In[129]:


sent_score = {}
for sent in sent_token:
    for word in sent.split():
        if word.lower() in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent] = word_freq[word]
            else:
                sent_score[sent] +=word_freq[word]
        print(word)


# In[130]:


sent_score


# In[131]:


import pandas as pd


# In[132]:


pd.DataFrame(list(sent_score.items()),columns=['Sentence','Score'])


# In[133]:


from heapq import nlargest


# In[134]:


num_sentences =3
n = nlargest(num_sentences,sent_score,key=sent_score.get)


# In[135]:


" ".join(n)


# In[136]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[137]:


get_ipython().system('from google.colab import files')
get_ipython().system('files.upload(history_ai_.py)  # Select history_ai_.py from your computer.')



# In[138]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

# Summarization function
def summarize_text(text, num_sentences=3):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Tokenize and remove stopwords
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc
              if not token.is_stop and not token.is_punct and token.text != '\n']

    # Word frequency
    word_freq = Counter(tokens)
    if not word_freq:
        return "Error: No valid words found in the text."

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

    # Sentence tokenization
    sent_token = [sent.text for sent in doc.sents]

    # Score sentences
    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                sent_score[sent] = sent_score.get(sent, 0) + word_freq[word]

    # Get top sentences
    summarized_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)
    return " ".join(summarized_sentences)

# text
text = """
history of artificial intelligence (AI), a survey of important events and people in the field of artificial intelligence (AI) from the early work of British logician Alan Turing in the 1930s to advancements at the turn of the 21st century. AI is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings. The term is frequently applied to the project of developing systems endowed with the intellectual processes characteristic of humans, such as the ability to reason, discover meaning, generalize, or learn from past experience. For modern developments in AI, see artificial intelligence.
Alan Turing and the beginning of AI of Theoretical work.The earliest substantial work in the field of artificial intelligence was done in the mid-20th century by the British logician and computer pioneer Alan Mathison Turing. In 1935 Turing described an abstract computing machine consisting of a limitless memory and a scanner that moves back and forth through the memory, symbol by symbol, reading what it finds and writing further symbols. The actions of the scanner are dictated by a program of instructions that also is stored in the memory in the form of symbols. This is Turing’s stored-program concept, and implicit in it is the possibility of the machine operating on, and so modifying or improving, its own program. Turing’s conception is now known simply as the universal Turing machine. All modern computers are in essence universal Turing machines.
During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945. Nevertheless, during the war he gave considerable thought to the issue of machine intelligence. One of Turing’s colleagues at Bletchley Park, Donald Michie (who later founded the Department of Machine Intelligence and Perception at the University of Edinburgh), later recalled that Turing often discussed how computers could learn from experience as well as solve new problems through the use of guiding principles—a process now known as heuristic problem solving.
Turing gave quite possibly the earliest public lecture (London, 1947) to mention computer intelligence, saying, “What we want is a machine that can learn from experience,” and that the “possibility of letting the machine alter its own instructions provides the mechanism for this.” In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery.” However, Turing did not publish this paper, and many of his ideas were later reinvented by others. For instance, one of Turing’s original ideas was to train a network of artificial neurons to perform specific tasks, an approach described in the section Connectionism.
"""
num_sentences = 2
summary = summarize_text(text, num_sentences)
print("Summary:")
print(summary)


# In[139]:


from transformers import pipeline


# In[140]:


summarizer=pipeline("summarization",model='t5-base',tokenizer='t5-base',framework='pt')


# In[141]:


text = """history of artificial intelligence (AI), a survey of important events and people in the field of artificial intelligence (AI) from the early work of British logician Alan Turing in the 1930s to advancements at the turn of the 21st century. AI is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings. The term is frequently applied to the project of developing systems endowed with the intellectual processes characteristic of humans, such as the ability to reason, discover meaning, generalize, or learn from past experience. For modern developments in AI, see artificial intelligence.
Alan Turing and the beginning of AI of Theoretical work.The earliest substantial work in the field of artificial intelligence was done in the mid-20th century by the British logician and computer pioneer Alan Mathison Turing. In 1935 Turing described an abstract computing machine consisting of a limitless memory and a scanner that moves back and forth through the memory, symbol by symbol, reading what it finds and writing further symbols. The actions of the scanner are dictated by a program of instructions that also is stored in the memory in the form of symbols. This is Turing’s stored-program concept, and implicit in it is the possibility of the machine operating on, and so modifying or improving, its own program. Turing’s conception is now known simply as the universal Turing machine. All modern computers are in essence universal Turing machines.
During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945. Nevertheless, during the war he gave considerable thought to the issue of machine intelligence. One of Turing’s colleagues at Bletchley Park, Donald Michie (who later founded the Department of Machine Intelligence and Perception at the University of Edinburgh), later recalled that Turing often discussed how computers could learn from experience as well as solve new problems through the use of guiding principles—a process now known as heuristic problem solving.
Turing gave quite possibly the earliest public lecture (London, 1947) to mention computer intelligence, saying, “What we want is a machine that can learn from experience,” and that the “possibility of letting the machine alter its own instructions provides the mechanism for this.” In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery.” However, Turing did not publish this paper, and many of his ideas were later reinvented by others. For instance, one of Turing’s original ideas was to train a network of artificial neurons to perform specific tasks, an approach described in the section Connectionism."""


# In[142]:


summary = summarizer(text,max_length=100,min_length=10,do_sample=False)


# In[143]:


summary


# In[144]:


print(summary[0]['summary_text'])


# In[145]:


get_ipython().system('pip install transformers')


# In[146]:


from transformers import pipeline

# Initialize the summarizer pipeline
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")

# Summarization function
def summarize_text(text, max_length=100, min_length=10):
    try:
        # Summarize the text
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

# Example usage
text = """
history of artificial intelligence (AI), a survey of important events and people in the field of artificial intelligence (AI) from the early work of British logician Alan Turing in the 1930s to advancements at the turn of the 21st century. AI is the ability of a digital computer or computer-controlled robot to perform tasks commonly associated with intelligent beings. The term is frequently applied to the project of developing systems endowed with the intellectual processes characteristic of humans, such as the ability to reason, discover meaning, generalize, or learn from past experience. For modern developments in AI, see artificial intelligence.
Alan Turing and the beginning of AI of Theoretical work.The earliest substantial work in the field of artificial intelligence was done in the mid-20th century by the British logician and computer pioneer Alan Mathison Turing. In 1935 Turing described an abstract computing machine consisting of a limitless memory and a scanner that moves back and forth through the memory, symbol by symbol, reading what it finds and writing further symbols. The actions of the scanner are dictated by a program of instructions that also is stored in the memory in the form of symbols. This is Turing’s stored-program concept, and implicit in it is the possibility of the machine operating on, and so modifying or improving, its own program. Turing’s conception is now known simply as the universal Turing machine. All modern computers are in essence universal Turing machines.
During World War II Turing was a leading cryptanalyst at the Government Code and Cypher School in Bletchley Park, Buckinghamshire, England. Turing could not turn to the project of building a stored-program electronic computing machine until the cessation of hostilities in Europe in 1945. Nevertheless, during the war he gave considerable thought to the issue of machine intelligence. One of Turing’s colleagues at Bletchley Park, Donald Michie (who later founded the Department of Machine Intelligence and Perception at the University of Edinburgh), later recalled that Turing often discussed how computers could learn from experience as well as solve new problems through the use of guiding principles—a process now known as heuristic problem solving.
Turing gave quite possibly the earliest public lecture (London, 1947) to mention computer intelligence, saying, “What we want is a machine that can learn from experience,” and that the “possibility of letting the machine alter its own instructions provides the mechanism for this.” In 1948 he introduced many of the central concepts of AI in a report entitled “Intelligent Machinery.” However, Turing did not publish this paper, and many of his ideas were later reinvented by others. For instance, one of Turing’s original ideas was to train a network of artificial neurons to perform specific tasks, an approach described in the section Connectionism.
"""
print("Original Text:")
print(text)

# Summarize
summary = summarize_text(text)
print("\nSummary:")
print(summary)


# In[146]:




