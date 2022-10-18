import fitz
import io
import docx
from pptx import Presentation
import re
import string
import nltk
import streamlit as st
nltk.download('all')
from nltk.corpus import stopwords
from nltk import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
stop_words=set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from rank_bm25 import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

def return_doc_from_bytes(pdfbytes):
  doc = fitz.open(stream=pdfbytes)
  return doc


def preprocessing(documents):
  documents_clean = []
  for d in documents:
    # Remove Unicode
    document_test = re.sub('[^a-zA-Z0-9]', ' ', str(d))
    # Remove Mentions
    document_test = re.sub(r'@\w+', '', document_test)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    # Lowercase the numbers
    document_test = re.sub(r'[0-9]', '', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    #tokenization
    document_test = document_test.split()
    #lemmmitization
    document_test = [lemmatizer.lemmatize(word) for word in document_test if not word in set(stopwords.words('english'))]
    document_test = ' '.join(document_test)
    documents_clean.append(document_test)
  return documents_clean
def data_string(cleaned_document):
  clean_text=''
  for i in cleaned_document:
    clean_text+=i+" "
  return clean_text

def search_report(documents_clean,query):
  tokenized_corpus = [doc.split(" ") for doc in documents_clean]
  bm25 = BM25Okapi(tokenized_corpus)
  tokenized_query = query.split()
  # doc_scores = bm25.get_scores(tokenized_query)
  result=bm25.get_top_n(tokenized_query,documents_clean , n=15)
  return result
def st_ui():
  st.set_page_config(layout = "wide")
  st.title("Auto Review Legal contracts - DocumentAI")  
  fileupload = st.sidebar.file_uploader("Upload a Contract here")
  select_category = st.sidebar.selectbox("select_category", ["category", "Content Analytics", "Risk Analytics","Search"])
  Button=st.sidebar.button('content Analytics')
  button=st.sidebar.button('Risk Analytics')
  Enter_text = st.sidebar.text_input("Text to search")
   
  if fileupload:
    text=[]
    pdfbytes = fileupload.getvalue()
    doc = return_doc_from_bytes(pdfbytes)
    for page in doc:
      text+=(page.get_text().split('\n'))
    cleaned_document=preprocessing(text)
    clean_text=data_string(cleaned_document)
    if select_category == "Content Analytics":
      if Button:
        st.header('wordcloud')
        wordcloud = WordCloud(width = 800, height =600,background_color ='white',min_font_size = 5,max_words=500).generate(clean_text)
        # plot the WordCloud image
        plt.figure(figsize = (15,10), facecolor = None)
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
        st.pyplot(fig=plt)
    if select_category == "Risk Analytics":
      tokens=[]
      for sentence in cleaned_document:
        tokens+=nltk.word_tokenize(sentence)
      a=Counter(tokens)
      risk_words=['omitted','Accident','Interruption','Failure','Consequence','Contingencies','harm','Crisis','Disaster','Emergency', 'Hazard','Intolerable', 'Mitigation','Uncertainties','possession','burdened','sublicensees',
                  'termination','indeminity','liability','breach','liquidity','missed delivery dates','warranty','problems','dispute','confidentiality' 'disclosures','litigation','compliance',
                  'conflicts','monetary','losses','Severity','interruption','Reduction','Damage','Vulnerability']
      for key, value in list(a.items()):
          if key not in risk_words:
              del a[key]
      r_text=''
      for key, value in list(a.items()):
          r_text+=key+" "
      st.write(r_text)
      #wordcloud = WordCloud(width=800,height=800,background_color='white').generate_from_frequencies(a)
      if button:
        st.header('risk analytics wordcloud')
        wordcloud = WordCloud(width=800,height=800,background_color='white').generate(r_text)
        # plot the WordCloud image
        plt.figure(figsize = (8,8), facecolor = None)
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
 
    if select_category == "Search":
      if Enter_text:
        result=search_report(cleaned_document,Enter_text.lower())
        st.header('Related information to clause')
        info=''
        for i in result:
            info+=i+" "
        st.write(info)
    
      

      
 

if __name__ == "__main__":
    st_ui()
