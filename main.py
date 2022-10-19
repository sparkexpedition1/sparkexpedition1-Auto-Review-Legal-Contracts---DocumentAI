__author__ = "Kalpana D"
__copyright__ = "Copyright 2022, Daisi Hackathon"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__email__ = "spark.expedition@gmail.com"
__status__ = "Development"
__description__ = "Working on 7 insights modules"

##STEP 1 - Approach 'Counselor' with File Types Supported for Contracts and Agreements
# PDF
# DOCX (Microsoft Office - Word)
# PPT (Microsoft Office - PowerPoint)
# Simple TEXT
# Scanned IMAGES
# Hand-written IMAGES

## STEP 2- 'Counselor' analyzed with the following AI Powered Investigation and Analysis Modules
# Simplify - Auto Summarization to shorten content
# Risk Analytics - 
# Spatial Analytics - Find locations in the Contracts
# Payments/Pricing Analytics
# Stakeholders/Parties Analytics - Who are all involved (individuals, agencies, companies) in this C
# Search Content - like Penalty on delay more than 7 days

## STEP 3- 'Counselor' analyzes the content based on request and provided Summary
# Occurence of word instances with density as Word Cloud
# Text Summary

## STEP 4 - 'Counselor' emails the Summary to requested email ID for complex analysis (not capable of showing instantly)

# Initialize Python Libraries
import fitz
import io
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
import numpy as np

__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "Extract the PDF document and provide the output as Bytes"

def return_doc_from_bytes(pdfbytes):
  doc = fitz.open(stream=pdfbytes)
  return doc


__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "NLP Cleansing Algorithm - Remove extra spaces, stop words , Mentions, Trivial text etc."

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


__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "Prepare the Data String for Visualization Formats"

def data_string(cleaned_document):
  clean_text=''
  for i in cleaned_document:
    clean_text+=i+" "
  return clean_text

__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "Search the User Provided Content in the Document"

def search_report(documents_clean,query):
  tokenized_corpus = [doc.split(" ") for doc in documents_clean]
  bm25 = BM25Okapi(tokenized_corpus)
  tokenized_query = query.split()
  # doc_scores = bm25.get_scores(tokenized_query)
  result=bm25.get_top_n(tokenized_query,documents_clean , n=15)
  return result

__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "Power the StreamLit UI with required controls"

def st_ui():
  st.set_page_config(layout = "wide")
  st.title("'Counselor' -DocumentAI powered Review")  

  file_upload = st.sidebar.file_uploader("Upload Contract/Agreement here..")
  select_category = st.sidebar.selectbox("Select Category", ["Simplify Content", "Risk Analytics","Spatial Analytics","Payments/Pricing Analytics","Stakeholders/Parties Analytics","Search Content"])
  search_text = st.sidebar.text_input("Text to search")
  action_button=st.sidebar.button('Analyze Contract')
   
  if file_upload:
    text=[]
    pdfbytes = file_upload.getvalue()
    doc = return_doc_from_bytes(pdfbytes)
    
    for page in doc:
      text+=(page.get_text().split('\n'))
    cleaned_document=preprocessing(text)
    clean_text=data_string(cleaned_document)
    
    if select_category == "Search Content":
      if search_text:
        result=search_report(cleaned_document,search_text.lower())
        st.header('Related information linked to Search Content')
        st.write(result)
    
    if select_category == "Simplify Content":
      if action_button:
        st.header('Simplifying the content as Word Cloud..')
        wordcloud = WordCloud(width = 800, height =600,background_color ='white',min_font_size = 5,max_words=500).generate(clean_text)
        # plot the WordCloud image
        plt.figure(figsize = (15,10), facecolor = None)
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
        st.pyplot(fig=plt)
        
    if select_category == "Payments/Pricing Analytics":
      if action_button:
        data = [[ 66386, 174296,  75131, 577908,  32015],
                [ 58230, 381139,  78045,  99308, 160454],
                [ 89135,  80552, 152558, 497981, 603535],
                [ 78415,  81858, 150656, 193263,  69638],
                [139361, 331509, 343164, 781380,  52269]]

        columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
        rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

        values = np.arange(0, 2500, 500)
        value_increment = 1000

        # Get some pastel shades for the colors
        colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
        n_rows = len(data)

        index = np.arange(len(columns)) + 0.3
        bar_width = 0.4

        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = np.zeros(len(columns))

        # Plot bars and create text labels for the table
        cell_text = []
        for row in range(n_rows):
            plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset = y_offset + data[row]
            cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
        # Reverse colors and text labels to display the last value at the top.
        colors = colors[::-1]
        cell_text.reverse()

        # Add a table at the bottom of the axes
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)

        plt.ylabel("Loss in ${0}'s".format(value_increment))
        plt.yticks(values * value_increment, ['%d' % val for val in values])
        plt.xticks([])
        plt.title('Loss by Disaster')

        plt.show()
        st.pyplot(fig=plt)
        
    if select_category == "Risk Analytics":
      if action_button:
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

        st.header('Listing all Risk Areas as Word Cloud..')
        wordcloud = WordCloud(width=800,height=800,background_color='white').generate_from_frequencies(a)
        # plot the WordCloud image
        plt.figure(figsize = (8,8), facecolor = None)
        plt.imshow(wordcloud,interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()
        st.pyplot(fig=plt)


__author__ = "Kalpana D"
__version__ = "1.0.1"
__maintainer__ = "Kalpana D"
__status__ = "Development"
__description__ = "Invoking the Primary UIX"

if __name__ == "__main__":
    st_ui()
