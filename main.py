import re
import string
import nltk
import streamlit as st
nltk.download('all')
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords  #stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#for adding daisi dependencies
# import pydaisi as pyd
# stable_diffusion = pyd.Daisi("laiglejm/Stable Diffusion")

def preprocessing(sentences):
  documents_clean = ''
  for d in sentences:
    # Remove Unicode
    document_test = re.sub('[^a-zA-Z0-9]', ' ', d)
    # Lowercase the document
    document_test = document_test.lower()
    # Remove punctuations
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    #Remove the numbers
    document_test = re.sub(r'[0-9]', '', document_test)
    # Remove the doubled space
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    #tokenization
    document_test = document_test.split()   
    #stopwords_removal
    document_test = [word for word in document_test if not word in set(stopwords.words('english'))]
    #stemming
    #document_test = [stemmer.stem(word) for word in document_test]
    #lemmmitization
    document_test = [lemmatizer.lemmatize(word) for word in document_test]
    document_test = ' '.join(document_test)
    documents_clean+=(document_test)
  return documents_clean
  # print(documents_clean)
def st_ui():
  st.set_page_config(layout = "wide")
  st.title("WordCloud of supplier contract")
#   Button=st.sidebar.button('Generate wordcloud')
#   if Button:
  text="""Word Clouds came out to be a game-changer visualization technique for understanding and determining patterns and evolving trends. 
          Whether to discover the political agendas of aspiring election candidates of a country or to analyze the customer reviews on the recently 
          launched product, one can get a visual representation by plotting the Word Cloud Word Cloud or Tag Clouds is a visualization technique for 
          texts that are natively used for visualizing the tags or keywords from the websites. These keywords typically are single words that depict the context 
          of the webpage the word cloud is being made from. These words are clustered together to form a Word Cloud Each word in this cloud has a variable font size 
          and color tone. Thus, this representation helps to determine words of prominence. A bigger font size of a word portrays its prominence more relative to 
          other words in the cluster. Word Cloud can be built in varying shapes and sizes based on the creators’ vision. The number of words plays an important role
          while creating a Word Cloud. More number of words does not always mean a better Word Cloud as it becomes cluttery and difficult to read. A Word Cloud must 
          always be semantically meaningful and must represent what it is meant for.Although, there are different ways by which Word Clouds can be created but the 
          most widely used type is by using the Frequency of Words in our corpus. And thus, we will be creating our Word Cloud by using the Frequency type.
          The history of Word Clouds dated back to 1976 when an American Social Psychologist Stanley Milgram conducted a Psychological Study and asked people about 
          the places from Paris. The main idea was to build a mental map of Paris when the people were asked about the city. He analyzed and drew a map based on the 
          responses received from people and kept a bigger font size for the frequently received responses The Tag Clouds came into the limelight when Flickr, a 
          photo-sharing website, in 2006 started using Tag Clouds for site exploration. By the end of the first decade of the 21st century, Word Cloud become a very 
          popular tool among the text miners.But, the trend of Tag Cloud keeps vacillating and eventually started declining over the period of time. And thus, Word 
          Clouds are being popularly used in today’s world."""
  sentences = nltk.sent_tokenize(text)
  documents_clean=preprocessing(sentences)  
  wordcloud = WordCloud(width = 800, height =600,background_color ='white',min_font_size = 5,max_words=500).generate(documents_clean)
  # plot the WordCloud image                      
  plt.figure(figsize = (15,10), facecolor = None)
  plt.imshow(wordcloud,interpolation="bilinear")
  plt.axis("off")
  plt.tight_layout(pad = 0)
  plt.show()

    
 

if __name__ == "__main__":
    st_ui()
