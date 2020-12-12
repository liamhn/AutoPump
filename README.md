# AutoPump
AutoPump was an idea for a generative neural network model that is trained on song lyrics and creates new song lyrics of the same style. The name AutoPump came from the original intention to do this project with lyrics from the artist Lil' Pump. We will start with a naive LSTM model that takes n words as input data, and tries to predict the n+1 word as output data

## Getting The Data
Before we construct our neural network model, we first need to acquire some data. We are interested in training hte neural netowrk on lyrics written by Lil' Pump. To get thes lyrics, we use the lyricsgenius api, which can be installed with  
  
```pip install lyricsgenius```  
  
The lyrics genius api makes it easy to rip lyrics a json file from the popular lyrics website _genius.com_.  
We start by importing lyricsgenius  
```python
import lyricsgenius
```
Next, we instantiate our genius class -- it needs a client token to work (you can get one from the genius api website)
```python
client_token = 'your client token here'
genius = lyricsgenius.Genius(client_token)
```
We next pick an artist and save their lyrics, here we use Lil Pump, and we choose to sort by title
```python
artist_name = "LilPump"
artist_tag=artist_name.replace(" ","")
```
Check if we have already downloaded this data, skip download if you have
```python
if glob.glob("Lyrics_"+artist_tag+".json")==[]:
    artist = genius.search_artist(artist_name, sort="title")
    artist.save_lyrics()
```
    
## Cleaning The Data
Our text data is now stored in a .json file, and we want to process this file and prepare it for submission to a neural net.  
We start by globbing and loading in all the json files using pandas.
```python
lyric_files = glob.glob("Lyrics_"+artist_tag+".json")

df = pd.DataFrame()
for i in range(len(lyric_files)):
    predf = pd.read_json(lyric_files[i],orient='index',typ='series')
    df = df.append(predf.songs)
```
For good taste, we title the columns and save it as a csv.  
```python
data = df[['title','lyrics']]
data.sample(3)

data.to_csv('lyrics_titles_'+artist_tag+'.csv')
```

We compile all the lyrics into one string
```python
corpus0 = ""
for row in data.itertuples():
    text = row.lyrics
    if type(text) == str:
        corpus0+=text
```
We remove unwanted strings, space punctuation away from words, and lowercase the entire data set.  
```python
corpus0 = corpus0.replace('[verse]' ,'')
corpus0 = corpus0.replace('[intro]', '')
corpus0 = corpus0.replace('[outro]', '')
corpus0 = corpus0.replace('[bridge]', '')
corpus0 = corpus0.replace('[chorus]' ,'')
corpus0 = corpus0.replace('[Intro]', '')
corpus0 = corpus0.replace('[Outro]', '')
corpus0 = corpus0.replace('[Bridge]', '')
corpus0 = corpus0.replace('[Chorus]', '')
corpus0 = corpus0.replace('[verse 1]', '')
corpus0 = corpus0.replace('[verse 2]', '')
corpus0 = corpus0.replace('[verse 3]', '')
corpus0 = corpus0.replace('[verse 4]', '')
corpus0 = corpus0.replace('Lyrics', '')

corpus0 = corpus0.replace(',', ' , ')
corpus0 = corpus0.replace('(', ' , ')
corpus0 = corpus0.replace(')', ' ) ')
corpus0 = corpus0.replace('[', ' [ ')
corpus0 = corpus0.replace(']', ' ] ')
corpus0 = corpus0.replace('.', ' . ')
corpus0 = corpus0.replace(';', ' ; ')
corpus0 = corpus0.replace(':', ' : ')
corpus0 = corpus0.replace('!', ' ! ')
corpus0 = corpus0.replace('?', ' ? ')
corpus0 = corpus0.replace('*', ' * ')
corpus0 = corpus0.replace("’", '\'')
corpus0 = corpus0.replace("\'\'", ' " ')
corpus0 = corpus0.replace('"', ' " ')
corpus0 = corpus0.replace("'", " ' ")
corpus0 = corpus0.replace('\r\n', ' \r\n ')
corpus0 = corpus0.replace('-', ' - ')
corpus0 = corpus0.replace('\n', ' \n ')
corpus0 = corpus0.replace('\u2005', ' ')
corpus0 = corpus0.replace('\u205f', ' ')
corpus0 = corpus0.replace('—', ' — ')
corpus0 = corpus0.replace('¿', ' ¿ ')
corpus0 = corpus0.replace('¡', ' ¡ ')

corpus0 = corpus0.lower()
```
We split the corpus up by word, such that we have a list containing each word in the corpus in order. We also remove some unwated empty strings.  
```python
corpus = corpus0.split(' ')
while (corpus.count('') > 0): 
    corpus.remove('')
```
Next, we want to convert these words to numbers, we use a simple encoding -- one integer for each unique word in the corpus. For this naive implementation, we use no embedding.
```python
words = sorted(list(set(corpus)))
num_words = len(words)

encoding = {w: i for i, w in enumerate(words)}
decoding = {i: w for i, w in enumerate(words)}
```
The corpus for all of Lil Pump's lyrics contains 2982 unique words. We have our data in the form of a list of cleaned words. Next we need to construct our model and prepare the data to be fed into it.  
## Constructing The Model  
We will treat this as a classification problem. The data we are trying to "classify" will be a sequence of text, and the classification categories will the set of unique words in the corpus. Our model will take a sequence of words, and try to predict the next word. We can split our corpus into many sequences of text (as x data), and the following word (as y data). Then it is a simple matter of training a neural network to be able to classify each sequence into the appropriate class (i.e. predict the next word). Here we get out first hyper parameter -- the length of each sequence of x-data. We will experiment with this value bit, but we start out with a sentence length of 10 words -- a reasonable estimate for the length of a line of a song. We construct lists to hold the x_data nd y_data, as encoded lists of 10 word sequences, and the next encoded word, respectively.  

```python
# We will call each sequence of 50 words a "sentence"
sentence_length = 10

# Map the entire corpus to N sentences of 50 words each #

# Initialize empty lists to store the data
x_data = []
y_data = []

# Loop over the corpus, take each sentence_length word sequence, encode it, and save it in x_data
# Take each sentence_length+1st word, encode it, and save it to y_data 
for i in range(0, len(corpus) - sentence_length):
    sentence = corpus[i: i + sentence_length]
    next_word = corpus[i + sentence_length]
    x_data.append([encoding[word] for word in sentence])
    y_data.append(encoding[next_word])
```
In the absence of an embedding, we one hot encode our words, and construct sentences as matrices made up of stacked, one-hot-encoded word vectors.   
```python
x = np.zeros((num_sentences, sentence_length, num_words), dtype = np.bool)
y = np.zeros((num_sentences, num_words), dtype = np.bool)

for i, sentence in enumerate(x_data):
    for t, encoded_word in enumerate(sentence):
        x[i, t, encoded_word] = 1
    y[i, y_data[i]] = 1
```


Our neural network architecture is as follows.  
- Input layer. The input layer's shape the number of unique words by the number of words per "sentence". It is shaped to the x-data (which is a vertically stacked matrix of 1-hot encoded word vectors.  
- LSTM layer. Since we are dealing with sequential data, we use an LSTM. Each LSTM node contains a "memory cell" which "remembers" the previous words in the sentence. We apply dropout after the LSTM layer to try to avoid overfitting (which turns out to be a bit of a futile task -- we simply don't have enough data using jsut Lil' Pump lyrics).
- Output layer. The number of nodes in our output layer is equal to the total number of unique words in our corpus we want the output to be one of these words. We use softmax as our activation function for this layer since we are essentially treating this as a categorical classification problem (where we are trying to classify each sentence_length sequence in the the "category" given by the next word).  

We use categorical crossentropy as our loss function and optimize with RMSprop.


![AutoPumpArch](https://github.com/liamhn/AutoPump/blob/main/AutoPump%20Architecture.png?raw=true)  
Or in code, 
```python
n_LSTM = 256
drop_rate=0.5
model = km.Sequential()
model.add(kl.Bidirectional(kl.LSTM(n_LSTM, return_sequences=False), input_shape = (sentence_length, num_words)))
model.add(kl.Dropout(drop_rate))
model.add(kl.Dense(num_words, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'],)

fit = model.fit(x, y, epochs = 25, batch_size = 128,validation_split=.1,verbose=2)
```

There are quite a few hyperparameters in our model (and in our data) that we can play with. In this case, overfitting is a huge problem, since we have an extremely limited amount of data. The number of different ways any given word in the english language can be used is essentially limitless, and training a model on only a few examples of those uses is sure to lead to either overfitting (in the case of high training accuracy), or just low test accuracy. Nevertheless, we will play with these hyper parameters to try to optimize our model on the validation set. We loop over a few different combinations of hyperparameters and compare the resulting validation accuracy after 50 epochs of training.  




