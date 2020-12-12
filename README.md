# AutoPump
AutoPump was an idea for a generative neural network model that is trained on song lyrics and creates new song lyrics of the same style. The name AutoPump came from the original intention to do this project with lyrics from the artist Lil' Pump. We will start with a naive LSTM model that takes n words as input data, and tries to predict the n+1 word as output data

## Getting The Data
Before we construct our neural network model, we first need to acquire some data. We are interested in training hte neural netowrk on lyrics written by Lil' Pump. To get thes lyrics, we use the lyricsgenius api, which can be installed with  
  
```pip install lyricsgenius```  
  
The lyrics genius api makes it easy to rip lyrics a json file from the popular lyrics website _genius.com_.  
We start by importing lyricsgenius  
```
import lyricsgenius
```
Next, we input instantiate our genius class -- it needs a client token to work (you can get one from the genius api website)
```
client_token = 'your client token here'
genius = lyricsgenius.Genius(client_token)
```
We next pick an artist and save their lyrics, here we use Lil Pump, and we choose to sort by title
```
artist_name = "LilPump"
artist_tag=artist_name.replace(" ","")
```
Check if we have already downloaded this data, skip download if you have
```
if glob.glob("Lyrics_"+artist_tag+".json")==[]:
    artist = genius.search_artist(artist_name, sort="title")
    artist.save_lyrics()
```
    
## Cleaning The Data
Our text data is now stored in a .json file, and we want to process this file and prepare it for submission to a neural net.  
We start by globbing and loading in all the json files using pandas.
```
lyric_files = glob.glob("Lyrics_"+artist_tag+".json")

df = pd.DataFrame()
for i in range(len(lyric_files)):
    predf = pd.read_json(lyric_files[i],orient='index',typ='series')
    df = df.append(predf.songs)
```
For good taste, we title the columns and save it as a csv.  
```
data = df[['title','lyrics']]
data.sample(3)

data.to_csv('lyrics_titles_'+artist_tag+'.csv')
```
We compile all the lyrics into one string
```
corpus0 = ""
for row in data.itertuples():
    text = row.lyrics
    if type(text) == str:
        corpus0+=text
```














