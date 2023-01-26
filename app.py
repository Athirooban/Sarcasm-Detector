import pandas as pd
import re
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from flask import request
from flask import jsonify
from flask import Flask, render_template
import mysql.connector

app = Flask(__name__)

sarcasm_data1 = pd.read_csv("./data/data.csv")
sarcasm_data2 = pd.read_csv("./data/train-balanced-sarcasm.csv")
sarcasm_data2.drop(['author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'], axis=1, inplace=True)
sarcasm_data2.dropna(inplace=True)

print('sarcasm1: ', sarcasm_data1.shape)
print('sarcasm2: ', sarcasm_data2.shape)

mispell_dict = {"ain't": "is not", "cannot": "can not", "aren't": "are not", "can't": "can not", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "wont": "will not", "won't've": "will not have", "would've": "would have",
                "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color',
                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                'theBest': 'the best', 'howdoes': 'how does', 'Etherium': 'Ethereum',
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
                'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

mispell_dict = {k.lower(): v.lower() for k, v in mispell_dict.items()}

def preprocessing_text(s):
    # making our string lowercase & removing extra spaces
    s = str(s).lower().strip()
    
    # remove contractions.
    s = " ".join([mispell_dict[word] if word in mispell_dict.keys() else word for word in s.split()])
    
    # removing \n
    s = re.sub('\n', '', s)
    
    # put spaces before & after punctuations to make words seprate. Like "king?" to "king", "?".
    s = re.sub(r"([?!,+=—&%\'\";:¿।।।|\(\){}\[\]//])", r" \1 ", s)
    
    # Remove more than 2 continues spaces with 1 space.
    s = re.sub('[ ]{2,}', ' ', s).strip()
    
    return s

frames = [sarcasm_data1, sarcasm_data2]
sarcasm_data = pd.concat(frames, sort=True)

# apply preprocessing_text function
sarcasm_data['comment'] = sarcasm_data['comment'].apply(preprocessing_text)
print('Shape: ', sarcasm_data.shape)

print('Dataset Loaded')
training_data = sarcasm_data[:1066096]
testing_data = sarcasm_data[-5:]

# total unique words we are going to use.
TOTAL_WORDS = 40000

# max number of words one sentence can have
MAX_LEN = 50

tokenizer = Tokenizer(num_words=TOTAL_WORDS)
tokenizer.fit_on_texts(list(training_data['comment']))
print('Tokenizer Loaded')

new_model = load_model('./model/final_model.h5')
print('LSTM Model Loaded')

now = datetime.now()
formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
print('Current date: ', formatted_date)

@app.route('/')
def home():
    return render_template('detect.html')

@app.route('/detect',methods=['POST'])
def detect():    
    message = request.get_json(force=True)
    name = message['sentence']
    # print('Previous sentence:', name)
    sentence = preprocessing_text(name)
    print('Preprocessed sentence:', sentence)
    print('sentence length:', len(sentence))
    if (len(sentence) >= 8):
        token = tokenizer.texts_to_sequences([sentence])
        # print('Token: ', token)
        pad = pad_sequences(token, maxlen = MAX_LEN)
        # print('Pad: ', pad)
        prediction = new_model.predict(pad)
        print('Prediction:', prediction)
    
        result = prediction[0][0] * 100
        result = float(result)
        ## Sarcasm code lines added without using the Database.
        print("result: ", result)
        response = {
            # 'greeting': "Input Sentence: " + sentence
            'prediction': str(prediction[0][0])
            }
        print("response: ", response)
        '''
        query = "INSERT INTO sarcastic_table(sentence, result, created_at) VALUES (%s, %s, %s)"
        params = (sentence, result, formatted_date)
        
        connection = mysql.connector.connect(host="localhost", user="root", password="cicadmin@123", database="sarcastic_database")
        cursor = connection.cursor()

        dbresult = db_insert(query, params, cursor)
        print('db_result: ', dbresult)
        
        response = db_result(dbresult, prediction)
        print('db_response: ', response)
        connection.commit()
        cursor.close()
        connection.close()
        '''
        
    else:
        print('Please enter more than one word!')
        response = {'prediction': str(500) }
    return jsonify(response) 

def db_insert(query, params, cur):
    result = cur.execute(query, params)
    if (result==None):
        return True
    else:
        return False
        
def db_result(dbresult, prediction):
    if (dbresult == True):
        print('Database Result: ', dbresult)
        response = {
            # 'greeting': "Input Sentence: " + sentence
            'prediction': str(prediction[0][0])
            }
        print('result:', response)
    else:
        response = {'prediction': str(600)}
    return response 

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)