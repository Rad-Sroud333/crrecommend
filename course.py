import pandas as pd
import numpy as np
import nltk
#nltk.download('stopwords')
from nltk import word_tokenize
import string
nltk.download('punkt')
import contractions
#stopword=stopwords.words("english")
#exclude=string.punctuation
exclude='''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
from annoy import AnnoyIndex
import pickle

stopword=pickle.load(open("stopwords.pkl","rb"))

exclude=string.punctuation
def remove_punc(s):
    s=str(s)
    txt=s.translate(str.maketrans("","",exclude))
    return txt

def tokenize(col):
    #col=" ".join(col)
    words=word_tokenize(col)
    txt= " ".join(words)
    return txt

def remove_num_words(s):
    l=[]
    for i in s.split():
        if i.isdigit():
            l.append("")
        else:
            l.append(i)
    txt= (" ").join(l) 
    return txt       

def stop_words(txt):
    l=[]
    for i in txt.split():
        if i in stopword:
            l.append("")
        else:
            l.append(i)
    x=l[:]
    l.clear()
    txt=" ".join(x)
    txt=txt.lower()
    return txt          

def cont(text):
    expanded_words = []   
    for word in text.split():
        expanded_words.append(contractions.fix(word))  
    expanded_text = ' '.join(expanded_words)
    return expanded_text  


def clean_text(txt):
    try:
        txt=cont(txt)
        txt=tokenize(txt)
        txt=remove_num_words(txt)
        txt=stop_words(txt)
        txt=remove_punc(txt)
        return txt
    except (AttributeError,ValueError,TypeError):
        return "Type Correctly"


model=pickle.load(open("model.pkl","rb"))
df=pickle.load(open("df.pkl","rb"))
#print(df.head(1))

def doc_vector(review):
    l=[]
    for i in review.split():
        if i in model.wv.index_to_key:
            l.append(i)
            
    if len(l)>1:
            return np.mean(model.wv[l],axis=0)
    else:
            return model.wv[l][0]
          
def arr_user_input_movie(course_name):
        course_name=clean_text(course_name)
        vec=doc_vector(course_name)
        return(vec)

def recommend1(course_name):
    #try: 
            inp=arr_user_input_movie(course_name)                                                                                                                                                                   
            f = 100
            u = AnnoyIndex(f, 'angular')
            u.load("m1.ann")
            re=u.get_nns_by_vector(inp,n=5,search_k=1000,include_distances=True)
            l=[]
            for i in re[0]:
                course_name=df.iloc[i]["Course Name"]
                university=df.iloc[i]['University']
                difficulty=df.iloc[i]['Difficulty Level']
                course_rating=df.iloc[i]['Course Rating']
                course_url=df.iloc[i]['Course URL']
                skills=df.iloc[i]['Skills']
                l.append([course_name,university,difficulty,course_rating,course_url,skills]) 
            return l    

    #except ValueError:
            return "use more appropriate keywords"

#------------------------#---------------------#--------------------#---------------------#--------------------->

from flask import Flask,render_template,request
app = Flask(__name__)

@app.route('/')
def recommnd():
        return render_template('recommend.html')  

@app.route('/recommend_course',methods=['GET','POST'])
def recommend_course():
    if request.method == 'POST':
        user_input=request.form.get('user_input')
        #print("USER INPUT IS",user_input)
        try:
            recommend=recommend1(user_input)
            #print(1)
            return render_template('recommend.html',recommend=recommend)   
        except ValueError:
            #print(2)
            return  render_template('recommend.html',error="use more appropriate keywords" )  
    else:
           return render_template('recommend.html')     
     
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)    
