import numpy as np
import math

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def tokenize(email):
    return email.lower().split() #["free","you","i"] ex.

def term_freq(L):
    tf={}
    for word in L:
        tf[word]=tf.get(word,0)+1
    return tf

def vectorize(vocab,L,df,tf,N):
    vec=np.zeros(len(vocab)) # one heat enc feature vector
    
    for word in tf:
        if word not in vocab:
            continue
        term_freq = tf[word] / len(L)
        idf = math.log2(N / df[word])
        vec[vocab[word]]=term_freq*idf
    
    return vec

    
corpus = [
"free mobile phone offer",
"win free cash now",
"claim your lottery prize today",
"exclusive deal buy one get one free",
"urgent you won a free vacation",
"limited offer free gift card",

"are you free for meeting tomorrow",
"let us schedule the project call",
"please review the attached report",
"meeting rescheduled to 3 pm",
"can we discuss the assignment",
"see you at lunch today",

"congratulations you won 500000 cash",
"claim your reward now",
"free entry in lucky draw",
"special discount available today",

"team meeting at 10 am",
"project deadline is tomorrow",
"please send the documents",
"let us finalize the presentation"
]

N=len(corpus) #we have 3 documents

y = np.array([
1,1,1,1,1,1,
0,0,0,0,0,0,
1,1,1,1,
0,0,0,0
]) #spam , not spam, spam

vocab={}
index=0
for email in corpus:
    tokens=tokenize(email)
    for word in tokens:
        if word not in vocab:
            vocab.setdefault(word,index)
            index+=1


#bag of words is literally the frequency of eahc word in vocab across documents -1 way of vectorization
#tfidf is giving more imprtance to rare words
#tf = freq * idf= log(N/df)

df={}


for email in corpus:
    L=set(tokenize(email))
    for word in L:
        df[word] = df.get(word,0) + 1
                
tf_idf_vec=[]
for email in corpus:
    tf={}
    L=tokenize(email)
    tf=term_freq(L)
    tf_idf_vec.append(vectorize(vocab,L,df,tf,N))

x=np.array(tf_idf_vec)
# shape = no of mails x vocab length
# print(vectorized_words.shape) 
# 3 x 10

n_samples,n_features=x.shape
epoch=1000
w=np.zeros(n_features)
b=0.0
lr=0.01

for _ in range(epoch):
    
    z = x @ w + b
    y_p = sigmoid(z)
    error=y_p - y
    dw = (x.T @ error)/n_samples
    db = np.mean(error)
    w-= lr*dw
    b-= lr*db
    
#-------------------------predicting------------------
def predict(message):
    L=tokenize(message)
    tf=term_freq(L)
    x=vectorize(vocab,L,df,tf,N)
    z=x@w +b
    prob=sigmoid(z)
    if prob>0.5:
        return "spam",prob
    else:
        return "not spam",prob
#------------------------test--------------------------
test1="please transfer moeny to my bank account"
test2="you won 10000 cash"
test3="you have bitcoin in your bank account"

print(f"'{test1}' -> {predict(test1)}")
