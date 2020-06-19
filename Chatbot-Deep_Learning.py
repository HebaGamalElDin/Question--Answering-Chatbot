# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:47:58 2020

@author: Heba Gamal El-Din
"""
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import string
import time
import datetime
import wikipedia as wk
import re
 ####################
 ## Data Creation ##
##################
"""record = {}
record['Data'] = []

def Data_(Class, Patterns, Responses, Context):
    record['Data'].append({'Tag': str(Class).lower(), 'Patterns': [Expr.strip().lower() for Expr in Patterns.split(',')], 
   'Responses': [Expr.strip().lower() for Expr in Responses.split(',')], 'Context': [Expr.strip().lower() for Expr in Context.split(',')]})
    with open('Data.txt', 'w') as OutFile:
        json.dump(record, OutFile)
    return "Done Created."

Flag = True
while(Flag):
    Class = input('Class: ')
    Patterns = input('Patterns: ')
    Responses = input('Responses: ')
    Context = input('Context: ')
    MSG = Data_(Class, Patterns, Responses, Context)
    if MSG == "Done Created.":
        Terminate = input("Done?! ")
        if Terminate == 'Yes':
            Flag = False
            """
##############################################################################
##############################################################################
            
with open('Data.txt') as JSON:
    Data = json.loads(JSON.read())
    
Uniques=[]
Classes = []
Labeling = []
Punc = []
Punc.extend(string.punctuation)
for record in Data['Data']:
    for Pattern in record['Patterns']:
        Tokens = nltk.word_tokenize(Pattern)
        Uniques.extend(Tokens)
        Labeling.append((Tokens, record['Tag']))
        if record['Tag'] not in Classes:
            Classes.append(record['Tag'])

Lemmatizer = WordNetLemmatizer()
Words = [Lemmatizer.lemmatize(Token.lower()) for Token in Uniques if Token not in Punc]
Words = sorted(list(set(Words)))
classes = sorted(list(set(Classes)))
print (len(Labeling), "documents")
print (len(Classes), "classes", classes)
print (len(Words), "unique lemmatized words", Words)
pickle.dump(Words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#################################
## Preparing The Training Set ##
###############################
Training_Set = []
Predicted_Class = [0] * len(classes)
for Rec in Labeling:
    Matrix = []
    Pattern = Rec[0]
    Pattern = [Lemmatizer.lemmatize(word.lower()) for word in Pattern]
    for word in Words:
        Matrix.append(1) if word in Pattern else Matrix.append(0)
    output_row = list(Predicted_Class)
    print(Rec)
    output_row[classes.index(Rec[1])] = 1
    Training_Set.append([Matrix, output_row])
random.shuffle(Training_Set)
Training_Set = np.array(Training_Set)
X_train = list(Training_Set[:,0])
Y_train = list(Training_Set[:,1])
print("Training Data has been Created!")

#####################
## Model Building ##
###################

Model = Sequential()
Model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(64, activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(len(Y_train[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
Model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = Model.fit(np.array(X_train), np.array(Y_train), epochs=100, batch_size=5, verbose=1)
Model.save('chatbot_Model.h5', hist)
print("Model has been Created!")

###############################
""" User Input Preparation """
###############################
def Input_Cleansing(User_Input):
    Tokens = nltk.word_tokenize(User_Input)
    Sent_Tokens = [Lemmatizer.lemmatize(word.lower()) for word in Tokens if word not in Punc]
    return Sent_Tokens

def Matrix_Generation(User_Input, Words, show_details=True):
    Sent_Tokens = Input_Cleansing(User_Input)
    Matrix_ = [0]*len(Words)
    for Sent in Sent_Tokens:
        for i,w in enumerate(Words):
            if w == Sent:
                Matrix_[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(Matrix_))

def Class_Prediction(User_Input, Model):
    matrix = Matrix_Generation(User_Input, Words,show_details=False)
    Output = Model.predict(np.array([matrix]))[0]
    ERROR_THRESHOLD = 0.25
    print(Output)
    results = [[i,r] for i,r in enumerate(Output) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"Class": classes[r[0]], "Probability": str(r[1])})
        print(return_list)
    return return_list

def Response_(Class, Data):
    Tag = Class[0]['Class']
    Doc = Data['Data']
    for i in Doc:
        if(i['Tag']== Tag):
            result = random.choice(i['Responses'])
            break
    return str(result).title()


##############################
## Wikipedia search Function ##
###############################
def wikipedia_data(input):
    Quest = ["tell me about","get for me data about", "provide me data about", "what is the", "talk to me about"]
    topic = [input[input.index(expr) + len(expr) :].strip() for expr in Quest if expr in input]
    try:
        if topic:
            wiki = "Wait for Me just Seconds! ... \n" + wk.summary(topic[0], sentences = 3)
            if (wiki != None):
                return wiki
    except Exception:
            print("I Don't Know About This Topic Sorry :( ")

###########################
## Date \ Time Responses ##
###########################
Formula = ["what's the time?","Do you have the time?".lower(),"Do you know what time it is?".lower(),"What time is it?".lower(),"Can you tell me what time it is, please?".lower()]
Formula1 = ["What's today?".lower(),"What day is it today?".lower(),"What's the date?".lower()]
def Date_Time(input):        
    if input.lower().strip() in Formula1:
        Date = datetime.datetime.now()
        Day = Date.strftime("%A")
        return Day
    elif input in Formula:
        Date = datetime.datetime.now()
        Time = Date.strftime("%I:%M:%S %p")
        return Time

######################
""" Base Function """
######################

def chatbot_response(Input):
    if wikipedia_data(Input) != None:
        Response = wikipedia_data(Input)
    elif Date_Time(Input) != None:
        Response = Date_Time(Input)
    else:
        Class = Class_Prediction(Input, Model)
        Response = Response_(Class, Data)
    return Response

#####################
    """ GUI """
#####################
import tkinter
from tkinter import *

def send():
    Input_ = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if Input_ != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + Input_ + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(Input_)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Chetty")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()