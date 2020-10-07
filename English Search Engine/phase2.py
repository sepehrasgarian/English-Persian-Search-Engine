# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:03:04 2020

@author: Dr Asgarian
"""
from hazm import *
import re 
import pickle 
import math
import sys 
         
import tkinter as tk
import string
import re
stemmer = Stemmer()
lemmatiz = Lemmatizer()
K=10
import pandas as pd 
import collections
from PyQt5.QtWidgets import QWidget,QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem
import pandas as pd
%gui qt5 

class MaxHeap: 
  
    def __init__(self, maxsize): 
        self.maxsize = maxsize 
        self.size = 0
        self.Heap = [0]*(self.maxsize + 1) 
        self.Heap[0] = sys.maxsize 
        self.FRONT = 1
  
    # Function to return the position of 
    # parent for the node currently 
    # at pos 
    def parent(self, pos): 
        return pos//2
  
    # Function to return the position of 
    # the left child for the node currently 
    # at pos 
    def leftChild(self, pos): 
        return 2 * pos 
  
    # Function to return the position of 
    # the right child for the node currently 
    # at pos 
    def rightChild(self, pos): 
        return (2 * pos) + 1
  
    # Function that returns true if the passed 
    # node is a leaf node 
    def isLeaf(self, pos): 
        if pos >= (self.size//2) and pos <= self.size: 
            return True
        return False
  
    # Function to swap two nodes of the heap 
    def swap(self, fpos, spos): 
        self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos] 
  
    # Function to heapify the node at pos 
    def maxHeapify(self, pos): 
  
        # If the node is a non-leaf node and smaller 
        # than any of its child 
        if not self.isLeaf(pos): 
            if (self.Heap[pos] < self.Heap[self.leftChild(pos)] or
                self.Heap[pos] < self.Heap[self.rightChild(pos)]): 
  
                # Swap with the left child and heapify 
                # the left child 
                if self.Heap[self.leftChild(pos)] > self.Heap[self.rightChild(pos)]: 
                    self.swap(pos, self.leftChild(pos)) 
                    self.maxHeapify(self.leftChild(pos)) 
  
                # Swap with the right child and heapify 
                # the right child 
                else: 
                    self.swap(pos, self.rightChild(pos)) 
                    self.maxHeapify(self.rightChild(pos)) 
  
    # Function to insert a node into the heap 
    def insert(self, element): 
        if self.size >= self.maxsize : 
            return
        self.size+= 1
        self.Heap[self.size] = element 
  
        current = self.size 
  
        while self.Heap[current] > self.Heap[self.parent(current)]: 
            self.swap(current, self.parent(current)) 
            current = self.parent(current) 
  
    # Function to print the contents of the heap 
    def Print(self): 
        for i in range(1, (self.size//2)+1): 
            print(" PARENT : "+str(self.Heap[i])+" LEFT CHILD : "+ 
                               str(self.Heap[2 * i])+" RIGHT CHILD : "+
                               str(self.Heap[2 * i + 1])) 
  
    # Function to remove and return the maximum 
    # element from the heap 
    def extractMax(self): 
  
        popped = self.Heap[self.FRONT] 
        self.Heap[self.FRONT] = self.Heap[self.size] 
        self.size-= 1
        self.maxHeapify(self.FRONT) 
        return popped 
    
    
    
def cheak_spell_arabic(text):
    aftery = re.sub("ء", "ئ", text)
    aftera = re.sub(r"[ٲٱإﺍأ]", r"ا", aftery)
    abfterb = re.sub(r"[ﺐﺏﺑ]", r"ب", aftera)
    afterp = re.sub(r"[ﭖﭗﭙﺒﭘ]", r"پ", abfterb)
    aftert = re.sub(r"[ﭡٺٹﭞٿټﺕﺗﺖﺘ]", r"ت", afterp)
    afterc = re.sub(r"[ﺙﺛ]", r"ث", aftert)
    afterj = re.sub(r"[ﺝڃﺠﺟ]", r"ج", afterc)
    afterch = re.sub(r"[ڃﭽﭼ]", r"چ", afterj)
    afterh = re.sub(r"[ﺢﺤڅځﺣ]", r"ح", afterch)
    afterkh = re.sub(r"[ﺥﺦﺨﺧ]", r"خ", afterh)
    afterd = re.sub(r"[ڏډﺪﺩ]", r"د", afterkh)
    afterz = re.sub(r"[ﺫﺬﻧ]", r"ذ", afterd)
    afterr = re.sub(r"[ڙڗڒڑڕﺭﺮ]", r"ر", afterz)
    afterzi = re.sub(r"[ﺰﺯ]", r"ز", afterr)
    afterzh = re.sub(r"ﮊ", r"ژ", afterzi)
    aftersin = re.sub(r"[ݭݜﺱﺲښﺴﺳ]", r"س", afterzh)
    aftersh = re.sub(r"[ﺵﺶﺸﺷ]", r"ش", aftersin)
    aftersad = re.sub(r"[ﺺﺼﺻ]", r"ص", aftersh)
    afterzad = re.sub(r"[ﺽﺾﺿﻀ]", r"ض", aftersad)
    afterta = re.sub(r"[ﻁﻂﻃﻄ]", r"ط", afterzad)
    afterza = re.sub(r"[ﻆﻇﻈ]", r"ظ", afterta)
    afterein = re.sub(r"[ڠﻉﻊﻋ]", r"ع", afterza)
    afterghein = re.sub(r"[ﻎۼﻍﻐﻏ]", r"غ", afterein)
    afterf = re.sub(r"[ﻒﻑﻔﻓ]", r"ف", afterghein)
    afterghaf = re.sub(r"[ﻕڤﻖﻗ]", r"ق", afterf)
    afterkaf = re.sub(r"[ڭﻚﮎﻜﮏګﻛﮑﮐڪك]", r"ک", afterghaf)
    aftergaf = re.sub(r"[ﮚﮒﮓﮕﮔ]", r"گ", afterkaf)
    afterlam = re.sub(r"[ﻝﻞﻠڵ]", r"ل", aftergaf)
    aftermim = re.sub(r"[ﻡﻤﻢﻣ]", r"م", afterlam)
    afternun = re.sub(r"[ڼﻦﻥﻨ]", r"ن", aftermim)
    aftervav = re.sub(r"[ވﯙۈۋﺆۊۇۏۅۉﻭﻮؤ]", r"و", afternun)
    afterhe = re.sub(r"[ﺔﻬھﻩﻫﻪۀەةہ]", r"ه", aftervav)
    afterye = re.sub(r"[ﭛﻯۍﻰﻱﻲںﻳﻴﯼېﯽﯾﯿێےىي]", r"ی", afterhe)
    afternot = re.sub(r'¬', r'‌', afterye)
    afterdot = re.sub(r'[•·●·・∙｡ⴰ]', r'.', afternot)
    aftercomma = re.sub(r'[,٬٫‚，]', r'،', afterdot)
    afterqu = re.sub(r'ʕ', r'؟', aftercomma)
    afterzero = re.sub(r'[۰٠]', r'0', afterqu)
    nc1 = re.sub(r'[۱١]', r'1', afterzero)
    nc2 = re.sub(r'[۲٢]', r'2', nc1)
    ec1 = re.sub(r'ـ|ِ|ُ|َ|ٍ|ٌ|ً|', r'', nc2)
    Sc1 = re.sub(r'( )+', r' ', ec1)
    final = re.sub(r'(\n)+', r'\n', Sc1)
    return final



def removetashdid(text):
     text = re.sub('\u0651', '', text)#tashdid
     text = re.sub('\u064a', '', text) #yeh  
     text = re.sub('\u0649', '', text) #yeh  
     text = re.sub('\u0652', '', text) #sukon 
     text = re.sub('\u064b', '', text) #fathatan 
     text = re.sub('\u064e', '', text) #fatha 
     text = re.sub('\u0650', '', text) #kasra 
     text = re.sub('\xa0',  ' ', text) #bad spaces 
         
     return text
 
    
def remove_punct(text):
        text = re.sub('[0-9]+', '', text)
        text = re.sub('[۱-۹]+', '', text) 
        return text
    
    
    
    
def emoji(text):
        import emoji
        
        allchars = [str for str in text.decode('utf-8')]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
        return clean_text    
    
def preprocess(data):
    data =  re.sub('[a-zA-Z]', '', data)
    
    
    import string
 
    data =  re.sub(r'&',' ', data)
    data = re.sub('<[^>]+>', '', data)
    data=  re.sub(' href\s*=\s*\"[^\"]*', '', data)

    
    
    data =  re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', data) # remove URLs
    data = re.sub('@[^\s]+', 'AT_USER', data)
    data = re.sub(r'#([^\s]+)', r'\1', data)

    print("done1")
    data=  remove_punct(data)
    
    
    print("done1")
    
    
    data=data.replace('[{}]'.format(string.punctuation), '')
    
    print("done1")
    clean = re.compile('<.*?>')
    data =  re.sub(clean,'', data)
    
    
    print("done1")
    
    print("done2")
    
    
    data=  emoji(data.encode('utf8'))
    data = re.sub('\u200c',' ', data)
    
    return data 
     
def read_allfile(addrs):#read file 
    list_result = []
    with open(addrs, 'r', encoding="utf_8") as f:
        stopwords = f.read()
        stopwords = stopwords.splitlines()
        for word in stopwords:
            list_result.append(word)

    return list_result  


def fix_phrases(text):
        for phrase in phrases:
            text = text.replace(phrase, iget_whole_phrase(phrase))
            text = fix_white_space(text)
        return text


def iget_whole_phrase(ph):
        ph = ph.replace(" ", '\u200c')
        return ph

def fix_white_space(text):
        text = re.sub("\s+", " ", text)
        text = text.strip()
        return text
    

#print(_fix_phrases('علی ای حال '))
       
def space_handel(string):   # space to half space
    string = re.sub(r'^(بی|می|نمی)( )', r'\1‌', string)
    #print(string)
    string = re.sub(r'( )(می|نمی|بی)( )', r'\1\2‌', string)
    string = re.sub(r'( )(هایی|ها|های|ایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|ات|ان|ین'
             r'|انی|بان|ام|ای|یم|ید|اید|اند|بودم|بودی|بود|بودیم|بودید|بودند|ست)( )', r'‌\2\3', string)
   # print(string)
    string = re.sub(r'( )(شده|نشده)( )', r'‌\2‌', string)
    #print(string)
    string = re.sub(r'( )(طلبان|طلب|گر|گرایی|گرایان|شناس|شناسی|گذاری|گذار|گذاران|شناسان|گیری|آوری|سازی|'
             r'بندی|کننده|کنندگان|پرداز|پردازی|پردازان|آمیز|سنجی|ریزی|داری|دهنده|پذیری'
             r'|پذیر|پذیران|گر|ریز|یاب|یابی|گانه|گانه‌ای|انگاری|گا|بند|رسانی|دهندگان|دار)( )?', r'‌\2\3 ', string)
    
    return string        
        

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):#remove stop words 
    text = [word for word in text if word not in stopword]
    #postingslist(text)
    return text
    



 
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text




print("done4")
def lemmatizer(text):
    #print("new line")
    s='#'
    listof=[]
    for word in text:
      lemi=  lemmatiz.lemmatize(word)
      if(s in lemi):
            lemi=lemi.split("#")[1]
            
            
      listof.append(str(lemi))      
    return listof



def elimination(output,input1,postinglist):
    listofeliminate=[]
    for word in input1:
        if word in postinglist.keys():
            second_list=list(postinglist[word])
            listofeliminate.extend(x for x in second_list if x not in listofeliminate)
            
    return   listofeliminate

def word_count(words):
    counts = dict()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts
def tffunc(token,frequency=1):
    tfDic = {}
    counter=word_count(token)
    for word, count in counter.items(): 
            tfDic[word] =1+math.log10(count)
            
            
            tfDic[word]=tfDic[word]*1
                
                
                
            
            
    return tfDic   

def vectorize(query):
   queryvec= tffunc(query,frequency=1)
    
    
    
    
    
   return queryvec


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator




def champion_combine(input1,cham_list):
    listcham=[]
    for word in input1:
        if word in cham_list.keys():
            second_list=list(cham_list[word])
            listcham.extend(x for x in second_list if x not in listcham)
            
    return   listcham
    
   
            
df1 = pd.read_csv('ir-news-0-2.csv')
df2= pd.read_csv('ir-news-2-4.csv')
df3= pd.read_csv('ir-news-4-6.csv')
df4= pd.read_csv('ir-news-6-8.csv')
df5= pd.read_csv('ir-news-8-10.csv')
df6= pd.read_csv('ir-news-10-12.csv')



df=pd.DataFrame()
df=df.append(df1)
df=df.append(df2)
df=df.append(df3)
df=df.append(df4)
df=df.append(df5)
df=df.append(df6)
df=df.reset_index()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
    
if __name__=='__main__':
    import time 

    
    always_true_flag=True
    flag_champion=False
    import PySimpleGUI as sg
    
    sg.theme('DarkAmber')# Add a touch of color
    # All the stuff inside your window.
    layout = [  [sg.Text('لطفا گزاره خود را وارد کنید ')],
                [sg.Text('در این جا'), sg.InputText()],
                [sg.Button('Ok'), sg.Button('Cancel'),sg.Button('Champion List')] ]
    
    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
            break
        if  event == 'Champion List':
            flag_champion=True
            
        print('You entered ', values[0])
    
    window.close()
    a= time.time()
    input1=values[0]
    input1=  cheak_spell_arabic(input1)
    input1=  removetashdid(input1)
    stopword=read_allfile('stopwords.txt')


    phrases = read_allfile('phrases.txt')
    phrases = set(phrases)
   
    input1 =  fix_phrases(input1)
    input1=  space_handel(input1)
   
    input1 =  tokenization(input1)
    print("tokenization done")
    input1 =  remove_stopwords(input1)
    print("remove_stopwords done")
    input1 = stemming(input1)
    input1= lemmatizer(input1)
    
    a_file = open(r"data1.pkl", "rb")
    output = pickle.load(a_file)
    print(output[0])
    a_file2 = open(r"data21.pkl", "rb")
    postinglist = pickle.load(a_file2)
    
    #a_file = open(r"data1.pkl", "rb")
    #output = pickle.load(a_file)
    #print(output[0])

    


    query_vec=vectorize(input1)
    print(query_vec)
    if(flag_champion==False):
            print("false")
            eliminate = elimination(output,input1,postinglist)
    if(flag_champion==True):
            print("true")
            a_file2 = open(r"data_cham_final.pkl", "rb")
            champion_dic = pickle.load(a_file2)
            eliminate=champion_combine(input1,champion_dic)
            print(eliminate)
            
        
    if(always_true_flag):        
            print([item for item, count in collections.Counter(eliminate).items() if count > 1])
            dict1={} 
            minHeap = MaxHeap(len(eliminate)) 
            for i in range(len(eliminate)):
                cosine = get_cosine(query_vec, output[eliminate[i]])
                minHeap.insert(cosine) 
                dict1[cosine]=eliminate[i]
            listiofoutput=[] 
            dataframefinal=pd.DataFrame()
            for i in range(0,K):
                listiofoutput.append((minHeap.extractMax()))
                                     
                                     
            for i in range(0,K):
               print(listiofoutput[i])     
            result=pd.DataFrame()                    
        
            for i  in range(0,len(listiofoutput)):
                
                
                if listiofoutput[i] in dict1.keys():
                  dataframefinal=dataframefinal.append(df[dict1[listiofoutput[i]]:dict1[listiofoutput[i]]+1])
                  result=result.append(df[dict1[listiofoutput[i]]:dict1[listiofoutput[i]]+1])
                  print(result)
    result['content'] = result['content'].map(lambda x: re.sub('[a-zA-Z]', '', x))


    result['content'] = result['content'].map(lambda x: re.sub(r'&',' ', x))
    result['content'] = result['content'].map(lambda x: re.sub('<[^>]+>', '', x))
    result['content'] = result['content'].map(lambda x: re.sub(' href\s*=\s*\"[^\"]*', '', x))
    regex = re.compile('[a-zA-Z]')
    
    
    result['content'] = result['content'].map(lambda x: re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', x)) # remove URLs
    result['content'] = result['content'].map(lambda x: re.sub('@[^\s]+', 'AT_USER', x))
    result['content'] = result['content'].map(lambda x: re.sub(r'#([^\s]+)', r'\1', x))
    clean = re.compile('<.*?>')
    result['content'] = result['content'].map(lambda x: re.sub(clean,'', x))      

    result['content'] = result['content'].replace(r'\\n',  ' ', regex=True).replace(r'\r',  ' ', regex=True).replace(r'\r\n',  ' ', regex=True).replace(r'\n',  ' ', regex=True)

    resultfinal=pd.DataFrame()
    resultfinal['Title']=result['title']
    resultfinal['Content']=result['content']
    b=time.time()
    print(str(b-a)+"time")
    win = QWidget()
    scroll = QScrollArea()
    layout = QVBoxLayout()
    table = QTableWidget()
    scroll.setWidget(table)
    layout.addWidget(table)
    win.setLayout(layout)    
    
    
    table.setColumnCount(len(resultfinal.columns))
    table.setRowCount(len(resultfinal.index))
    for i in range(len(resultfinal.index)):
        for j in range(len(resultfinal.columns)):
            table.setItem(i,j,QTableWidgetItem(str(resultfinal.iloc[i, j])))
    
    win.show()         
