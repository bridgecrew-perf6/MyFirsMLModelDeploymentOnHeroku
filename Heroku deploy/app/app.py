#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kaleido

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])

def home():
    if request.method=="GET":
           return render_template('index.html',href='static/base_pic.svg')
    else:
        text=request.form['text']
        path='static/prediction_pic.svg'
        model=pickle.load(open('model.pkl','rb'))
        np_arr=floats_str_to_np_arr(text)
        make_picture('AgesAndHeights.pkl',model,np_arr,path)
        return render_template('index.html',href=path)

def make_picture(training_data_filename,model,new_inp_np_arr,output_file):
    data=pd.read_pickle(training_data_filename)
    data=data[data['Age']>0]
    ages=data['Age']
    heights=data['Height']
    x_new=np.array(list(range(19))).reshape(19,1)
    preds=model.predict(x_new)
    fig=px.scatter(x=ages,y=heights,title='Age vs Height', labels={'x':'Ages (years)','y':'Heights (inches)'})
    fig.add_traces(go.Scatter(x=x_new.reshape(19),y=preds,mode='lines',name='Model'))
    new_preds=model.predict(new_inp_np_arr)
    fig.add_traces(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)),y=new_preds,name='New_Output',mode='markers',marker=dict(color='purple',size=20,line=dict(color='purple',width=2))))
    fig.write_image(output_file, width=600,engine='kaleido')
    
def floats_str_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats=np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats),1)

if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




