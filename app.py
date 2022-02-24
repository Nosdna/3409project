#!/usr/bin/env python
# coding: utf-8

# In[16]:


import joblib
from flask import Flask


# In[32]:


d = joblib.load('columnstats')
d


# In[33]:


app = Flask(__name__)


# In[34]:


#backend flask- to create in app.py


from flask import request, render_template

@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == 'POST':
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income,age,loan)
        model = joblib.load('default_mlp')
        pred = model.predict([[(float(income)-d[0,0])/d[1,0],(float(age)-d[0,1])/d[1,1],(float(loan)-d[0,2])/d[1,2]]])
        s = "The predicted bankruptcy score is : " + str(pred)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="predicting")) 


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




