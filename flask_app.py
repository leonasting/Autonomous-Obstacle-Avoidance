# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:02:16 2021

@author: Checkout
"""

from flask import Flask, redirect, url_for, render_template, request
from main import GridPOMDP
import pandas as pd
     


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/main", methods=["POST", "GET"])
def main():
    if request.method == "POST":
        side = request.form["grid"]
        bomb = request.form["count"]
        start= request.form["start"]
        goal= request.form["goal"]
        #print("wowo"+nd)
        
        #sentence=nd
        new_grid=GridPOMDP(int(side.strip()),int(bomb.strip()),start=[int(i) for i in start.split(',')],goal=[int(i) for i in goal.split(',')]) 
        new_grid.add_bomb()
        new_grid.q_learning()
        new_grid.q_val
        df_q_val,df_pol=new_grid.get_q_val_policy()
        df_reward=pd.DataFrame(new_grid.reward)
        print(df_pol)
        #print(df_pol.to_json())
        for ind_col in list(df_pol.columns):
            df_pol[ind_col] = df_pol[ind_col].map(lambda x:".  "+ x+"   .")
        df=df_pol
        
       
        return render_template("label.html",tables=[df.to_html(classes='data')], titles=df.columns.values)
    else:
        return render_template("main.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)