# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:02:16 2021

@author: Checkout
"""

from flask import Flask, redirect, url_for, render_template, request
from main import GridPOMDP
import pandas as pd
import plotly   
import plotly.express as px
import json
import matplotlib.pyplot as plt 

#fig = px.imshow([[1, 20, 30],
#                 [20, 1, 60],
#                 [30, 60, 1]]) 


#graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
#fig.show()
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
        option= request.form["option"]
        #print("wowo"+nd)
        
        #sentence=nd
        new_grid=GridPOMDP(int(side.strip()),int(bomb.strip()),start=[int(i) for i in start.split(',')],goal=[int(i) for i in goal.split(',')]) 
        new_grid.add_bomb()
        if option=="Comparision":
            new_grid.q_learning()
            df=new_grid.display_path()
            new_grid.reset_q_val()
            q_reward=new_grid.ls_reward
            
            new_grid.SARSA(episode_count=1000, learning_rate=0.9, discount_factor=0.9, epsilon=0.99)
            sarsa_reward=new_grid.ls_reward
            
            df2=new_grid.display_path()
            for ind_col in list(df.columns):
                df[ind_col] = df[ind_col].map(lambda x:".  . "+ x+" .   .")
            for ind_col in list(df2.columns):
                df2[ind_col] = df2[ind_col].map(lambda x:".  . "+ x+" .   .")
            
            df_res = pd.DataFrame({"Q- Learning Accumulated reward":q_reward,"SARSA Accumulated reward":sarsa_reward})
             
            
            plt.figure()
            df_res.plot(title='Q-learning vs SARSA Accumulated Rewards')
            plt.xlabel('Episodes')
            plt.ylabel('Accumulated Rewards')
            plt.savefig('./static/images/new_plot.png')
            
            return render_template("compare.html",table1=df.to_html(classes='data'),table2=df2.to_html(classes='data'),url ='./static/images/new_plot.png')
            
        
        elif option=="SARSA":
            new_grid.SARSA(episode_count=1000, learning_rate=0.9, discount_factor=0.9, epsilon=0.99)
        elif option=="Q-Learning":
            new_grid.q_learning()
        df=new_grid.display_path()
        for ind_col in list(df.columns):
            df[ind_col] = df[ind_col].map(lambda x:".  "+ x+"   .")
        return render_template("label.html",tables=[df.to_html(classes='data')],option=option, titles=df.columns.values)
    #,graphJSON=graphJSON, header=header,option=option,description=description)
    else:
        return render_template("main.html")

@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)