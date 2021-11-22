# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:22:47 2021

@author: Checkout
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

import pickle
import random
from main import GridPOMDP


app = Flask(__name__)

@app.route('/testApi',methods=['POST'])

def run_sim():
    data= request.get_json(force=True)
    new_grid=GridPOMDP(int(data['side']),int(data['bomb_count']),start=[int(i) for i in data['start']],goal=[int(i) for i in data['goal']]) 
    new_grid.add_bomb()
    new_grid.q_learning()
    new_grid.q_val
    df_q_val,df_pol=new_grid.get_q_val_policy()
    df_reward=pd.DataFrame(new_grid.reward)
    print(df_pol)
    print(df_pol.to_json())
    return jsonify(df_pol.to_json())
    


if __name__ == "__main__" :
    app.run(port=5000,debug=True)
    