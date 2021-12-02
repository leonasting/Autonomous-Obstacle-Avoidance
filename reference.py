# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:41:44 2021

@author: Checkout
"""

import requests
url = 'http://localhost:5000/BuyingPropensityAPI'
r = requests.post(url,json={'BPData':{"ANNUAL_PREMIUM": 280,
"PREMIUM_COLLECTION_METHOD": "Credit Card",
"VEHICLE_USAGE_TYPE": "Commercial",
"AGE": 30,
"VEHICLE_AGE_POL_START": 3,
"NO_OF_DRIVER":1,
"RISK_SUM_INSURED": 3000,
"DEDUCTIBLE":250
		}})
#returns probablity of Quote converting into policy
print(r.text)

"server"
app = Flask(__name__)
# Load the model
#
model = pickle.load(open('BuyingPropensity.pkl','rb'))
@app.route('/BuyingPropensityAPI',methods=['POST'])