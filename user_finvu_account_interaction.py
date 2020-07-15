
import requests
import json
from datetime import datetime

def user_registration():
    return False

web_view_reg_url = 'https://finvu.in/webview/onboarding/webview-register'

#header = {'content-type': 'application/json'}

form_data = {
    "aaId" : "geetha_g",
    "mobile": "9490803074",
    "password": "1603",
    "reEnterPassword": "1603",
    "termCondition": "on"
}

r = requests.get(web_view_reg_url)#, data= form_data)#, headers=header)

r = requests.post(web_view_reg_url, data= form_data)

print(r.text)