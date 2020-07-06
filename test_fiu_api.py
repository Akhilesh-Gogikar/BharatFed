
import requests
import json
from datetime import datetime

# Registration test
reg_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/registerUser'

header = {'content-type': 'application/json'}

reg_test = {
    "header": {
        "rid": "50fe9172-0144-4af3-8b23-041f38471455",
        "ts": str(datetime.now()).replace(" ","T"),#"2020-06-15T11:03:44.427+0000",
        "channelId": "finsense"
    },
    "body": {
        "userName": "bharatfed",
        "password": "6190",
        "confirmPassword": "6190"
    }
}

r = requests.post(reg_url, data=json.dumps(reg_test), headers=header)
print(r.content)

#Login test
login_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/User/Login'

login_test = {
    "header": {
        "rid": "42c06b9f-cc5b-4a53-9119-9ca9d8e9acdb",
        "ts":  str(datetime.now()).replace(" ","T"),
        "channelId": "finsense"
    },
    "body":{
        "userId": "bharatfed",
        "password": "6190"
    }
}

r = requests.post(login_url, data=json.dumps(login_test), headers=header)
print(r.content)

#Consent Template test

consent_temp_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/ConsentTemplate'

headers = {'content-type': 'application/json', "Authorization":"eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjA1ODUyNTgsImp0aSI6ImxCQlBmbWY2ckxHNEVBV2JXYUtZUlEiLCJpYXQiOjE1OTQwMjUyNTgsIm5iZiI6MTU5NDAyNTI1OCwic3ViIjoiYmhhcmF0ZmVkIiwicm9sIjoiZGV2ZWxvcGVyIn0.V2vQKt5fG6Y52g84JTvtUaYrACL7t950S88L6omLWF0-1hv-Nd1jlSUFvbQF70ibIWZ7e7T2AWOEDJexVTgoPSUIhW-iGClm6a4icvvRzr8yXGKRZOAYbVdg-IA5KirxDPHxZUmaW879hy-xGnNXHx96C6ezepZ2DyyC18ktpE_fjiVyiMSciTOvSFN-iWweDj3vcMWvJd8J8-JPjD47frRaKulQAc1p4RkXQNMj8UDKWz30xM2kaJ4eEjTU6rawBvNsFlOUKQYnXi11NmxQaG1yXd5M1l3_qZ0XSehGzk8QBPx7lq9dr8ZiNOACc7Z2lNe6zy7RlyJiTtN6sIaE2A"}

consent_template = {
    "header": {
        "rid": "50fe9172-0144-4af3-8b23-041f38471455",
        "ts": str(datetime.now()).replace(" ","T"),
        "channelId": None
    },
    "body": {
                "templateName": "BHARATFED_PERIDOIC_DAILY_CONSENT",
                "templateDescription": "BHARATFED_PERIDOIC_DAILY_CONSENT",
                "consentMode": "STORE",
                "consentTypes": [
                    "PROFILE",
                    "TRANSACTIONS",
                    "SUMMARY"
                ],
                "fiTypes": [
                    "DEPOSIT"
                ],
                "Purpose": {
                    "code": "101",
                    "refUri": "https://api.rebit.org.in/aa/purpose/101.xml",
                    "text": "Wealth management service",
                    "Category": {
                        "type": "Personal Finance"
                    }
                },
                "fetchType": "PERIODIC",
                "Frequency": {
                    "unit": "DAY",
                    "value": 5
                },
                "DataLife":{
                    "unit": "YEAR",
                    "value": 5
                },
                 "ConsentExpiry":{
                    "unit": "YEAR",
                    "value": 5
                },
                "consentStartDays": 1,
                "dataRangeStartMonths": 3,
                "dataRangeEndMonths": 2,
                "approvalStatus": "pending verification"
            }
}

r = requests.post(consent_temp_url, data=json.dumps(consent_template), headers=headers)
print(r.content)

#Consent template approval test

consent_temp_approval_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/ApproveConsentTemplate'

consent_temp_approval = {
    "header": {
        "rid": "50fe9172-0144-4af3-8b23-041f38471455",
        "ts": str(datetime.now()).replace(" ","T"),
        "channelId": None
    },
    "body": {
        "templateName": "BHARATFED_PERIDOIC_DAILY_CONSENT"
    }
}

r = requests.post(consent_temp_approval_url, data=json.dumps(consent_temp_approval), headers=headers)
print(r.content)

#Consent request
consent_req_api='https://finsense.finvu.in/ConnectHub/FIU/API/V1/SubmitConsentRequest'

consent_req = {
    "header": {
        "rid": "50fe9172-0144-4af3-8b23-041f38471455",
        "ts": str(datetime.now()).replace(" ", "T"),
        "channelId": None
    },
    "body": {
        "custId": "bharatfed99@finvu",
        "consentDescription": "Wealth Management Service",
        "templateName":"BHARATFED_PERIDOIC_DAILY_CONSENT"
    }

}

r = requests.post(consent_req_api, data=json.dumps(consent_req), headers=headers)
print(r.content)