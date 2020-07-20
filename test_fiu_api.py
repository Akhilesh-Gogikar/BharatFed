
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
        "userName": "bharatfed99",
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
        "userId": "bharatfed99",
        "password": "6190"
    }
}

r = requests.post(login_url, data=json.dumps(login_test), headers=header)
print(r.content)

#Consent Template test

consent_temp_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/ConsentTemplate'
token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTY1NjksImp0aSI6Ik9TY2hQQUNZR3BtSGhYNnJaWTRtR2ciLCJpYXQiOjE1OTUxMzY1NjksIm5iZiI6MTU5NTEzNjU2OSwic3ViIjoiYmhhcmF0ZmVkIiwicm9sIjoiZGV2ZWxvcGVyIn0.a76TgpKq81l0S7Lf16id4Aew45yay8BzLB488yPWq72Jj5EEMltNt_VpRsuxnKq-fHoXNgcq9x34ncfTbJwZhlIL22z0l_p0d3cY49DqsvTc4CCv7gfLiEXN56jtIPqd41YaLAYpowRVxLv1lFrFIJZ2MBo_lYB62vDFMfBIA7e-JAdJH_Zxchlmz3IAYSmhLwzbcGem4WPy5EAhelvAS3fZFP5zhDpl44g42R5oxevoom7MTUKQ_uY0xi3bwCMyQbIcpgK3nTP5c2Ya8ePx3WH3krIDmFmCHt0W68E20DX337NOz9hQol0bFe4VZzJvNY5rL8BBW2fTRI38onWOfQ"
headers = {'content-type': 'application/json', "Authorization":token}

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
def consent_req(cust_id="bharatfed99@finvu", templateName="BHARATFED_PERIDOIC_DAILY_CONSENT"):
    consent_req_api='https://finsense.finvu.in/ConnectHub/FIU/API/V1/SubmitConsentRequest'

    consent_req = {
        "header": {
            "rid": "50fe9172-0144-4af3-8b23-041f38471455",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": {
            "custId": "".format(cust_id),
            "consentDescription": "Wealth Management Service",
            "templateName":"".format(templateName)
        }

    }

    r = requests.post(consent_req_api, data=json.dumps(consent_req), headers=headers)
    print(r.content)
    return r.content

#Consent request status
def check_consent_status(consent_handle="5741f541-a02e-4349-9383-03eab4f6fed8", cust_id="bharatfed99@finvu"):
    consent_req_status_api='https://finsense.finvu.in/ConnectHub/FIU/API/V1/ConsentStatus/{}/{}'.format(consent_handle, cust_id)

    #'{"header":{"rid":"50fe9172-0144-4af3-8b23-041f38471455","ts":"2020-07-07T13:15:03.093+0000","channelId":null},"body":{"custId":"bharatfed99@finvu","consentHandle":"5741f541-a02e-4349-9383-03eab4f6fed8","consentPurpose":"Wealth management service","consentDescription":"Wealth Management Service","requestDate":"2020-07-07T13:15:03.096+0000","consentStatus":"REQUESTED","requestSessionId":null,"requestConsentId":null,"dateTimeRangeFrom":"2020-04-07T13:15:02.932+0000","dateTimeRangeTo":"2020-09-07T13:15:02.932+0000"}}'

    r = requests.get(consent_req_status_api, headers=headers)

    print(r.content)
    return r.content

#data request

data_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIRequest'

data_req = {
    "header": {
        "rid": "50fe9172-0144-4af3-8b23-041f38471455",
        "ts": str(datetime.now()).replace(" ", "T"),
        "channelId": None
    },
    "body": {
        "custId": "bharatfed99@finvu",
        "consentId": "38b7d0e7-3465-47d2-81e4-3365d8d51def",
        "consentHandleId": "5741f541-a02e-4349-9383-03eab4f6fed8",
        "dateTimeRangeFrom": "2020-04-07T13:15:02.932+0000",
        "dateTimeRangeTo": "2020-07-06T08:10:45.006+0000"
    }

}

headers = {'content-type': 'application/json', "Authorization":"eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjA1ODUyNTgsImp0aSI6ImxCQlBmbWY2ckxHNEVBV2JXYUtZUlEiLCJpYXQiOjE1OTQwMjUyNTgsIm5iZiI6MTU5NDAyNTI1OCwic3ViIjoiYmhhcmF0ZmVkIiwicm9sIjoiZGV2ZWxvcGVyIn0.V2vQKt5fG6Y52g84JTvtUaYrACL7t950S88L6omLWF0-1hv-Nd1jlSUFvbQF70ibIWZ7e7T2AWOEDJexVTgoPSUIhW-iGClm6a4icvvRzr8yXGKRZOAYbVdg-IA5KirxDPHxZUmaW879hy-xGnNXHx96C6ezepZ2DyyC18ktpE_fjiVyiMSciTOvSFN-iWweDj3vcMWvJd8J8-JPjD47frRaKulQAc1p4RkXQNMj8UDKWz30xM2kaJ4eEjTU6rawBvNsFlOUKQYnXi11NmxQaG1yXd5M1l3_qZ0XSehGzk8QBPx7lq9dr8ZiNOACc7Z2lNe6zy7RlyJiTtN6sIaE2A"}


r = requests.post(data_api, data=json.dumps(data_req), headers=headers)
print(r.content)

#data req status api
#b'{"header":{"rid":"50fe9172-0144-4af3-8b23-041f38471455","ts":"2020-07-07T13:58:56.757+0000","channelId":null},"body":{"ver":"1.0","timestamp":"2020-07-07T13:58:56.722+0000","txnid":"abc81ec3-cb4a-411d-9906-6dfbc8753c35","consentId":"38b7d0e7-3465-47d2-81e4-3365d8d51def","sessionId":"afa5d6fc-dab6-451e-b043-b662a53b4b2d","consentHandleId":null}}'

data_req_stat_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIStatus/"{}"/"{}"/"{}"/"{}"'.format("38b7d0e7-3465-47d2-81e4-3365d8d51def", "afa5d6fc-dab6-451e-b043-b662a53b4b2d", "5741f541-a02e-4349-9383-03eab4f6fed8", "bharatfed99@finvu")

headers = {'content-type': 'application/json', "Authorization":"eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjA1ODUyNTgsImp0aSI6ImxCQlBmbWY2ckxHNEVBV2JXYUtZUlEiLCJpYXQiOjE1OTQwMjUyNTgsIm5iZiI6MTU5NDAyNTI1OCwic3ViIjoiYmhhcmF0ZmVkIiwicm9sIjoiZGV2ZWxvcGVyIn0.V2vQKt5fG6Y52g84JTvtUaYrACL7t950S88L6omLWF0-1hv-Nd1jlSUFvbQF70ibIWZ7e7T2AWOEDJexVTgoPSUIhW-iGClm6a4icvvRzr8yXGKRZOAYbVdg-IA5KirxDPHxZUmaW879hy-xGnNXHx96C6ezepZ2DyyC18ktpE_fjiVyiMSciTOvSFN-iWweDj3vcMWvJd8J8-JPjD47frRaKulQAc1p4RkXQNMj8UDKWz30xM2kaJ4eEjTU6rawBvNsFlOUKQYnXi11NmxQaG1yXd5M1l3_qZ0XSehGzk8QBPx7lq9dr8ZiNOACc7Z2lNe6zy7RlyJiTtN6sIaE2A"}

r = requests.get(data_req_stat_api, headers=headers)
print(r.content)

#data_fetch_api
data_fetch_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIFetch/bharatfed99@finvu/38b7d0e7-3465-47d2-81e4-3365d8d51def/afa5d6fc-dab6-451e-b043-b662a53b4b2d'

headers = {'content-type': 'application/json', "Authorization":"eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjA1ODUyNTgsImp0aSI6ImxCQlBmbWY2ckxHNEVBV2JXYUtZUlEiLCJpYXQiOjE1OTQwMjUyNTgsIm5iZiI6MTU5NDAyNTI1OCwic3ViIjoiYmhhcmF0ZmVkIiwicm9sIjoiZGV2ZWxvcGVyIn0.V2vQKt5fG6Y52g84JTvtUaYrACL7t950S88L6omLWF0-1hv-Nd1jlSUFvbQF70ibIWZ7e7T2AWOEDJexVTgoPSUIhW-iGClm6a4icvvRzr8yXGKRZOAYbVdg-IA5KirxDPHxZUmaW879hy-xGnNXHx96C6ezepZ2DyyC18ktpE_fjiVyiMSciTOvSFN-iWweDj3vcMWvJd8J8-JPjD47frRaKulQAc1p4RkXQNMj8UDKWz30xM2kaJ4eEjTU6rawBvNsFlOUKQYnXi11NmxQaG1yXd5M1l3_qZ0XSehGzk8QBPx7lq9dr8ZiNOACc7Z2lNe6zy7RlyJiTtN6sIaE2A"}


r = requests.get(data_fetch_api, headers=headers)
print(r.content)

with open('data_response.txt','wb') as f:
    f.write(r.content)