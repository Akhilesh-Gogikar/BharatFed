def reg_consent_template():
    consent_temp_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/ConsentTemplate'

    token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTk0ODMsImp0aSI6IjlMVWtaaXpIdnJtYWRHS1VpOGtkU3ciLCJpYXQiOjE1OTUxMzk0ODMsIm5iZiI6MTU5NTEzOTQ4Mywic3ViIjoiYmhhcmF0ZmVkOTkiLCJyb2wiOiJkZXZlbG9wZXIifQ.IyQ3LbX4WlLUzOjfX6jt8cOOSH9yYsaTHpXTROj_MMXjxIFfWatBCQ7doFIsRkWQzZE5tb75UcDK4mCVW21SGCGSigtrkoZTRpD4iV-QGlxzp6uJTL0WHqqKsJFspn8Uvh3eC6F1z-6dto8KZNWT8VObytZuu7hqAXxYJaIu3-wIEn3Er7EfGNDKKTLYs2lpsIm0IsyJGrg2Mz6Fo7UocmofuKQjpnJftoyXIwiAkvO3LSSgdPY3dKzWysaoNaIcuFGLUVXrP77zp53sNY5AqmGGnqu0KmNgJgtGt0FfXTwKnj77ReI_MICvKHIuJf1o6x13E-rLvIFRQJgAdV3r-A"

    headers = {'content-type': 'application/json', "Authorization": token}
    consent_template = {
        "header": {
            "rid": "50fe9172-0144-4af3-8b23-041f38471456",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": {
            "templateName": "BHARATFED_ONETIME_YEAR_CONSENT",
            "templateDescription": "BHARATFED_ONETIME_YEAR_CONSENT",
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
            "fetchType": "ONETIME",
            "Frequency": {
                "unit": "DAY",
                "value": 5
            },
            "DataLife": {
                "unit": "YEAR",
                "value": 5
            },
            "ConsentExpiry": {
                "unit": "YEAR",
                "value": 5
            },
            "consentStartDays": 1,
            "dataRangeStartMonths": 12,
            "dataRangeEndMonths": 2,

            "approvalStatus": "pending verification"
        }
    }

    r = requests.post(consent_temp_url, data=json.dumps(consent_template), headers=headers)
    print(r.content)


def template_status():
    consent_temp_approval_url = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/ApproveConsentTemplate'

    token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTk0ODMsImp0aSI6IjlMVWtaaXpIdnJtYWRHS1VpOGtkU3ciLCJpYXQiOjE1OTUxMzk0ODMsIm5iZiI6MTU5NTEzOTQ4Mywic3ViIjoiYmhhcmF0ZmVkOTkiLCJyb2wiOiJkZXZlbG9wZXIifQ.IyQ3LbX4WlLUzOjfX6jt8cOOSH9yYsaTHpXTROj_MMXjxIFfWatBCQ7doFIsRkWQzZE5tb75UcDK4mCVW21SGCGSigtrkoZTRpD4iV-QGlxzp6uJTL0WHqqKsJFspn8Uvh3eC6F1z-6dto8KZNWT8VObytZuu7hqAXxYJaIu3-wIEn3Er7EfGNDKKTLYs2lpsIm0IsyJGrg2Mz6Fo7UocmofuKQjpnJftoyXIwiAkvO3LSSgdPY3dKzWysaoNaIcuFGLUVXrP77zp53sNY5AqmGGnqu0KmNgJgtGt0FfXTwKnj77ReI_MICvKHIuJf1o6x13E-rLvIFRQJgAdV3r-A"
    headers = {'content-type': 'application/json', "Authorization": token}
    consent_temp_approval = {
        "header": {
            "rid": "bc82d96a-da27-4269-a8dd-192d28c20888",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": {
            "templateName": "BHARATFED_ONETIME_YEAR_CONSENT"
        }
    }

    r = requests.post(consent_temp_approval_url, data=json.dumps(consent_temp_approval), headers=headers)
    print(r.content)


def consent_req(cust_id="bharatfed@finvu", templateName="BHARATFED_ONETIME_YEAR_CONSENT"):
    consent_req_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/SubmitConsentRequest'

    consent_req = {
        "header": {
            "rid": "2fc76189-1b7d-4b04-8392-4f3c8d0f347f",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": "finsense"
        },
        "body": {
            "custId": "{}".format(cust_id),
            "consentDescription": "Wealth management service",
            "templateName": "{}".format(templateName)
        }

    }

    r = requests.post(consent_req_api, data=json.dumps(consent_req), headers=headers)
    print(r.content)
    return r.content
'''
cust_id = 
b'{"header":{"rid":"2fc76189-1b7d-4b04-8392-4f3c8d0f347f","ts":"2020-07-19T06:31:54.664+0000","channelId":"finsense"},"body":{"custId":"bharatfed@finvu","consentHandle":"f6c0c910-c7c2-4856-90ce-708ece62fdbe","consentPurpose":"Wealth management service","consentDescription":"Wealth management service","requestDate":"2020-07-19T06:31:54.667+0000","consentStatus":"REQUESTED","requestSessionId":null,"requestConsentId":null,"dateTimeRangeFrom":"2019-07-19T06:31:54.526+0000","dateTimeRangeTo":"2020-09-19T06:31:54.526+0000"}}'
'''
'''
b'{"header":{"rid":"2fc76189-1b7d-4b04-8392-4f3c8d0f347f","ts":"2020-07-19T07:08:05.430+0000","channelId":"finsense"},"body":{"custId":"bharatfed99@finvu","consentHandle":"f962616e-1ba9-4e1f-bfac-d756a498973c","consentPurpose":"Wealth management service","consentDescription":"Wealth management service","requestDate":"2020-07-19T07:08:05.433+0000","consentStatus":"REQUESTED","requestSessionId":null,"requestConsentId":null,"dateTimeRangeFrom":"2019-07-19T07:08:05.322+0000","dateTimeRangeTo":"2020-09-19T07:08:05.322+0000"}}'
'''

#Consent request status
def check_consent_status(consent_handle="f6c0c910-c7c2-4856-90ce-708ece62fdbe", cust_id="bharatfed99@finvu"):
    consent_req_status_api='https://finsense.finvu.in/ConnectHub/FIU/API/V1/ConsentStatus/{}/{}'.format(consent_handle, cust_id)

    #'{"header":{"rid":"50fe9172-0144-4af3-8b23-041f38471455","ts":"2020-07-07T13:15:03.093+0000","channelId":null},"body":{"custId":"bharatfed99@finvu","consentHandle":"5741f541-a02e-4349-9383-03eab4f6fed8","consentPurpose":"Wealth management service","consentDescription":"Wealth Management Service","requestDate":"2020-07-07T13:15:03.096+0000","consentStatus":"REQUESTED","requestSessionId":null,"requestConsentId":null,"dateTimeRangeFrom":"2020-04-07T13:15:02.932+0000","dateTimeRangeTo":"2020-09-07T13:15:02.932+0000"}}'

    r = requests.get(consent_req_status_api, headers=headers)

    print(r.content)
    return r.content

'''
b'{"header":{"rid":"13432d3f-50fc-428c-a0d1-e57a84b7646e","ts":"2020-07-19T07:11:19.863+0000","channelId":null},"body":{"consentStatus":"ACCEPTED","consentId":"8a0a5ff3-ee39-43d6-b201-2fff711e7776"}}'
'''

def data_req(cust_id="bharatfed99@finvu",consentId="2036a0b1-fb5b-431e-9163-0f0c2ad07e3c",consentHandleId="f6c0c910-c7c2-4856-90ce-708ece62fdbe", start="2019-07-19T13:15:02.932+0000",end="2020-07-06T08:10:45.006+0000"):

    # data request

    data_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIRequest'

    data_req = {
        "header": {
            "rid": "50fe9172-0144-4af3-8b23-041f38471455",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": {
            "custId": "{}".format(cust_id),
            "consentId": "{}".format(consentId),
            "consentHandleId": "{}".format(consentHandleId),
            "dateTimeRangeFrom": "{}".format(start),
            "dateTimeRangeTo": "{}".format(end)
        }

    }

    token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTk0ODMsImp0aSI6IjlMVWtaaXpIdnJtYWRHS1VpOGtkU3ciLCJpYXQiOjE1OTUxMzk0ODMsIm5iZiI6MTU5NTEzOTQ4Mywic3ViIjoiYmhhcmF0ZmVkOTkiLCJyb2wiOiJkZXZlbG9wZXIifQ.IyQ3LbX4WlLUzOjfX6jt8cOOSH9yYsaTHpXTROj_MMXjxIFfWatBCQ7doFIsRkWQzZE5tb75UcDK4mCVW21SGCGSigtrkoZTRpD4iV-QGlxzp6uJTL0WHqqKsJFspn8Uvh3eC6F1z-6dto8KZNWT8VObytZuu7hqAXxYJaIu3-wIEn3Er7EfGNDKKTLYs2lpsIm0IsyJGrg2Mz6Fo7UocmofuKQjpnJftoyXIwiAkvO3LSSgdPY3dKzWysaoNaIcuFGLUVXrP77zp53sNY5AqmGGnqu0KmNgJgtGt0FfXTwKnj77ReI_MICvKHIuJf1o6x13E-rLvIFRQJgAdV3r-A"
    headers = {'content-type': 'application/json', "Authorization": token}

    r = requests.post(data_api, data=json.dumps(data_req), headers=headers)
    print(r.content)

'''
b'{"header":{"rid":"50fe9172-0144-4af3-8b23-041f38471455","ts":"2020-07-19T06:58:03.088+0000","channelId":null},"body":{"ver":"1.0","timestamp":"2020-07-19T06:58:03.058+0000","txnid":"8304625a-a8f6-4271-9198-1189f31dded4","consentId":"2036a0b1-fb5b-431e-9163-0f0c2ad07e3c","sessionId":"504c7d08-f06d-4bde-885d-f38790361f22","consentHandleId":null}}'
'''

'''
b'{"header":{"rid":"50fe9172-0144-4af3-8b23-041f38471455","ts":"2020-07-19T07:13:06.260+0000","channelId":null},"body":{"ver":"1.0","timestamp":"2020-07-19T07:13:06.242+0000","txnid":"14ce1e67-11f2-4082-bd42-cb5a2b47eb0c","consentId":"8a0a5ff3-ee39-43d6-b201-2fff711e7776","sessionId":"26c2f749-02b6-429f-a13d-6f7fa4f8646b","consentHandleId":null}}'
'''
def data_req_stat(cust_id, consentId, consentHandleId, sessionId):
    data_req_stat_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIStatus/"{}"/"{}"/"{}"/"{}"'.format(
        consentId, sessionId,
        consentHandleId, cust_id)

    token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTk0ODMsImp0aSI6IjlMVWtaaXpIdnJtYWRHS1VpOGtkU3ciLCJpYXQiOjE1OTUxMzk0ODMsIm5iZiI6MTU5NTEzOTQ4Mywic3ViIjoiYmhhcmF0ZmVkOTkiLCJyb2wiOiJkZXZlbG9wZXIifQ.IyQ3LbX4WlLUzOjfX6jt8cOOSH9yYsaTHpXTROj_MMXjxIFfWatBCQ7doFIsRkWQzZE5tb75UcDK4mCVW21SGCGSigtrkoZTRpD4iV-QGlxzp6uJTL0WHqqKsJFspn8Uvh3eC6F1z-6dto8KZNWT8VObytZuu7hqAXxYJaIu3-wIEn3Er7EfGNDKKTLYs2lpsIm0IsyJGrg2Mz6Fo7UocmofuKQjpnJftoyXIwiAkvO3LSSgdPY3dKzWysaoNaIcuFGLUVXrP77zp53sNY5AqmGGnqu0KmNgJgtGt0FfXTwKnj77ReI_MICvKHIuJf1o6x13E-rLvIFRQJgAdV3r-A"
    headers = {'content-type': 'application/json', "Authorization": token}

    r = requests.get(data_req_stat_api, headers=headers)
    print(r.content)

def fetch_data(consentId, sessionId, cust_id):
    data_fetch_api = 'https://finsense.finvu.in/ConnectHub/FIU/API/V1/FIFetch/{}/{}/{}'.format(cust_id,consentId,sessionId)

    token = "eyJraWQiOiJyc2ExIiwiYWxnIjoiUlMyNTYifQ.eyJpc3MiOiJjb29raWVqYXIiLCJhdWQiOiJjb29raWVqYXIiLCJleHAiOjIwNjE2OTk0ODMsImp0aSI6IjlMVWtaaXpIdnJtYWRHS1VpOGtkU3ciLCJpYXQiOjE1OTUxMzk0ODMsIm5iZiI6MTU5NTEzOTQ4Mywic3ViIjoiYmhhcmF0ZmVkOTkiLCJyb2wiOiJkZXZlbG9wZXIifQ.IyQ3LbX4WlLUzOjfX6jt8cOOSH9yYsaTHpXTROj_MMXjxIFfWatBCQ7doFIsRkWQzZE5tb75UcDK4mCVW21SGCGSigtrkoZTRpD4iV-QGlxzp6uJTL0WHqqKsJFspn8Uvh3eC6F1z-6dto8KZNWT8VObytZuu7hqAXxYJaIu3-wIEn3Er7EfGNDKKTLYs2lpsIm0IsyJGrg2Mz6Fo7UocmofuKQjpnJftoyXIwiAkvO3LSSgdPY3dKzWysaoNaIcuFGLUVXrP77zp53sNY5AqmGGnqu0KmNgJgtGt0FfXTwKnj77ReI_MICvKHIuJf1o6x13E-rLvIFRQJgAdV3r-A"
    headers = {'content-type': 'application/json', "Authorization": token}

    r = requests.get(data_fetch_api, headers=headers)
    print(r.content)

    with open('data_response_{}_1yr.txt'.format(cust_id), 'wb') as f:
        f.write(r.content)