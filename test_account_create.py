import yaml
import requests
import json
from datetime import datetime, timedelta
import random
import string


def add_user_test_acc(detail_dict):
    req_api = 'http://api.finvu.in/Accounts/add'

    req = {
        "header": {
            "rid": "50fe9172-0144-4af3-8b23-041f38471455",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": {
            'UserAccountInfo': {
                "UserAccount": {
                    "accountNo": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
                    "accountRefNo": ''.join(random.choices(string.ascii_uppercase + string.digits, k=11)),
                    "accountTypeEnum": 'CURRENT',
                    "FIType": 'DEPOSIT',
                    "Identifiers": {

                        "pan": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
                        "mobile": detail_dict['phone'],
                        "email": detail_dict['email'],
                        "aadhar": ''.join(random.choices(string.digits, k=10))
                    }
                }
            }
        }

    }

    request = {
        "header": {
            "rid": "50fe9172-0144-4af3-8b23-041f38471455",
            "ts": str(datetime.now()).replace(" ", "T"),
            "channelId": None
        },
        "body": '<UserAccountInfo>'
                '<UserAccount accountRefNo={} accountNo={} accountTypeEnum={} '
                'FIType={}> '
                '<Identifiers pan={} mobile={} email={} '
                'aadhar={}></Identifiers> '
                '</UserAccount>'
                '</UserAccountInfo>'.format(req['body']['UserAccountInfo']['UserAccount']['accountRefNo'],
                                            req['body']['UserAccountInfo']['UserAccount']['accountNo'],
                                            req['body']['UserAccountInfo']['UserAccount']['accountTypeEnum'],
                                            req['body']['UserAccountInfo']['UserAccount']['FIType'],
                                            req['body']['UserAccountInfo']['UserAccount']['Identifiers']['pan'],
                                            req['body']['UserAccountInfo']['UserAccount']['Identifiers']['mobile'],
                                            req['body']['UserAccountInfo']['UserAccount']['Identifiers']['email'],
                                            req['body']['UserAccountInfo']['UserAccount']['Identifiers']['aadhar'])


    }

    Headers = {'Content-type': 'application/json'}

    detail_dict['accountRefNo'] = req['body']['UserAccountInfo']['UserAccount']['accountRefNo']
    detail_dict['accountNo'] = req['body']['UserAccountInfo']['UserAccount']['accountNo']

    r = requests.post(req_api, data=json.dumps(request), headers=Headers)
    print(r.content)

    return detail_dict


def add_transaction(number):
    # print(number)

    return 0


start_day = "2017-07-03"

part_add = datetime.strptime(start_day, "%Y-%m-%d")

today = datetime.now().date()

config = yaml.load(open('test_account_conf.yml'))

days = today - part_add.date()

print(days)

for key in config['Users']:

    config['Users'][key] = add_user_test_acc(config['Users'][key])

    for day in range(int(days.days)):
        num = random.random()

        add_transaction(num)

config_doc = yaml.dump(config, open("test_account_conf.yml", "w"))

print(config_doc)
