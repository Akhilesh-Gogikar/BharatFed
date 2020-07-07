import yaml
import requests
import json
from datetime import datetime, timedelta
import random
import string
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.sax.saxutils import unescape
import re


def add_user_test_acc(detail_dict):
    global req

    req_api = 'http://api.finvu.in/Accounts/add'

    req = {

        "body": {
            'UserAccountInfo': {
                "UserAccount": {
                    "accountNo": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
                    "accountRefNo": ''.join(random.choices(string.ascii_uppercase + string.digits, k=11)),
                    "accountTypeEnum": 'CURRENT',
                    "FIType": 'DEPOSIT',
                    "Identifiers": {

                        "pan": ''.join(random.choices(string.ascii_uppercase, k=5))+''.join(random.choices(string.digits, k=4))+''.join(random.choices(string.ascii_uppercase, k=1)),
                        "mobile": detail_dict['phone'],
                        "email": detail_dict['email'],
                        "aadhar": ''.join(random.choices(string.digits, k=10))
                    }
                }
            }
        }

    }

    top = Element('UserAccountInfo')
    useraccount = SubElement(top, 'UserAccount',
                             {'accountRefNo': req['body']['UserAccountInfo']['UserAccount']['accountRefNo'],
                              'accountNo': req['body']['UserAccountInfo']['UserAccount']['accountNo'],
                              'accountTypeEnum': req['body']['UserAccountInfo']['UserAccount']['accountTypeEnum'],
                              'FIType': req['body']['UserAccountInfo']['UserAccount']['FIType']})
    identifier = SubElement(useraccount, 'Identifiers',
                            {'pan': req['body']['UserAccountInfo']['UserAccount']['Identifiers']['pan'],
                             'mobile': str(req['body']['UserAccountInfo']['UserAccount']['Identifiers']['mobile']),
                             'email': req['body']['UserAccountInfo']['UserAccount']['Identifiers']['email'],
                             'aadhar': req['body']['UserAccountInfo']['UserAccount']['Identifiers']['aadhar']})

    Headers = {'Content-type': 'application/xml'}

    detail_dict['accountRefNo'] = req['body']['UserAccountInfo']['UserAccount']['accountRefNo']
    detail_dict['accountNo'] = req['body']['UserAccountInfo']['UserAccount']['accountNo']
    detail_dict['pan'] = req['body']['UserAccountInfo']['UserAccount']['Identifiers']['pan']
    detail_dict['aadhar'] = req['body']['UserAccountInfo']['UserAccount']['Identifiers']['aadhar']

    r = requests.post(req_api, data=tostring(top), headers=Headers)
    print(r.text)

    return detail_dict


def add_transaction(summary, draw_limit, txn_day, day):
    expense_cats = ['food', 'travel', 'recharge', 'purchase', 'gift', 'beauty', 'gym', 'outing', 'medical',
                    'entertainment']

    income_cats = ['e_transfer', 'gift', 'loan', 'sale']

    # if days from start is multiple of 30 credit salary
    if day % 30 == 0:
        deb_cr = "CREDIT"
        amount = 45000
        summary['currentBalance'] += amount
        ref = 'SALARY'

    elif day % 29 == 0:
        deb_cr = "DEBIT"
        amount = 10000
        summary['currentBalance'] -= amount
        ref = 'RENT'

    else:

        if random.random() > 0.5:
            deb_cr = "CREDIT"
            amount = round(random.randrange(1000, 10000), 2)
            summary['currentBalance'] += amount
            ref = ''.join(random.choices(income_cats, k=1))
        else:
            deb_cr = "DEBIT"
            amount = round(random.randrange(100, summary["currentBalance"]), 2)
            if amount < draw_limit:
                summary['currentBalance'] -= amount
            else:
                amount = draw_limit
                summary['currentBalance'] -= amount
            ref = ''.join(random.choices(expense_cats, k=1))

    txn = {
        "txnId": ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)),
        "type": deb_cr,
        "mode": "OTHERS",
        "amount": str(amount),
        "currentBalance": str(summary['currentBalance']),
        "transactionTimestamp": str(datetime.combine(txn_day, datetime.min.time())).replace(" ", "T"),
        "valueDate": str(txn_day.date()),
        "narration": ref,
        "reference": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    }

    txn_str = '<Transaction txnId="{}" type="{}" mode="{}" amount="{}" currentBalance="{}" ' \
              'transactionTimestamp="{}" valueDate="{}" ' \
              'narration="{}" reference="{}"/>'.format(txn["txnId"],
                                                    txn["type"],
                                                    txn["mode"],
                                                    txn["amount"],
                                                    txn["currentBalance"],
                                                    txn["transactionTimestamp"],
                                                    txn["valueDate"],
                                                    txn["narration"],
                                                    txn["reference"])

    return txn_str


def add_transactions_data(number, start_day, data_dict):
    # print(number)
    trans_api = "http://api.finvu.in/Accounts/Transaction/add"
    start = datetime.strptime(start_day, "%Y-%m-%d")
    draw_limit = 10000
    summary = {"currentBalance": round(random.randrange(2000, 20000), 2),
               "currency": "INR",
               "exchngeRate": "5.5",
               "balanceDateTime": str(datetime.combine(start, datetime.min.time())).replace(" ", "T"),
               "type": "CURRENT",
               "branch": "Bangalore",
               "facility": "CC",
               "ifscCode": "BARBOSHIPOO",
               "micrCode": "411012011",
               "openingDate": start_day,
               "currentODLimit": "20000",
               "drawingLimit": str(draw_limit),
               "status": "Active"}

    summary_str = '<Summary currentBalance="{}" currency="{}" exchgeRate="{}" ' \
                  'balanceDateTime="{}" type="{}" branch="{}" facility="{}" ' \
                  'ifscCode="{}" micrCode="{}" openingDate="{}" currentODLimit="{}" ' \
                  'drawingLimit="{}" status="ACTIVE"><Pending amount="20.0"/></Summary> '.format(summary["currentBalance"],
                                                                                               summary["currency"],
                                                                                               summary["exchngeRate"],
                                                                                               summary["balanceDateTime"],
                                                                                               summary["type"],
                                                                                               summary["branch"],
                                                                                               summary["facility"],
                                                                                               summary["ifscCode"],
                                                                                               summary["micrCode"],
                                                                                               summary["openingDate"],
                                                                                               summary["currentODLimit"],
                                                                                               summary["drawingLimit"],
                                                                                               summary["status"])

    c_data_str = '<![CDATA[<Account xmlns="http://api.rebit.org.in/FISchema/deposit" ' \
                 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
                 'xsi:schemaLocation="http://api.rebit.org.in/FISchema/deposit ../FISchema/deposit.xsd" ' \
                 'linkedAccRef="{}" maskedAccNumber="{}" version="1.0" ' \
                 'type="deposit"><Profile><Holders type="JOINT"><Holder name="YOUR NAME" dob="1995-04-19" ' \
                 'mobile="{}" nominee="NOT-REGISTERED" email="{}" pan="{}" ' \
                 'ckycCompliance="true"/></Holders></Profile> '.format(data_dict['accountRefNo'],
                                                                       data_dict['accountNo'],
                                                                       data_dict['phone'],
                                                                       data_dict['email'],
                                                                       data_dict['pan'])

    end_str = '</Transactions></Account>]]>'

    transactions = []

    for day in range(number):
        try:
            transactions.append(add_transaction(summary, draw_limit, start + timedelta(day), day))
        except Exception as e:
            #print(e)
            continue

    #print(transactions)

    end_day = start + timedelta(number)

    txns = '<Transactions startDate="{}" endDate="{}">'.format(start_day, str(end_day.date()))

    cdata_str = c_data_str + summary_str + txns + "".join(txn for txn in transactions) + end_str

    #print(cdata_str)

    top = Element('UserAccountTrans')

    useraccount = SubElement(top, 'UserAccount', {})

    acc_ref_no = SubElement(useraccount, 'accountRefNo', {})
    acc_ref_no.text = data_dict['accountRefNo']
    acc_no = SubElement(useraccount, 'accountNo', {})
    acc_no.text = data_dict['accountNo']
    fi_type = SubElement(useraccount, 'FIType', {})
    fi_type.text = 'DEPOSIT'
    acc_type_enum = SubElement(useraccount, 'accountTypeEnum', {})
    acc_type_enum.text = 'CURRENT'
    acc_data = SubElement(useraccount, 'accountData', {})
    acc_data.text = cdata_str

    Headers = {'Content-type': 'application/xml'}

    print(unescape(tostring(top).decode()))

    r = requests.post(trans_api, data=unescape(tostring(top).decode()), headers=Headers)
    print(r.text)

    return 0




start_day = "2017-07-03"

part_add = datetime.strptime(start_day, "%Y-%m-%d")

today = datetime.now().date()

config = yaml.load(open('test_account_conf.yml'))

days = today - part_add.date()

print(days)

for key in config['Users']:
    config['Users'][key] = add_user_test_acc(config['Users'][key])

    add_transactions_data(int(days.days), start_day, config['Users'][key])

config_doc = yaml.dump(config, open("test_account_conf.yml", "w"))

print(config_doc)
