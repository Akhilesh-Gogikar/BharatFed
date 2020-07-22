import sqlite3
import json
import email
import os
import base64
from datetime import datetime, timedelta

bharat_fed = "bharat_fed.db"

loan_table = "loans"

loan_req_table = "loan_reqs"

profile_table = "profiles"

# UserID, First_name, Last_Name, Pan, Last_Balance, Last_Category, Last_Txn_Amt, Credit_Score, Bill_Income_ratio, profile_img, text, frugality_factor, fin_goals

# c.execute('''CREATE TABLE profiles
# (userid text, first_name text, last_name text, Pan text, phone text, email text, last_balance real, last_txn_amt real,
# credit_score real, bill_income_ratio real, profile_img text,
# profile_text text, frugality_factor real, fin_goals text)''')


'''
c.execute("INSERT INTO profiles VALUES ('000000','Akhilesh','Gogikar','DSDIU3206D','9002305646','gogikar.akhilesh@gmail.com',"
                                "'200000','1221','700','0.4','False','Cool Guy','0.2','Buy Mansion in California')"
                                
c.execute("INSERT INTO profiles VALUES ('000001','Himanshu','Ojha','DSDIU3206F','9002305646','himanshuojha56@gmail.com',"
                                "'1500','2258','650','0.6','False','Cool Dude','0.05','Start gulab jamun business')")
'''


# Loan_req_id, created_at, borrower_id, loan_amount, interest_rate, loan_period,  req_text, lender_id, req_status, approved_at, txn_id, loan_id

# c.execute('''CREATE TABLE loan_reqs
# (loan_req_id text, created_at text, borrower_id text, loan_amount real, interest_rate real, loan_period real,
# req_text text, lender_id text, req_status text,
# approved_at text, txn_id text, loan_id text)''')

# Loan_id, borrower_id, lender_id, txn_id, initial_principal, remaining_principal, EMI, next_pay_date, next_emi

# c.execute('''CREATE TABLE loans
# (Loan_id text, borrower_id text, lender_id text, txn_id text, initial_principal real, next_principal real,
# EMI real, next_pay_date text, principal_emi real, last_updated text)''')

class p2p_module():

    def __init__(self):
        self.conn = sqlite3.connect('bharat_fed.db')
        self.cursor = self.conn.cursor()

    def calculate_emis(self, interestRate, numberOfMonths, principalBorrowed):
        import numpy as np

        principal2Pay = np.ppmt(interestRate / 12, 1, numberOfMonths, principalBorrowed)

        interest2Pay = np.ipmt(interestRate / 12, 1, numberOfMonths, principalBorrowed)

        print(principal2Pay, interest2Pay)

        return (principal2Pay, interest2Pay)

    def register_loan_request(self, data):
        '''
        :param data: json containing the keys for loan requests entry e.g. data = {'borrower_id': '000001',
        'loan_amount':100000, 'interest_rate':0.12, 'loan_period':2, 'text':'shaadi ke liye udhar',
        'lender_id':'000000'}

        :return: True if loan req registration worked
        '''
        out = False
        borrower = data['borrower_id']
        loan_req_id = base64.b64encode(os.urandom(12)).decode('ascii')
        loan_amt = data['loan_amount']
        if int(loan_amt) <= 0:
            return "Loan amount can't be zero or less. Much be greater than zero!"
        int_r = data['interest_rate']
        if int_r < 0:
            return "Interest rate can't be negative"
        loan_p = data['loan_period']

        if loan_p <= 0:
            return "Loan period can't be a negative number or zero"

        req_txt = data["text"]
        lender = data["lender_id"]

        if lender is None:
            return "Lender can't be a Null value"

        req_status = "PENDING"
        approved_at = None
        txn_id = None
        loan_id = None

        try:
            print("Works uptill before insert!")
            '''
            self.cursor.execute("INSERT INTO loan_reqs VALUES ({},{},{},{},{},{},"
                                "{},{},{},{},{},{})".format(loan_req_id,
                                                            str(
                                                                datetime.now()),
                                                            borrower,
                                                            loan_amt,
                                                            int_r,
                                                            loan_p,
                                                            req_txt,
                                                            lender,
                                                            req_status,
                                                            approved_at,
                                                            txn_id,
                                                            loan_id))
            '''
            data = (str(loan_req_id), str(datetime.now()), borrower, float(loan_amt), float(int_r),
             float(loan_p), str(req_txt), str(lender), str(req_status), str(approved_at),
             str(txn_id), str(loan_id))
            self.cursor.execute("""INSERT INTO loan_reqs VALUES(?,?,?,?,?,?,?,?,?,?,?,?);""", data)
            self.conn.commit()
            notice_data = {"lender": lender, "borrower": borrower, "loan_amount": loan_amt, "interest_rate": int_r,
                           "loan_period": loan_p, "req_id": loan_req_id}
            self.send_loan_req_notification(notice_data)
            out = True
        except Exception as e:
            return e

        return out

    def applied_loan_requests(self, cust_id):
        '''
        :param cust_id of the borrower eg. '000001'
        :return: List of all entries matching the borrower_id with cust_id and Loan status as pending
        '''

        out_list = []

        try:

            data = (cust_id, 'PENDING')

            self.cursor.execute(
                "SELECT * FROM loan_reqs WHERE borrower_id=? AND req_status=?", data)

            out_list = self.cursor.fetchall()

        except Exception as e:
            print(e)

        return out_list

    def fetch_loan_requests(self, cust_id):
        '''
        :param cust_id of the lender eg. '000000'
        :return: List of all entries matching the lender_id with cust_id and Loan status as pending
        '''
        out_list = []

        try:
            data = (cust_id, 'PENDING')

            self.cursor.execute(
                "SELECT * FROM loan_reqs WHERE lender_id=? AND req_status=?", data)

            out_list = self.cursor.fetchall()

        except Exception as e:
            print(e)

        return out_list

    def approve_loan_request(self, loan_req_id):
        '''
        :param cust_id - the id of the lender, loan_req_id - the req_id for the loan being approved
        :return: True if transaction occurs and the data stored in the table
        '''
        txn_id = self.make_loan_transaction()

        if len(txn_id) <= 12:
            print(txn_id)
            return False

        loan_id = base64.b64encode(os.urandom(12)).decode('ascii')

        # fetch email and phone numbers of lender and borrower
        try:
            self.cursor.execute(
                "SELECT lender_id, borrower_id, loan_amount, interest_rate, loan_period FROM loan_reqs WHERE loan_req_id=?",(loan_req_id,))

            lender, borrower, loan_amt, interest, period = self.cursor.fetchone()

            self.cursor.execute(
                "SELECT phone, email FROM profiles WHERE userid=?",(lender,))

            len_phone, len_email = self.cursor.fetchone()

            self.cursor.execute(
                "SELECT phone, email FROM profiles WHERE userid=?",(borrower,))

            bor_phone, bor_email = self.cursor.fetchone()

            send_data = {"loan_id": loan_id, "len_phone": len_phone, "len_email": len_email, "bor_phone": bor_phone,
                         "bor_email": bor_email, "txn_id": txn_id}
        except Exception as e:
            print(e)

        if send_data:
            self.send_approved_email(send_data)
            self.send_approved_sms(send_data)
            out = True

        # update the loan req in table
        data = ("APPROVED", loan_id,loan_req_id)
        self.cursor.execute(
            "UPDATE loan_reqs SET req_status=?, loan_id=? WHERE loan_req_id=?", data)

        self.conn.commit()

        # (Loan_id text, borrower_id text, lender_id text, txn_id text, initial_principal real, next_principal real,
        # EMI real, next_pay_date text, next_emi real)''')

        try:
            principal_emi, interest_emi = self.calculate_emis(float(interest), int(period), float(loan_amt))

            emi = principal_emi + interest_emi

            next_pay_date = datetime.now() + timedelta(30)

            data = (loan_id,borrower,lender,txn_id,loan_amt,(loan_amt - principal_emi),emi,str(next_pay_date.date()),principal_emi,str(datetime.now()))

            print(data)

            # insert in loan data into loan
            self.cursor.execute("INSERT INTO loans VALUES(?,?,?,?,?,?,?,?,?,?)", data)
            self.conn.commit()

        except Exception as e:
            print(e)

        return out

    def reject_loan_request(self, loan_req_id, comment):
        '''
        :param comment:
        :param loan_req_id:
        :param cust_id - the id of the lender, loan_req_id - the req_id for the loan being rejected and optional comments
        :return: True if status of the req marked as Rejected and comment appended to req_text
        '''

        out = False

        try:
            self.cursor.execute(
                "SELECT lender_id, borrower_id, loan_amount, interest_rate, loan_period FROM loan_reqs WHERE loan_req_id=?",(loan_req_id,))

            lender, borrower, loan_amt, interest, period = self.cursor.fetchone()

            self.cursor.execute(
                "SELECT phone, email FROM profiles WHERE userid=?",(lender,))

            len_phone, len_email = self.cursor.fetchone()

            self.cursor.execute(
                "SELECT phone, email FROM profiles WHERE userid=?",(borrower,))

            bor_phone, bor_email = self.cursor.fetchone()

            send_data = {"len_phone": len_phone, "len_email": len_email, "bor_phone": bor_phone,
                         "bor_email": bor_email}
        except Exception as e:
            print(e)

        if send_data:
            self.send_rejected_email(send_data)

        try:

            self.cursor.execute(
                "SELECT req_text FROM loan_reqs WHERE loan_req_id=?", (loan_req_id,))

            # update data to the loan req in table
            data = ("REJECTED", comment, loan_req_id)
            self.cursor.execute(
                "UPDATE loan_reqs SET req_status=?, req_text=? WHERE loan_req_id=?", data)

            self.conn.commit()

            out = True

        except Exception as e:
            print(e)

        return out

    def send_loan_req_notification(self, data):
        '''
        :param data: id of the lender, loan request details and email & phonenumber of the lender
        :return: True if successfully sent a notification
        '''
        print(data)
        return False

    def send_approved_email(self, data):
        '''
        :param data: ids of lender and borrower and details of the loan transaction
        :return: True if the email is successfully sent
        '''
        print(data)
        return False

    def send_rejected_email(self, data):
        '''

        :param data: email id of the borrower and comments by the lender are emailed
        :return: True if the email is successfully sent
        '''
        print(data)
        return False

    def send_approved_sms(self, data):
        '''

        :param data: ids of the borrower and lender and details of the loan transaction
        :return: True if sms successfully sent
        '''
        print(data)
        return False

    def make_loan_transaction(data):
        '''

        :param data: Amount and UserID of lender and borrower
        :return: True if the loan transaction was successful
        '''
        try:
            txn_id = base64.b64encode(os.urandom(12)).decode('ascii')
        except Exception as e:
            return e
        return str(txn_id)
