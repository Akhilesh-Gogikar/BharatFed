import json


def find_spare_balance(data=[], balance=0):
    '''

    :param data: Predictions based on the past data
    :return: Amount that can be safely deducted from the balance given th predictions

    '''

    #print(data, balance)

    amount_list = []

    for entry in data:
        if entry['type'] == 'CREDIT':
            amount_list.append(entry['amount'])
        else:
            amount_list.append(-1 * entry['amount'])

    neg_sum = 0

    #print(amount_list)

    for amt in reversed(amount_list):

        neg_sum += amt

        if neg_sum > 0:
            neg_sum = 0

    safe_amt = balance + neg_sum

    if safe_amt < 0:
        safe_amt = 0

    return safe_amt


#print(find_spare_balance())
