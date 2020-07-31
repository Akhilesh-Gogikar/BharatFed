# BharatFed

BharatFed is a platform for the federation of financial tools for India - where the expense analysis, financial advice and access to microcredit is simple and easy for the underbanked and undereducated India.

We intend to serve the credit needs of 400 million Indians and to become a key component in the financial data value chain.

We built a platform which will utilize the consent approved financial data of a user through a new regulatory framework(Account Aggregators) and provide personal finance management, expense forecasting, p2p lending and micro loans.

Our secret sauce is our expertise in privacy preserving machine learning which generate insights into credit worthiness of a User.

Usage:
1) Install the requirements in a conda environment using pip > requirements.txt

2) The LSTM based model is trained in train_tfe_model.py: run - python train_tfe_model.py

3) The differential privacy based linear ml_model is trained in train_dp_model.py: python train_dp_model.py

4) Testing finvu aa is done in some of the other files

Sample data is present in the JSON files.

Happy Exploring!
