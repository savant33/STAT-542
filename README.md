# STAT-542
Project-4 Lending Club Loan Status Analysis
This repository contains analysis of Lending Club Loan Status using histroical loan data issued by Lending Club. The goal of this project is to buil a model to predict the chance of default for a loan. 

Data Sources:
https://www.kaggle.com/wendykan/lending-club-loan-data: data 2007-15.
https://www.kaggle.com/wordsforthewise/lending-club: all data till 2018Q2.

The dataset has over 100 features, but some of them have too many NA values, and some are not suposed to be available at the beginning of the loan. For example, it is not meaningful to predict the status of a loan if we knew the date/amount of the last payment of that loan.

Some data cleaning is required and is not included in the current .R file. 

