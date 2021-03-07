# Sparkify Project

## Introduction
A startup called Sparkify wants to analyze the data they've been collecting on songs and user activity on their new music streaming app. The analytics team is particularly interested in understanding what songs users are listening to. Currently, they don't have an easy way to query their data, which resides in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

They'd like a data engineer to create a Postgres database with tables designed to optimize queries on song play analysis, and bring you on the project. Your role is to create a database schema and ETL pipeline for this analysis. You'll be able to test your database and ETL pipeline by running queries given to you by the analytics team from Sparkify and compare your results with their expected results.

## Project Description
In this project, you'll apply what you've learned on data modeling with Postgres and build an ETL pipeline using Python. To complete the project, you will need to define fact and dimension tables for a star schema for a particular analytic focus, and write an ETL pipeline that transfers data from files in two local directories into these tables in Postgres using Python and SQL.

(Udacity)


## Files

Dataset for the ETL process

### Data with music information (.json):

Example:
``
song_data / A / B / C / TRABCEI128F424C983.json
song_data / A / A / B / TRAABJL12903CDCF1A.json
``

### Log Data (.json)

Example:
``
log_data / 2018/11 / 2018-11-12-events.json
log_data / 2018/11 / 2018-11-13-events.json
``

### Tables:

* Dimension:
   - users: contains users in the music application.
   - songs: contains songs in the database.
   - artists: contains artists in the database.
   - time: timestamp of records in songplays divided into specific units.

* Fact
   - songplays: records in the log data associated with song plays.
   
### ETL pipeline:
   - Using Python, transfer the data from the .json files to the tables with SQL.

## Walkthrough of commands for the ETL pipeline and these

> `create_tables.py`

> `test.ipynb`

> `etl.ipynb`

> `create_tables.py`

> `etl.py`


Thanks guys from Udacty