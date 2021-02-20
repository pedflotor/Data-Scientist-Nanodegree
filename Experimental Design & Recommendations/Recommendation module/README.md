# Movie Recommendation Module

## Table of Contents
1. [Instructions](#Installation)
2. [Motivation](#Motivation)
3. [Files](#Files)
4. [Acknowledgements](#Acknowledgements)

## Instructions <a name="Installation"></a>:
Run the following commands in the project's root directory:
+ `python`
+ `from recommender_template import Recommender`
+ `rec = Recommender('train_data.csv','movies_clean.csv')`

## Motivation <a name='Motivation'></a>

Module for making a prediction for any user-movie pair in the dataset. It uses FunkSVD algorithm for existing users and movies on the dataset. 
For new users and movies the recommendation is done using content based and ranked based methods.

## Files <a name="Files"></a>

* recommender_template.py
It contains the `Recommender` class and its objects.

* recommender_functions.py
It contains functions that are called on our recommender_template for many of the methods.

* movie_clean.csv
Dataset with columns: movie_id, movie, genre, date, years (one hot encoded), genres (one hot encoded)

* train_data.csv
Dataset with columns: user_id, movie_id, rating, timestamp, date, month(one hot encoded), years (one hot encoded).

## Acknowledgements <a name='Acknowledgements'></a>

Udacity Data Science Nanodegree for providing the datasets