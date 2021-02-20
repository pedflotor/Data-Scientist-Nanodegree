import numpy as np
import pandas as pd
import sys # can use sys to take command line arguments

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, reviews_path, movies_path):
        '''
        It takes the data path of the data used in the recommender
        
        Parameters
        ----------
        reviews_path: str
            path to csv file with the following four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_path: str
            path to csv file with movies information
        
        Returns
        -------
        It stores:
        reviews 
            dataframe of reviews. It contains four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies
            dataframe with the movies data
        '''
        self.reviews = pd.read_csv(reviews_path)
        self.movies = pd.read_csv(movies_path)



    def fit(self, latent_features=4, learning_rate=0.01, iters=100):
        '''
        Function to perform matrix factorization using a basic form of FunkSVD with no regularization
        
        Parameters
        ----------
        latent_features: int
            number of features we want to extract from our reviews data set, default = 4
        learning_rate: float
            the rate at which we want our model to learn at, defualt = 0.01
        iters: int
            the number of iterations, default = 100
        
        Returns
        -------
        It stores:
        n_users: int
            number of users
        n_movies: int
            number of movies
        num_ratings: int
            number of ratings performed
        reviews: object
            dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies: object
            dataframe that has
                user_item_mat: object
                    a user by item numpy array with ratings and nans for values
                latent_features: int
                    the number of latent features used
                learning_rate: float
                    the learning rate
                iters: int
                    the number of iterations
        '''
        # Create user-by-item matrix
        user_items = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_by_movie)

        # Store the number of users and movies
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))

        # transfer specified attributes
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Use the training data to create a series of users and movies that matches the ordering in training data
        self.user_ids_series = np.array(self.user_by_movie.index)
        self.movie_ids_series = np.array(self.user_by_movie.columns)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, latent_features)
        movie_mat = np.random.rand(latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0
        self.mean_sse = 0

        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iteration | Mean Squared Error")

        # for each iteration
        for iteration in range(self.iters):
            
            # update our sse_accum
            old_sse = sse_accum
            sse_accum = 0
            
            # For each user-movie pair
            for i in range(self.n_users):
                # For each movie
                for j in range(self.n_movies):
                    
                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:
                        
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
            self.mean_sse = sse_accum/self.num_ratings

        # Store our matrices
        self.user_mat = user_mat
        self.movie_mat = user_mat

        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)


    def predict_rating(self, user_id, movie_id):
        '''
        This function will make the prediction of the movies rating a user would give
        
        Parameters
        ----------
        user_id: int
            the user_id from the reviews df
        movie_id:int 
            the movie_id according the movies df
        
        Returns
        -------
        pred
            the predicted rating for user_id-movie_id according to FunkSVD
        '''
        try:
            # User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("A prediction could not be made for this user-movie pair.  Seems like one of these items does not exist in the current database.")

            return None

    def make_recs(self,):
        '''
        Function, which uses the above information as necessary to provide recommendations for every user in the val_df dataframe

        Parameters
        ----------
        _id - the user or movie id we want to make recommendations for
        _id_type -  a declaration of which type of id we are analyzing (default = "movie")
        num_recs - number of recommendations we want to make

        Returns
        -------
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names


if __name__ == '__main__':
    # test different parts to make sure it works
