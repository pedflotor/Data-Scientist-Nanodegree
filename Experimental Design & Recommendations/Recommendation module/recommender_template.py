import numpy as np
import pandas as pd
import recommender_functions as rf
import sys  # can use sys to take command line arguments


class Recommender():
    """
    Python package that uses FunkSVD algorithm for making recommendations
    """

    def __init__(self):
        """
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
        """
        #self.reviews = pd.read_csv(reviews_path)
        #self.movies = pd.read_csv(movies_path)

    def fit(self, reviews_pth, movies_pth, latent_features=12, learning_rate=0.01, iters=100):
        """
        Function to perform matrix factorization using a basic form of FunkSVD with no regularization
        Parameters
        ----------
        reviews_pth: object
            path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth: object
            path to csv with each movie and movie information in each row
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
        """

        # Store inputs as attributes
        self.reviews = pd.read_csv(reviews_pth)
        self.movies = pd.read_csv(movies_pth)

        # Create user-item matrix
        usr_itm = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_item_df = usr_itm.groupby(['user_id','movie_id'])['rating'].max().unstack()
        self.user_item_mat= np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))
        self.user_ids_series = np.array(self.user_item_df.index)
        self.movie_ids_series = np.array(self.user_item_df.columns)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.user_item_mat[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))

        # SVD based fit
        # Keep user_mat and movie_mat for safe keeping
        self.user_mat = user_mat
        self.movie_mat = movie_mat

        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)

    def predict_rating(self, user_id, movie_id):
        """
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
        """
        try:# User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print(
                "A prediction could not be made for this user-movie pair.  Seems like one of these items does not "
                "exist in the current database.")

            return None

    def make_recs(self, _id, _id_type='movie', rec_num=5):
        """
        Function which provide recommendations for every user in the dataframe

        Parameters
        ----------
        _id: int
            either a user or movie id
        _id_type: str
            "movie" or "user"(default = "movie")
        rec_num: int
            number of recommendations to return

        Returns
        -------
        recs: object
            (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        """
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
                print("The user specified is not in the database. The top movie recommendation will be given")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("The movie doesn't exist in the database.  No recomendation could be done.")

        return rec_ids, rec_names


if __name__ == '__main__':
    # test different parts to make sure it works
    import recommender_template as r

    # instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_pth='/Users/petanth/PycharmProjects/Data-Scientist-Nanodegree/Experimental Design & Recommendations/Recommendation module/train_data.csv', movies_pth='/Users/petanth/PycharmProjects/Data-Scientist-Nanodegree/Experimental Design & Recommendations/Recommendation module/movies_clean.csv', learning_rate=.01, iters=2)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    # user that is in the dataset
    print(rec.make_recs(8, 'user'))
    # user that is not in the dataset
    print(rec.make_recs(1, 'user'))
    # movie that is in the dataset
    print(rec.make_recs(1853728))
    # movie that is not in dataset
    print(rec.make_recs(1))
    print("Users on the dataset: ", rec.n_users)
    print("Movies on the dataset: ", rec.n_movies)
    print("Ratings on the dataset: ", rec.num_ratings)