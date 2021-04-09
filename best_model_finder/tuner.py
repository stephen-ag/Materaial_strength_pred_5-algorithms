
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.svm import SVC
from xgboost import XGBRegressor
#import xgboost as xgb
import pandas as pd
import openpyxl

from mongoDBoperation import MongodbOperation
# from python file import class name
from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder and python file name import class name
from application_logging.loggerDB import App_LoggerDB



class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,log_database,log_collection,execution_id):
        #self.file_object = file_object
        #self.logger_object = logger_object

        self.execution_id=execution_id
        self.log_db_writer=App_LoggerDB(execution_id=execution_id)
        self.log_database=log_database
        self.log_collection=log_collection
        self.az_blob_mgt=AzureBlobManagement()
        self.mongoDBObject = MongodbOperation()

        self.linearReg = LinearRegression()
        self.RandomForestReg = RandomForestRegressor()
        self.DecisionTreeReg = DecisionTreeRegressor()
        self.XGBoostReg = XGBRegressor()
        self.AdaboostReg = AdaBoostRegressor()
        self.svm = SVC()
        #self.mse = mean_squared_error()
        #self.mae = mean_absolute_error()



    def get_best_params_for_Random_Forest_Regressor(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_Random_Forest_Regressor
                                                Description: get the parameters for Random_Forest_Regressor Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the RandomForestReg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_Random_forest_Tree = {
                                "n_estimators": [10,20,30],
                                "max_features": ["auto", "sqrt", "log2"],
                                "min_samples_split": [2,4,8],
                                "bootstrap": [True, False]
                                                     }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.RandomForestReg, self.param_grid_Random_forest_Tree, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.bootstrap = self.grid.best_params_['bootstrap']

            # creating a new model with the best parameters
            self.randomForestReg = RandomForestRegressor(n_estimators=self.n_estimators, max_features=self.max_features,
                                                         min_samples_split=self.min_samples_split, bootstrap=self.bootstrap)
            # training the mew models
            self.randomForestReg.fit(train_x, train_y)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'RandomForestReg best params: ' + str(
                                       self.grid.best_params_) + '. Exited the RandomForestReg method of the Model_Finder class')
            return self.randomForestReg
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in RandomForestReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'RandomForestReg Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self, train_x, train_y):
        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the XG boost Reg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_XGboost = {
                'learning_rate': [.001, 0.01, .1],
                'max_depth': [2, 6, 8, 10, 14],
                'min_child_weight': [1,3,5,7]
            }
                #'gamma': [0.0,0.1,0.2]

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.XGBoostReg, self.param_XGboost, verbose=3, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            #self.gamma = self.grid.best_params_['gamma']

            # creating a new model with the best parameters
            self.xgboostReg = XGBRegressor(learning_rate = self.learning_rate,
                                           max_depth = self.max_depth,
                                           min_child_weight = self.min_child_weight)

            # training the mew models
            self.xgboostReg.fit(train_x, train_y)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'xgboostReg best params: ' + str(
                                       self.grid.best_params_) + '. Exited the DecisionTreeReg  method of the Model_Finder class')
            return self.xgboostReg
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in xgboostReg   method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'xgboostReg   Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_decisionTree(self, train_x, train_y):
        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the Decision Tree Reg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_decision_Tree = {
                'criterion': ['mse', 'mae'],
                'max_depth': [2, 6, 8, 10, 14, 18, 20],
                'min_samples_leaf': [20, 40, 100],
                'min_samples_split': [10, 20, 40]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.DecisionTreeReg, self.param_decision_Tree,verbose=3, n_jobs =1, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            self.decisionTreeReg = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth,
                                                         min_samples_leaf=self.min_samples_leaf,
                                                         min_samples_split=self.min_samples_split,
                                                        )
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'DecisionTreeReg best params: ' + str(
                                       self.grid.best_params_) + '. Exited the DecisionTreeReg  method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in DecisionTreeReg  method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'DecisionTreeReg  Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_adaboostReg(self, train_x, train_y):

        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the Adda boost Reg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_ada_boost = {'n_estimators':[500,1000,2000],
                                    'learning_rate':[.001,0.01,.1],
                                    'random_state':[1]}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.AdaboostReg, self.param_ada_boost,
                                     scoring='neg_mean_squared_error',verbose=3, n_jobs =1, cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.n_estimators = self.grid.best_params_['n_estimators']
            self.learning_rate = self.grid.best_params_['learning_rate']

            # creating a new model with the best parameters- for adaboost  base estimator is Decision tree with 3 depth
            self.adaboostReg = AdaBoostRegressor(n_estimators=self.n_estimators,
                                                 learning_rate=self.learning_rate,random_state=1)

            #training the mew models
            self.adaboostReg.fit(train_x, train_y)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Ada boost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the AdaBoost Reg  method of the Model_Finder class')
            return self.adaboostReg
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in Ada BoostReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Adaboost Reg  Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_linearReg(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_linearReg
                                        Description: get the parameters for LinearReg Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                """
        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the get_best_params_for_linearReg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_linearReg = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]
            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(self.linearReg,self.param_grid_linearReg, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.fit_intercept = self.grid.best_params_['fit_intercept']
            self.normalize = self.grid.best_params_['normalize']
            self.copy_X = self.grid.best_params_['copy_X']

            # creating a new model with the best parameters
            self.linReg = LinearRegression(fit_intercept=self.fit_intercept,normalize=self.normalize,copy_X=self.copy_X)
            # training the mew model
            self.linReg.fit(train_x, train_y)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'LinearRegression best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            return self.linReg
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in get_best_params_for_linearReg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'LinearReg Parameter tuning  failed. Exited the get_best_params_for_linearReg method of the Model_Finder class')
            raise Exception()

    def get_model_metrics(self,name):

        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the get_model Metrics of the Model_Finder class')

        self.Reg_metrics = pd.DataFrame(self.Regression_score)
        self.Reg_metrics.to_excel(name+'.xlsx')

        return self.Reg_metrics

    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.log_db_writer.log(self.log_database,self.log_collection,
                               'Entered the get_best_model method of the Model_Finder class')

        try:

            self.decisionTreeReg = self.get_best_params_for_decisionTree(train_x, train_y)
            self.prediction_decisionTreeReg = self.decisionTreeReg.predict(test_x)  # Predictions using the decisiontreeReg Model
            self.prediction_decisionTreeReg_error = r2_score(test_y, self.prediction_decisionTreeReg)
            self.decisionTreeReg_mse=mean_squared_error(test_y, self.prediction_decisionTreeReg)
            self.decisionTreeReg_mae=mean_absolute_error(test_y, self.prediction_decisionTreeReg)

            # create best model for Linear Regression
            self.LinearReg= self.get_best_params_for_linearReg(train_x, train_y)
            self.prediction_LinearReg = self.LinearReg.predict(test_x) # Predictions using the LinearReg Model
            self.LinearReg_error = r2_score(test_y,self.prediction_LinearReg)
            self.LinearReg_mse=mean_squared_error(test_y, self.prediction_LinearReg)
            self.LinearReg_mae=mean_absolute_error(test_y, self.prediction_LinearReg)


         # create best model for randomforest
            self.randomForestReg = self.get_best_params_for_Random_Forest_Regressor(train_x, train_y)
            self.prediction_randomForestReg = self.randomForestReg.predict(test_x)  # Predictions using the randomForestReg Model
            self.prediction_randomForestReg_error = r2_score(test_y,self.prediction_randomForestReg)
            self.randomForestReg_mse=mean_squared_error(test_y, self.prediction_randomForestReg)
            self.randomForestReg_mae=mean_absolute_error(test_y, self.prediction_randomForestReg)

            # create best model for XGBoost
            self.XGBoostReg = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboostReg = self.XGBoostReg.predict(test_x)  # Predictions using the xgboostReg Model
            self.prediction_xgboostReg_error = r2_score(test_y, self.prediction_xgboostReg)
            self.XGBoostReg_mse=mean_squared_error(test_y, self.prediction_xgboostReg)
            self.XGBoostReg_mae=mean_absolute_error(test_y, self.prediction_xgboostReg)
          # create best model for Decision Tree


            # create best model for Ada boost
            self.adaboostReg = self.get_best_params_for_adaboostReg(train_x, train_y)
            self.prediction_adaboostReg = self.adaboostReg.predict(test_x)  # Predictions using the adaboostReg Model
            self.prediction_adaboostReg_error = r2_score(test_y, self.prediction_adaboostReg)
            self.adaboostReg_mse=mean_squared_error(test_y, self.prediction_adaboostReg)
            self.adaboostReg_mae=mean_absolute_error(test_y, self.prediction_adaboostReg)

            self.Regression_score={"LinearRegression":[self.LinearReg_error,self.LinearReg,self.LinearReg_mse,self.LinearReg_mae],
                          "randomForestRegressor":[self.prediction_randomForestReg_error,self.randomForestReg,
                                                   self.randomForestReg_mse,self.randomForestReg_mae],
                          "xg-BoostRegressor" : [self.prediction_xgboostReg_error,self.XGBoostReg,
                                                 self.XGBoostReg_mse,self.XGBoostReg_mae],
                          "decisionTreeRegressor" : [self.prediction_decisionTreeReg_error,self.decisionTreeReg,
                                                     self.decisionTreeReg_mse,self.decisionTreeReg_mae],
                          "ada-BoostRegressor" : [self.prediction_adaboostReg_error,self.adaboostReg,
                                                  self.adaboostReg_mse,self.adaboostReg_mae]
                          }


            # metrics table

            self.select=list(self.Regression_score.values())[0][0] # assigming first value from key value pair to variable
            #self.name = list(self.r2score.keys())[0][0] # assigming first key from key value pair to variable

            for i in self.Regression_score.items(): # for each i in a key value pair called by items()
                if i[1][0]>=self.select:
                    self.select=i[1][0]
                    self.name=i[0]
                    self.model=i[1][1]
            print(self.name,self.model," r2 score=", self.select)
            return self.name, self.model

            #comparing the two models
            #if(self.LinearReg_error <  self.prediction_randomForestReg_error):
            #    return 'RandomForestRegressor',self.randomForestReg
            #else:
           # #    return 'LinearRegression',self.LinearReg

        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()


