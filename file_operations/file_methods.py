import pickle
import os
import shutil
import re
from mongoDBoperation import MongodbOperation
# from python file import class name
from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder and python file name import class name
from application_logging.loggerDB import App_LoggerDB


class File_Operation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
    def __init__(self,log_database,log_collection,execution_id):
        #self.file_object = file_object
        #self.logger_object = logger_object
        #self.model_directory='models/'

        self.log_database=log_database
        self.log_collection=log_collection
        self.execution_id=execution_id
        self.log_db_writer=App_LoggerDB(execution_id=self.execution_id)
        self.model_directory='model'
        self.az_blob_mgt=AzureBlobManagement()

    def save_model(self, model, filename):
        """`
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None
"""
        self.log_db_writer.log(self.log_database, self.log_collection,
                               'Entered the save_model method of the File_Operation class')
        directory_name = self.model_directory + '-' + filename
        print(directory_name)
        try:
            self.az_blob_mgt.createDir(directory_name, is_replace=True)  # create or replace directory
            # path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            # if os.path.isdir(path): #remove previously existing models for each clusters
            #     shutil.rmtree(self.model_directory)
            #     os.makedirs(path)
            # else:
            #     os.makedirs(path) #

            # with open(path +'/' + filename+'.sav',
            #           'wb') as f:
            #     pickle.dump(model, f) # save the model to file
            self.az_blob_mgt.saveObject(directory_name=directory_name, filename=filename + '.sav',object_name=model)
            self.log_db_writer.log(self.log_database, self.log_collection,
                                   'Model File ' + filename + ' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.log_db_writer.log(self.log_database, self.log_collection,
                                   'Exception occured in save_model method of the'
                                   ' Model_Finder class. Exception message:  ' + str(e))
            self.log_db_writer.log(self.log_database, self.log_collection,
                                   'Model File ' + filename + ' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()

    def load_model(self,filename):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """
        self.log_db_writer.log(self.log_database,self.log_collection, 'Entered the load_model method of the File_Operation class')
        try:
            #with open(self.model_directory + filename + '/' + filename + '.sav','rb') as f:
            directory=self.model_directory+'-'+filename
            filename=filename+'.sav'
            object_model=self.az_blob_mgt.loadObject(directory,filename)
            self.log_db_writer.log(self.log_database,self.log_collection,
                                       'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
            return object_model
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()

    def find_correct_model_file(self,cluster_number):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        self.log_db_writer.log(self.log_database,self.log_collection, 'Entered the find_correct_model_file method of the File_Operation class')
        try:
            self.cluster_number= cluster_number
            self.folder_name=self.model_directory
            self.list_of_model_files = []
            self.required_files = self.az_blob_mgt.dir_list
            self.list_of_files = []
            #self.list_of_files = os.listdir(self.folder_name)
            for dir in self.required_files:
                #if re.search("^model[-][a-zA-z]{2,17}[0-9]",dir):
                if re.search("^model-xg[-][a-zA-z]{2,17}[0-9]|^model[-][a-zA-z]{2,17}[0-9]", dir):

                    self.list_of_files.append(dir)
                    print(self.list_of_files)

            for self.file in self.list_of_files:
                try:
                    #selecting model file name in models
                    models=self.az_blob_mgt.getAllFileNameFromDirectory(self.file)
                    for model_name_ in models:
                        if (model_name_.index(str(self.cluster_number))!=-1):
                            self.model_name=model_name_

                            print(self.model_name)
            #for self.file in self.list_of_files:
            #    try:
            #        if (self.file.index(str( self.cluster_number))!=-1):
            #            self.model_name=self.file
                except:
                    continue
            self.model_name=self.model_name.split('.')[0]
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exited the find_correct_model_file method of the Model_Finder class.')
            return self.model_name
        except Exception as e:
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_db_writer.log(self.log_database,self.log_collection,
                                   'Exited the find_correct_model_file method of the Model_Finder class with Failure')
            raise Exception()

