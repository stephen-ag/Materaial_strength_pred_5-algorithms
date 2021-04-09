import sqlite3
from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
#from application_logging.logger import App_Logger
from application_logging.loggerDB import App_LoggerDB
# from folder import python file logger#
from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder import python file#
from mongoDBoperation import MongodbOperation






class Prediction_Data_validation:
    """
               This class shall be used for handling all the validation done on the Raw Prediction Data!!.

               Written By: iNeuron Intelligence
               Version: 1.0
               Revisions: None

               """

    def __init__(self,path,execution_id):
        self.Batch_Directory = path
        self.execution_id=execution_id
        self.collection_name = "strength_schema_prediction"    #code added by Avnish yadav
        self.database_name = "Wafer-sys"    #code added by Avnish yadav
        self.logger_db_writer=App_LoggerDB(execution_id=execution_id) #code added by Avnish yadav
        self.mongdb=MongodbOperation()
        self.az_blob_mgt=AzureBlobManagement()
        self.good_directory_path="good-raw-file-prediction-validated"
        self.bad_directory_path="bad-raw-file-prediction-validated"

    def valuesFromSchema(self):
        """
                                Method Name: valuesFromSchema
                                Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                                Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                                On Failure: Raise ValueError,KeyError,Exception

                                 Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                                        """
        log_database = "strength_prediction_log"
        log_collection = "values_from_schema_validation"
        try:
            log_database="wafer_prediction_log"
            log_collection="values_from_schema_validation"
            df_schema_training = self.mongdb.getDataFrameofCollection(self.database_name, self.collection_name)
            dic = {}
            [dic.update({i: df_schema_training.loc[0, i]}) for i in df_schema_training.columns]
            del df_schema_training
            #with open(self.schema_path, 'r') as f:
            #    dic = json.load(f)
            #    f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            #file = open("Training_Logs/valuesfromSchemaValidationLog.txt", 'a+')
            message ="LengthOfDateStampInFile:: %s" %LengthOfDateStampInFile + "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile +"\t " + "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.logger_db_writer.log(log_database,log_collection,message)

            #file.close()

        except ValueError:
            self.logger_db_writer.log(log_database,log_collection,"KeyError:Key value error incorrect key passed")
            raise ValueError

        except KeyError:
            self.logger_db_writer.log(log_database,log_collection,"KeyError:Key value error incorrect key passed")
            raise KeyError
        except Exception as e:
            self.logger_db_writer.log(log_database, log_collection, str(e))
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns



    def manualRegexCreation(self):

        """
                                      Method Name: manualRegexCreation
                                      Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                                  This Regex is used to validate the filename of the prediction data.
                                      Output: Regex pattern
                                      On Failure: None

                                       Written By: iNeuron Intelligence
                                      Version: 1.0
                                      Revisions: None

                                              """
        regex = "['cement_strength']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def createDirectoryForGoodBadRawData(self):

        """
                                        Method Name: createDirectoryForGoodBadRawData
                                        Description: This method creates directories to store the Good Data and Bad Data
                                                      after validating the prediction data.

                                        Output: None
                                        On Failure: OSError

                                         Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                                """
        log_database = "strength_prediction_log"
        log_collection = "general_log"
        try:
            log_database = "strength_prediction_log"
            log_collection = "general_log"
            self.az_blob_mgt.createDir(self.good_directory_path, is_replace=True)
            self.az_blob_mgt.createDir(self.bad_directory_path, is_replace=True)
            msg = self.good_directory_path + " and " + self.bad_directory_path + " created successfully."
            self.logger_db_writer.log(log_database, log_collection, msg)
        except Exception as e:
            msg = "Error Occured in class Prediction_Data_validation method:createDirectoryForGoodBadRawData error: Failed to create directory " + self.good_directory_path + " and " + self.bad_directory_path
            self.logger_db_writer.log(log_database, log_collection, msg)
            raise e


    def deleteExistingGoodDataTrainingFolder(self):
        """
                                            Method Name: deleteExistingGoodDataTrainingFolder
                                            Description: This method deletes the directory made to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        log_database = "strength_prediction_log"
        log_collection = "general_log"
        try:
            log_database = "strength_prediction_log"
            log_collection = "general_log"
            self.az_blob_mgt.deleteDir(self.good_directory_path)
            self.logger_db_writer.log(log_database, log_collection,
                                      self.good_directory_path + " deleted successfully!!")
        except Exception as e:
            msg = "Error Occured in class Raw_Data_validation method:deleteExistingGoodDataTrainingFolder Error occured while deleting :" + self.good_directory_path
            self.logger_db_writer.log(log_database, log_collection, msg)
            raise e


    def deleteExistingBadDataTrainingFolder(self):

        """
                                            Method Name: deleteExistingBadDataTrainingFolder
                                            Description: This method deletes the directory made to store the bad Data.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        log_database = "strength_prediction_log"
        log_collection = "general_log"

        try:
            log_database = "strength_prediction_log"
            log_collection = "general_log"
            self.az_blob_mgt.deleteDir(self.bad_directory_path)
            self.logger_db_writer.log(log_database, log_collection,
                                      self.bad_directory_path + " deleted successfully!!")

        except Exception as e:
            msg = "Error Occured in class Raw_Data_validation method:deleteExistingGoodDataTrainingFolder Error occured while deleting :" + self.good_directory_path
            self.logger_db_writer.log(log_database, log_collection, msg)
            raise e

    def moveBadFilesToArchiveBad(self):

        """
                                            Method Name: moveBadFilesToArchiveBad
                                            Description: This method deletes the directory made  to store the Bad Data
                                                          after moving the data in an archive folder. We archive the bad
                                                          files to send them back to the client for invalid data issue.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        log_database = "strength_prediction_log"
        log_collection = "general_log"

        try:
            log_database = "strength_prediction_log"
            log_collection = "general_log"

            # source = 'Training_Raw_files_validated/Bad_Raw/'
            source = self.bad_directory_path
            destination = "lap-" + self.execution_id
            self.logger_db_writer.log(log_database, log_collection, "Started moving bad raw data..")
            for file in self.az_blob_mgt.getAllFileNameFromDirectory(source):
                self.az_blob_mgt.moveFileinDir(source, destination, file)
                self.logger_db_writer.log(log_database, log_collection,
                                          "File:" + file + " moved to directory:" + destination + " successfully.")

            self.logger_db_writer.log(log_database, log_collection,
                                      "All bad raw file moved to directory:" + destination)

            self.az_blob_mgt.deleteDir(source)
            self.logger_db_writer.log(log_database, log_collection, "Deleting bad raw directory:" + source)

        except Exception as e:
            self.logger_db_writer.log(log_database, log_collection,
                                      "class Raw_Data_validation method:moveBadFilesToArchiveBad Error while moving bad files to archive:" + str(
                                          e))
            raise e




    def validationFileNameRaw(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        """
            Method Name: validationFileNameRaw
            Description: This function validates the name of the prediction csv file as per given name in the schema!
                         Regex pattern is used to do the validation.If name format do not match the file is moved
                         to Bad Raw Data folder else in Good raw data.
            Output: None
            On Failure: Exception

             Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.createDirectoryForGoodBadRawData()
        onlyfiles = self.az_blob_mgt.getAllFileNameFromDirectory(self.Batch_Directory)
        try:
            log_database = "strength_prediction_log"
            log_collection = "name_validation_log"
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[2]) == LengthOfDateStampInFile:
                        if len(splitAtDot[3]) == LengthOfTimeStampInFile:
                            self.az_blob_mgt.CopyFileinDir(self.Batch_Directory, self.good_directory_path,
                                                                 filename)
                            self.logger_db_writer.log(log_database, log_collection,
                                                      "Valid File name!! File moved to " + self.good_directory_path + filename)

                        else:
                            self.az_blob_mgt.CopyFileinDir(self.Batch_Directory, self.bad_directory_path,
                                                                 filename)
                            msg = "Invalid File Name !! File moved to " + self.bad_directory_path + filename
                            self.logger_db_writer.log(log_database, log_collection, msg)
                    else:
                        self.az_blob_mgt.CopyFileinDir(self.Batch_Directory, self.bad_directory_path,
                                                             filename)
                        msg = "Invalid File Name !! File moved to " + self.bad_directory_path + filename
                        self.logger_db_writer.log(log_database, log_collection, msg)

                else:
                    self.az_blob_mgt.CopyFileinDir(self.Batch_Directory, self.bad_directory_path,
                                                         filename)
                    msg = "Invalid File Name !! File moved to " + self.bad_directory_path + filename
                    self.logger_db_writer.log(log_database, log_collection, msg)
        except Exception as e:
            msg = "Error occured while validating FileName " + str(e)
            self.logger_db_writer.log(log_database, log_collection, msg)
            raise e




    def validateColumnLength(self,NumberofColumns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            log_database="strength_prediction_log"
            log_collection="column_validation_log"
            self.logger_db_writer.log(log_database,log_collection,"Column length validation Started!!")
            #for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
            for file in self.az_blob_mgt.getAllFileNameFromDirectory(self.good_directory_path):
                #csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                csv=self.az_blob_mgt.readCSVFilefromDir(self.good_directory_path,file)
                print(csv.shape)
                if csv.shape[1] == NumberofColumns:
                    #csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    print(csv)
                    #csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
                    self.az_blob_mgt.saveDataFrametoCSV(self.good_directory_path,file,csv,index=None,header=True)
                else:
                    self.az_blob_mgt.moveFileinDir(self.good_directory_path,self.bad_directory_path,file)
                    self.logger_db_writer.log(log_database,log_collection,"Invalid Column Length for the file!! "
                                                                          "File moved to Bad Raw Folder :: %s" % file)
            self.logger_db_writer.log(log_database,log_collection,"Column Length Validation Completed!!")

        except Exception as e:
            self.logger_db_writer.log(log_database,log_collection,'Error Occured::'+str(e) )
            raise e



    def deletePredictionFile(self):
        try:
            log_database = "strenth_prediction_log"
            log_collection = "general_log"
            directory="prediction-file"
            filename="Prediction.csv"
            if directory in self.az_blob_mgt.dir_list:
                filenames=self.az_blob_mgt.getAllFileNameFromDirectory(directory_name=directory)
                if filename in filenames:
                    self.az_blob_mgt.deleteFilefromDir(directory_name=directory,filename=filename)
                    self.logger_db_writer.log(log_database,log_collection,filename+" is deleted from dir:"+directory+" successfully")
        except Exception as e:

            self.logger_db_writer.log(log_database,log_collection,"Error occure while deleting prediction file from prediction-file directory"+str(e))
            raise e

    def validateMissingValuesInWholeColumn(self):
        """
                                  Method Name: validateMissingValuesInWholeColumn
                                  Description: This function validates if any column in the csv file has all values missing.
                                               If all the values are missing, the file is not suitable for processing.
                                               SUch files are moved to bad raw data.
                                  Output: None
                                  On Failure: Exception

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                              """
        try:
            log_database="strength_prediction_log"
            log_collection="missing_values_in_column"
            #f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            #self.logger.log(f, "Missing Values Validation Started!!")
            self.logger_db_writer.log(log_database,log_collection, "Missing Values Validation Started!!")

            #for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
            for file in self.az_blob_mgt.getAllFileNameFromDirectory(self.good_directory_path):
                #csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                csv=self.az_blob_mgt.readCSVFilefromDir(self.good_directory_path,file)
                print(csv)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        #shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                        #            "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.az_blob_mgt.moveFileinDir(self.good_directory_path,self.bad_directory_path,file)
                        #self.logger.log(f,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        self.logger_db_writer.log(log_database,log_collection,
                                        "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)

                        break
                if count==0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    print("column unnamed may not be present")
                    self.az_blob_mgt.saveDataFrametoCSV(self.good_directory_path,file,csv,index=None,header=True)
                    #csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
        except Exception as e:

            self.logger_db_writer.log(log_database,log_collection,"Error occured:"+str(e))
            raise e













