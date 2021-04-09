from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation
from DataTransformation_Prediction.DataTransformationPrediction import dataTransformPredict
from application_logging import logger

from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder and python file name import class name
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import DbOperationMongoDB
#from folder and python file import class name
from application_logging.loggerDB import App_LoggerDB

class pred_validation:
    def __init__(self,path,execution_id):
        self.raw_data = Prediction_Data_validation(path,execution_id)
        self.dataTransform = dataTransformPredict(execution_id)
        #self.dBOperation = dBOperation()
        #self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        #self.log_writer = logger.App_Logger()
        self.dBOperationMongoDB = DbOperationMongoDB(execution_id)
        self.log_database = "strength_prediction_log"
        self.log_collection = "stg-prediction_main_log"
        self.execution_id = execution_id
        self.logDB_write = App_LoggerDB(execution_id=execution_id)
        self.az_blob_mgt = AzureBlobManagement()


    def prediction_validation(self):

        try:

            self.logDB_write.log(self.log_database,self.log_collection,'Start of Validation on files for prediction!!')
            #extracting values from prediction schema
            LengthOfDateStampInFile,LengthOfTimeStampInFile,column_names,noofcolumns = self.raw_data.valuesFromSchema()
            #getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            #validating filename of prediction files
            self.raw_data.validationFileNameRaw(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)
            #validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            #validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            self.logDB_write.log(self.log_database,self.log_collection,"Raw Data Validation Complete!!")

            self.logDB_write.log(self.log_database,self.log_collection,"Creating Prediction_Database and tables on the basis of given schema!!!")
            #create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperationMongoDB.insertIntoTableGoodData(column_names)
            self.logDB_write.log(self.log_database,self.log_collection,"Table creation Completed!!")
            self.logDB_write.log(self.log_database,self.log_collection,"Insertion of Data into Table started!!!!")
            #insert csv files in the table
            #self.dBOperationMongoDB.insertIntoTableGoodData('Prediction') #***************************** NEED TO CHECK BEFORE RUNNING******
            #self.logDB_write.log(self.log_database,"Insertion in Table completed!!!")
            #self.logDB_write.log(self.log_database,"Deleting Good Data Folder!!!")
            #Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.logDB_write.log(self.log_database,self.log_collection,"Good_Data folder deleted!!!")
            self.logDB_write.log(self.log_database,self.log_collection,"Moving bad files to Archive and deleting Bad_Data folder!!!")
            #Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.logDB_write.log(self.log_database,self.log_collection,"Bad files moved to archive!! Bad folder Deleted!!")
            self.logDB_write.log(self.log_database,self.log_collection,"Validation Operation completed!!")
            self.logDB_write.log(self.log_database,self.log_collection,"Extracting csv file from table")
            #export data in table to csvfile
            self.dBOperationMongoDB.selectingDatafromtableintocsv()

        except Exception as e:
            raise e









