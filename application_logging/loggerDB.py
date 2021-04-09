from datetime import datetime
import pandas as pd
from mongoDBoperation import MongodbOperation
# from python file import class name
from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder and python file name import class name


class App_LoggerDB:
    def __init__(self, execution_id):
        self.mongoDBObject = MongodbOperation()
        self.azureBlobObject = AzureBlobManagement()
        self.execution_id = execution_id
        pass

    def log(self, database_name, collection_name, log_message):
         try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log = {
                'Log_updated_date': self.now,
                'Log_update_time': self.current_time,
                'Log_message': log_message,
                'execution_id': self.execution_id
            }

            res = self.mongoDBObject.insertRecordInCollection(database_name,collection_name,log)
            if res > 0:
                return True
            else:
                log = {
                     'Log_updated_date': [self.now],
                     'Log_update_time': [self.current_time],
                     'Log_message': [log_message],
                     'execution_id': self.execution_id
                }
            self.azureBlobObject.saveDataFrametoCSV("db-fail-log","log_"+self.execution_id, pd.DataFrame(log),
                                                    mode="a+", header=True)
            return True
         except Exception as e:

             log = {
                 'Log_updated_date': [self.now],
                 'Log_update_time': [self.current_time],
                 'Log_message': [log_message],
                 'execution_id': self.execution_id
             }
             log["Log_message"][0] = log["Log_message"][0] + str(e)

             self.azureBlobObject.saveDataFrametoCSV("db-fail-log",
                                                     "log_" + self.execution_id, pd.DataFrame(log),
                                                     mode="a+", header=True)


