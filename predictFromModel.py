import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation

from AzureBlobStorage.AzureStorageMgmt import AzureBlobManagement
# from folder and python file name import class name
from application_logging.loggerDB import App_LoggerDB
#from folder and python file import class name
from application_logging.logger import App_Logger

class prediction:

    def __init__(self,path,execution_id):
        #self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        #self.log_writer = logger.App_Logger()
        #self.pred_data_val = Prediction_Data_validation(path)
        self.execution_id = execution_id
        self.log_database="strength_prediction_log"
        self.log_collection="prediction_log"
        self.log_db_writer=App_LoggerDB(execution_id)
        self.az_blob_mgt=AzureBlobManagement()
        if path is not None:
            self.pred_data_val = Prediction_Data_validation(path,execution_id)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_db_writer.log(self.log_database,self.log_collection,'Start of Prediction')
            print("start of prediction")
            data_getter=data_loader_prediction.Data_Getter_Pred(self.log_database,self.log_collection,self.execution_id)
            data=data_getter.get_data()

            path=""
            if data.__len__()==0:
                self.log_db_writer.log(self.log_database,self.log_collection,"No data was present to perform prediction existing prediction method")
                return path,"No data was present to perform prediction"

            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor=preprocessing.Preprocessor(self.log_database,self.log_collection,self.execution_id)

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            data  = preprocessor.logTransformation(data)
            print("after log Transformation")
            print(data)

            #scale the prediction data
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)

            print("standard scaling for data completed")
            print(data_scaled)

            #data=data.to_numpy()
            file_loader=file_methods.File_Operation(self.log_database,self.log_collection,self.execution_id)
            kmeans=file_loader.load_model('kkmeans')

            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            clusters=kmeans.predict(data_scaled)#drops the first column for cluster prediction
            data_scaled['clusters']=clusters
            clusters=data_scaled['clusters'].unique()
            result=[] # initialize blank list for storing predicitons
            # with open('EncoderPickle/enc.pickle', 'rb') as file: #let's load the encoder pickle file to decode the values
            #     encoder = pickle.load(file)

            for i in clusters:
                cluster_data= data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                print(model_name)
                model = file_loader.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)

            result = pandas.DataFrame(result, columns=['strength-Predictions'])

            #result = list(model.predict(cluster_data))
                #self.result = pandas.DataFrame(list(zip(result)), columns=['Prediction'])
                #for val in (model.predict(cluster_data.values)):
                #    result.append(val)
                #print(self.result.shape)
            print("results after prediction with prediction columns")
            print(result)

            path="Prediction-Output-File"
            #result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.az_blob_mgt.saveDataFrametoCSV(path, "cement-strength-prediction.csv", result, header=True, mode="a+")

            self.log_db_writer.log(self.log_database,self.log_collection,'End of Prediction')
        except Exception as ex:
            self.log_db_writer.log(self.log_database,self.log_collection, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path ,result.head().to_json(orient="records")

            # old code
            # i=0
            # for row in data:
            #     cluster_number=kmeans.predict([row])
            #     model_name=file_loader.find_correct_model_file(cluster_number[0])
            #
            #     model=file_loader.load_model(model_name)
            #     #row= sparse.csr_matrix(row)
            #     result=model.predict([row])
            #     if (result[0]==-1):
            #         category='Bad'
            #     else:
            #         category='Good'
            #     self.predictions.write("Wafer-"+ str(wafer_names[i])+','+category+'\n')
            #     i=i+1
            #     self.log_writer.log(self.file_object,'The Prediction is :' +str(result))
            # self.log_writer.log(self.file_object,'End of Prediction')
            #print(result)




