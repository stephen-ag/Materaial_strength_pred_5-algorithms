import os, uuid
import yaml
import pandas as pd
from io import StringIO
import pickle
from azure.storage.blob import ContainerClient, BlobServiceClient,BlobClient, __version__
from azure.storage.blob import ContentSettings, ContainerClient
import dill

class AzureBlobManagement:
    #Myconnectionstring ="DefaultEndpointsProtocol=https;AccountName=storageaccountrgai197de;AccountKey=o56dC8V4h6GmP1bin7a9NO0e6pUWTJrf/lzO3ogX1vFOTnodTCKbZp5VsRHfCSSHDLF8XqMKP0wDVi82eiaT7Q==;EndpointSuffix=core.windows.net"

    Myconnectionstring = "DefaultEndpointsProtocol=https;AccountName=stephenaipy;AccountKey=Tsptjs6Bk1xLQPOriEFXXgmVnrgXSVYHID03HGWzs/00n8SQP18VnWTFnzVZ2Y7c+wqLo+L2Smjegp60qki9kQ==;EndpointSuffix=core.windows.net"
    video_container_name = "videos"
    container_name = "cement-strength-training-database"
    source_folder = "/home/gerald"

    def __init__(self, connectionstring =None):
        if connectionstring is not None:
            self.__connectionstring = connectionstring
        else:
            self.__connectionstring = AzureBlobManagement.Myconnectionstring
        self.blob_service_client = BlobServiceClient.from_connection_string(self.__connectionstring)
        self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
        print(self.dir_list)

    def getAllFileNameFromDirectory(self, directory_name):
        self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
        directory_name = directory_name.lower()

        if directory_name in self.dir_list:
            container_client = self.blob_service_client.get_container_client(directory_name)
            filename= [files.name for files in container_client.list_blobs()]
            print(filename)

            return filename


    def createDir(self,directory_name, is_replace = True):
        """
        directory_name: Name of folder (name should be in lower case)
        =======================================
        return True if direcory will be created
        """
        container_name = directory_name.lower()
        if container_name not in self.dir_list:
            self.blob_service_client.create_container(container_name)
        elif is_replace and container_name in self.dir_list:
            for file in self.getAllFileNameFromDirectory(container_name):
                print("!!! Moving file " + file + " to recycle-bin folder...")
                self.moveFileinDir(container_name,"recycle-bin",file )
                #print(file)
        else:

            raise Exception ( 'Error occured in createDir method in azure')

        self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
        return True

    def deleteDir(self,directory_name):
        """  :param directory_name: directory to be deleted
        :return True if directory is deleted.
        """
        try:
            directory_name = directory_name.lower()
            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
            if directory_name not in self.dir_list:
                return True
            else:
                self.blob_service_client.delete_container(container=directory_name)
                return True

        except Exception as e:
            raise Exception("Error Occured in class: AzureBlobManagement method:deleteDirectory error:" + str(e))


    def readCSVFilefromDir(self,directory_name,filename, drop_unnamed_col=False):
        try:
            directory_name = directory_name.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if directory_name not in self.dir_list:
                raise Exception("Error occured in class: AzureBlobManagement method:readCsvFileFromDirectory error:Directory {0} not found".format(directory_name))

            if filename not in self.getAllFileNameFromDirectory(directory_name):
                raise Exception("Error occured in class: AzureBlobManagement method:readCsvFileFromDirectory error:File {0} not found".format(filename))

            blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=filename)
            df= pd.read_csv(StringIO(blob_client.download_blob().readall().decode()))
            print("reading from the saved inputfile",df)
            #df.columns.to_list()
            if drop_unnamed_col is False:
                return df.copy()
            elif "Unnamed: 0" in df.columns.to_list():
                df.drop(columns=['Unnamed: 0'], axis=1,inplace=True)
                return df.copy()
            else:
                return df.copy()
        except Exception as e:
            raise Exception( "Error occured in class: AzureBlobManagement method:readCsvFileFromDirectory error:" + str(e))

    def deleteFilefromDir(self, directory_name,filename):

        """
          :param directory_name: directory to be deleted
          :return: True if directory is deleted.
        """

        try:
            directory_name = directory_name.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if directory_name not in self.dir_list:
                raise Exception(
                    "Error occured in class: AzureBlobManagement method:deleteFilefromDir error:specified Directory not found")
            if filename in self.getAllFileNameFromDirectory(directory_name):

                blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=filename)
                blob_client.delete_blob()
                print("Directory deleted ", directory_name)

            else:

                raise Exception("Error occured in class: AzureBlobManagement method:deleteFilefromDir error:File  not found")
        except Exception as e:
            raise Exception("Error Occured in class: AzureBlobManagement method:deleteFileFromDirectory error:"+str(e))

    def loadObject(self, directory_name,filename):
        """:param directory_name: Directory name
            Gets the model by referencing filename from the specified folder
          :param file_name: file name
          :return: object
          """
        try:
            directory_name = directory_name.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if directory_name not in self.dir_list:
                raise Exception(
                    "Error occured in class: AzureBlobManagement method:loadModel error:Directory {0} not found".format(
                        directory_name))
            if filename not in self.getAllFileNameFromDirectory(directory_name):
                raise Exception("Error occured in class: AzureBlobManagement method:loadModel error:File {0} not found".format(
                    filename))
            blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=filename)
            model = dill.loads(blob_client.download_blob().readall())

            return model
        except Exception as e:
            raise Exception("Error occured in class: AzureBlobManagement method:loadModel error:File {0} not found".format(
                        filename))


    def saveObject(self, directory_name, filename, object_name):
        """
               directory_name: Directory Name
               file_name: File Name
               upload the object to blob
               object: object to save  eg: any model or any class object can be saved
               -------------------------------------
               return True if Model is created
               """
        try:
            directory_name = directory_name.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if directory_name not in self.dir_list:
                self.createDir(directory_name)
            if filename in self.getAllFileNameFromDirectory(directory_name):
                blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=filename)
                blob_client.delete_blob()
            blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=filename)
            blob_client.upload_blob(dill.dumps(object_name))

            return True
        except Exception as e:
            raise Exception(
                "Error occured in class: AzureBlobManagement method:saveObject error:" + str(e))

    def saveDataFrametoCSV(self, directory_name, file_name, data_frame, **kwargs):
        """
        :param directory_name: source directory name
        :param file_name: file name in which datafarme need to be save
        :param data_frame: pandas dataframe object
        :return: True if dataframe saved into azure blob
        """

        try:
            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]
            allowed_keys = ['index', 'header', 'mode']
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

            directory_name = directory_name.lower()
            if file_name.split(".")[-1] != "csv":
                file_name = file_name + ".csv"

            if directory_name not in self.dir_list:
                self.createDir(directory_name)
            if file_name in self.getAllFileNameFromDirectory(directory_name) and 'mode' in self.__dict__.keys():
                if self.mode == 'a+':
                    df = self.readCSVFilefromDir(directory_name=directory_name, filename=file_name)
                    data_frame = df.append(data_frame)
                    if 'Unnamed: 0' in data_frame.columns:
                        data_frame = data_frame.drop(columns=['Unnamed: 0'], axis=1)

            if file_name in self.getAllFileNameFromDirectory(directory_name):
                blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=file_name)
                blob_client.delete_blob()

            blob_client = self.blob_service_client.get_blob_client(container=directory_name, blob=file_name)
            if "index" in self.__dict__.keys() and "header" in self.__dict__.keys():
                output = data_frame.to_csv(encoding="utf-8", index=self.index, header=self.header)
            elif "header" in self.__dict__.keys() and 'mode' in self.__dict__.keys():
                output = data_frame.to_csv(encoding="utf-8", header=self.header)
            else:
                output = data_frame.to_csv(encoding="utf-8")
            print(output)
            blob_client.upload_blob(output)
            return True
        except Exception as e:
            raise Exception("Error occured in class: AzureBlobManagement method:saveDataFrameTocsv error:" + str(e))

    def moveFileinDir(self, source_directory, target_directory, filename):
        try:

            #source_directory = source_directory.lower()
            #target_directory = target_directory.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if source_directory not in self.dir_list:
                raise Exception('source dir not found'+ str(source_directory))
            if filename not in self.getAllFileNameFromDirectory(source_directory):
                raise Exception('file not found'+ str(filename))

            if target_directory not in self.dir_list:
                self.createDir(target_directory)
            if filename in self.getAllFileNameFromDirectory(target_directory):
                blob_client = self.blob_service_client.get_blob_client(container=target_directory, blob=filename)
                blob_client.delete_blob()

            account_name = self.blob_service_client.account_name
            source_blob = (f"https://{account_name}.blob.core.windows.net/{source_directory}/{filename}")
            copied_blob = self.blob_service_client.get_blob_client(target_directory, filename)
            copied_blob.start_copy_from_url(source_blob)
            remove_blob = self.blob_service_client.get_blob_client(source_directory, filename)
            remove_blob.delete_blob()

            #print(account_name)
            #print(source_blob)

            return True
        except Exception as e:
            raise Exception("Error occured in class: AzureBlobManagement method:moveFileInDirectory error:" + str(e))


    def CopyFileinDir(self, source_directory, target_directory, filename ):
        """
        :param source_directory: source directory name
        :param target_directory: target directory name
        :param file_name: file name to be copied
        :return: True if file will be copied successfully
        """
        try:

            source_directory = source_directory.lower()
            target_directory = target_directory.lower()

            self.dir_list = [container_name.name for container_name in self.blob_service_client.list_containers()]

            if source_directory not in self.dir_list:
                raise Exception("Error occured in class: AzureBlobManagement method:copyFileInDirectory source dir not found" + str(source_directory))
            if filename not in self.getAllFileNameFromDirectory(source_directory):
                raise Exception(
                    "Error occured in class: AzureBlobManagement method:copyFileInDirectory error:Source File not found" + str(
                        filename))
            if target_directory not in self.dir_list:
                self.createDir(target_directory)
            if filename in self.getAllFileNameFromDirectory(target_directory):
                blob_client = self.blob_service_client.get_blob_client(container=target_directory, blob=filename)
                blob_client.delete_blob()

            account_name = self.blob_service_client.account_name
            source_blob = (f"https://{account_name}.blob.core.windows.net/{source_directory}/{filename}")
            copied_blob = self.blob_service_client.get_blob_client(target_directory, filename)
            copied_blob.start_copy_from_url(source_blob)
            return True
        except Exception as e:
            raise Exception("Error occured in class: AzureBlobManagement method:copyFileInDirectory error:" + str(e))


#a=AzureBlobManagement()
#c=a.deleteDir('lap-a24380db-b919-4872-9453-cf44e7962870')
##s=a.createDir('demoineuron-step')
#
#s= a.getAllFileNameFromDirectory('training-batch-files')
#s=a.readCSVFilefromDir('prediction-output-file','prediction.csv')
#prediction-batch-files
#count =0
#df=pd.DataFrame(s)
#df.to_csv('prediction_output11.csv')
#print(df)
#df =pd.DataFrame()
#for i in s:
#    print(i)
#   #df.append(i)
#    count=count+1#
###  # print(df)
#print("total file count",count)
#print(s)

#loc=os.listdir('../Prediction_Batch_files/')
