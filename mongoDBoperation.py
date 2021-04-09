
import os
import pandas as pd
import json
import pymongo
# importing mongodb file

class MongodbOperation:

    def __init__(self):
        self.user_name = "step1234"
        self.password = "step1234"
        #self.DB_URL = 'mongodb+srv://{0}:{1}@cluster0.tpqna.mongodb.net/<dbname>?retryWrites=true&w=majority'.format(self.user_name, self.password)
       #self.client = pymongo.MongoClient("mongodb+srv://{0}:{1}@cluster0.tpqna.mongodb.net/Projectdb?retryWrites=true&w=majority")
        self.DB_URL = 'mongodb+srv://{0}:{1}@stepdata.cy5xk.mongodb.net/<dbname>?retryWrites=true&w=majority'.format(self.user_name, self.password)

        #db = self.client.get_database('Projectdb')

    def getDataBaseClientObject(self):
        """ Return pymongoClient object to perform action with MongoDB"""
        try:
            self.client = pymongo.MongoClient(self.DB_URL)
            return self.client
        except Exception as e:
            raise Exception("Error occurred in class: MongoDBOperation method:getDataBaseClientObject error:Failed to "
                            "create database connection object-->" + str(e))


    def closeDataBaseClientObject(self, obj_name):

        """  obj_name : pymongo client
            DESCRIPTION.pymongo client object/ Exception  Failed to close database connection-->. Returns   bool  True
            if connection closed """
        try:
            obj_name.close()
            return True
        except Exception as e:
            raise Exception("MongoDBOperation method:closeDatabaseClientobject error:Failed to close database "
                            "connection-->" + str(e))

    def checkDatabase(self, client, db_name):
        try:
            if db_name in client.list_database_names():
                a=client.list_database_names()
                print("database " + db_name + " is present in the list " + str(a))
                return True

            else:
                return False
        except Exception as e:
            raise Exception("Error occurred in class: MongoDBOperation method:checkDataBase error:Failed to check "
                            "database exist or not" + str(e))

    def createDatabase(self, client, dbname):

        """     client: client object of database       db_name:data base name"""
        try:

            a=client[dbname]
            print("databse created")
            #print(a)
            return a
        except Exception as e:

            raise Exception("Error occurred in class: MongoDBOperation method: createDatabase error:" + str(e))

    def createCollectionInDatabase(self, database, collection_name):
        """        database:database  /  collection_name: name of collection   return: collection object"""
        try:
            q=database[collection_name]
            #print("collection created")
            #print(q)
            return q
        except Exception as e:
            raise Exception("Error occurred in class: MongoDBOperation method:createCollectionInDatabase  error: ")

    def checkExistingCollection(self, collection_name, database):

        """
            collection_name : collection_name
            DESCRIPTION.collection name which needs to verify database : TYPE DESCRIPTION.database in which collection needs to check for existence
        Raises
        ------
        Exception DESCRIPTION. Returns  bool= true if collection present in database.
        """
        try:
            collection_list = database.list_collection_names()
            if collection_name in collection_list:
                print(f"Collection:'{collection_name}' is in Database: & exists")
                return True
            else:
                print(f"Collection:'{collection_name}' in Database: does not exists OR \n        no documents are present in the collection")
                return False
        except Exception as e:
            raise Exception("Error occurred in class: MongoDBOperation method:checkExistenceCollection error:" + str(e))

    def getCollection(self, collection_name, database):

        """
        collection_name:collection name
        database=database
        ------------------------------------------
        return collection object
        """
        try:
            collection = self.createCollectionInDatabase(database, collection_name)
            print("collection Object created")
            return collection
        except Exception as e:
            raise Exception("Error occured in class: MongoDBOperation method:getCollection error:Failed to find collection" +
                            str(e))


    def isRecordPresent(self, db_name, collection_name, record):
        try:
            client = self.getDataBaseClientObject()
            database = self.createDatabase(client, db_name)
            collection = self.getCollection(collection_name, database)
            recordfound = collection.find(record)
            if recordfound.count() > 0:
                client.close()
                print("Record found")
                return True
            else:
                client.close()
                print("Checking found that, No Record present")
                return False

        except Exception as e:
            raise Exception("Error occured in class: MongoDBOperation method:isRecordPresent error:Failed to insert record " +str(e))

    def createOneRecord(self, collection, data):
        try:
            collection.insert_one(data)
            print("record inserted")
            return 1
        except Exception as e:
            raise Exception("Error occured in class: MongoDBOperation method:createOneRecord error:Failed to insert record " + str(e))


    def createMutlipleRows(self, collection, data):

        collection.insert_many(data)
        return len(data)

    def insertRecordInCollection(self, db_name, collection_name, record):
        try:

            no_of_row_inserted = 0
            client = self.getDataBaseClientObject()
            database = self.createDatabase(client, db_name)
            collection = self.getCollection(collection_name, database)
            if not self.isRecordPresent(db_name, collection_name, record):
                no_of_row_inserted = self.createOneRecord(collection=collection, data=record)
            client.close()
            return no_of_row_inserted
        except Exception as e:
            raise Exception(
                "Error occured in class: MongoDBOperation method:insertRecordInCollection error:Failed to insert record " + str(
                    e))



    def dropCollection(self, db_name, collection_name):
        """   :param db_name: database name
        :param collection_name:  collection name
        :return: True if collection droped successfully.    """
        try:
            client = self.getDataBaseClientObject()
            database = self.createDatabase(client, db_name)
            if self.checkExistingCollection(collection_name,database):
                collection_name = self.getCollection(collection_name,database)
                collection_name.drop()
            return True
        except Exception as e:
            raise Exception("Error occured in class MongoDBOperation method drop collection error:"+str(e))


    def insertRecordsInCollection(self, db_name, collection_name, records):
        """ db_name: database name
                collection_name: collection name
                records: records to insert  """
        try:
            no_of_rows_inserted = 0
            client = self.getDataBaseClientObject()
            database = self.createDatabase(client, db_name)
            collection = self.getCollection(collection_name, database)

            for record in records:

                if not self.isRecordPresent(db_name,collection_name,record):

                    no_of_rows_inserted = no_of_rows_inserted + self.createOneRecord(collection=collection, data=records)
            client.close()
            return no_of_rows_inserted

        except Exception as e:
            raise Exception("Error occured in class: MongoDBOperation method:insertRecordsInCollection error:Failed to insert record " +
                            str(e))

    def insertDataFrame(self, db_name, collection_name, data_frame):
        """        db_name:Database Name
        collection_name: collection name
        data_frame: dataframe which needs to be inserted
        return:        """
        try:
            no_of_row_inserted = 0
            records = list(json.loads(data_frame.T.to_json()).values())
            client = self.getDataBaseClientObject()
            database = self.createDatabase(client, db_name)
            collection = self.getCollection(collection_name, database)
            collection.insert_many(records)

            """
            for record in records:
                if not self.isRecordPresent(db_name, collection_name, record):
                    no_of_row_inserted = no_of_row_inserted + self.insertRecordInCollection(db_name, collection_name,                                                                                          record)
            """

            return len(records)
        except Exception as e:
            raise Exception("Error occured in class: MongoDBOperation method:insertDataFrame error: Not able to insert dataframe into collection " +str(e))

    def getDataFrameofCollection(self, db_name, collection_name):
        """        Parameters
        ----------
        db_name : string
            DESCRIPTION. database name
        collection_name : string
            DESCRIPTION.collection name

        Returns
        -------
        Pandas data frame of  collection name present database.        """

        try:
            client = self.getDataBaseClientObject()
            #database= client.db_name
            database= self.createDatabase(client,db_name)
            collection = self.getCollection(collection_name=collection_name, database = database)
            df = pd.DataFrame(list(collection.find()))
            print(df)

            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], axis=1, inplace=True)
            print(df)
            return df
        except Exception as e:
            raise Exception(" Error in Class: MongoDBOperation method:getDataFrameofCollection error:  " + str(e))

##!checking for connection and view the database in mongodb
#inserting training schema to mongodb
#a=MongodbOperation()
#c=a.getDataBaseClientObject()
#dir=a.createDatabase(c,'Wafer-sys')
#col=a.createCollectionInDatabase(dir,'strength-schema-training')
#data ={	"SampleFileName": "cement_strength_08012020_120000.csv",
#	"LengthOfDateStampInFile": 8,
#	"LengthOfTimeStampInFile": 6,
#	"NumberofColumns" : 9,
#	"ColName": {
#		"Cement _component_1" : "FLOAT",
#		"Blast Furnace Slag _component_2" : "FLOAT",
#		"Fly Ash _component_3" : "FLOAT",
#		"Water_component_4" : "FLOAT",
#		"Superplasticizer_component_5" : "FLOAT",
#		"Coarse Aggregate_component_6" : "FLOAT",
#		"Fine Aggregate_component_7" : "FLOAT",
#		"Age_day" : "INTEGER",
#		"Concrete_compressive _strength" : "FLOAT"
#	}
#}
#
##data={'at':1,'sl':2,'no':3,"data":{'new':1,'old':2,'avg':3,'Wafer':"varchar",'Sensor - 1':" float"}}
##
#rec=a.createOneRecord(col,data)
#inserting prediction schema to mongodb
##!
#a=MongodbOperation()
#c=a.getDataBaseClientObject()
#dir=a.createDatabase(c,'Wafer-sys')
#col=a.createCollectionInDatabase(dir,'strength_schema_prediction')
#data ={	"SampleFileName": "cement_strength_08012020_120000.csv",
#	"LengthOfDateStampInFile": 8,
#	"LengthOfTimeStampInFile": 6,
#	"NumberofColumns" : 8,
#	"ColName": {
#	"Cement _component_1" : "FLOAT",
#		"Blast Furnace Slag _component_2" : "FLOAT",
#		"Fly Ash _component_3" : "FLOAT",
#		"Water_component_4" : "FLOAT",
#		"Superplasticizer_component_5" : "FLOAT",
#		"Coarse Aggregate_component_6" : "FLOAT",
#		"Fine Aggregate_component_7" : "FLOAT",
#		"Age_day" : "INTEGER"
#	}
#}


#data={'at':1,'sl':2,'no':3,"data":{'new':1,'old':2,'avg':3,'Wafer':"varchar",'Sensor - 1':" float"}}
#
#rec=a.createOneRecord(col,data)