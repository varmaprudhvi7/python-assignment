import sqlite3
import pandas as pd 
from sqlalchemy import create_engine
import numpy as np
from bokeh.plotting import figure, show


""" 
Connection function
input parameter : need to give name of the database as an input
Returns : It creates the database and connect to the database.
Here we have used exception block to print error message,if the 
program fails to connect with sqlite database.

"""
def connection(database_name):
    try:
        cnx = create_engine('sqlite:///'+database_name).connect()
        return cnx
    except:
        print("Oops! error occurred while connecting")
     
"""
CSV to SQLITE convertion function
input parameters : database_name
input parameters at run time : no of tables, table names, location of csv files.
Returns : convert csv files to Sqlite datatables

"""

def csv_sql(database_name):    
    no_of_tables = int(input("enter no of table need to be stored in database :"))
    con = sqlite3.connect(database_name)
    cursor = con.cursor()
    i = 0
    for i in range(no_of_tables):
        file_dir = input("enter the file_dir :")
        table_name = input("enter the table name :")
        cursor.execute("DROP TABLE IF EXISTS %s;" % table_name)
        with open(file_dir, 'r') as file:
            split_lines = file.readline()[:-1].split(',')
            read_lines = file.readlines()
            db = [tuple(read_lines[i][:-1].split(',')) for i in range(len(read_lines))]
            print(split_lines)
        header = ','.join(split_lines)
        cursor.execute("CREATE TABLE IF NOT EXISTS %s (%s);" % (table_name,header))
        cursor.executemany("INSERT INTO %s (%s) VALUES (%s);" % (table_name,header,('?,'*len(split_lines))[:-1]), db)
    con.commit()
    con.close()

"""
Uploading Dataframe to sqlite 
Input parameter : database_name, temp3(The final output test table in dataframe format)
Return : saves the final test dataframe file in sqlite database

"""
    
def df_sql(database_name,temp3):
    con = sqlite3.connect(database_name)
    temp3.to_sql(name='output_table', con=con)

    

""" 
Function to sort 4 best ideal functions
Input parameters : training table, ideal table
Returns : Sorted function 4 ideal functions from 50 ideal functions

"""

def ideal_fifty(df_train, df_ideal):
    column_ytrain = list(df_train.columns.values)
    column_yideal = list(df_ideal.columns.values)
    list_of_mse = [] 
    for z in column_ytrain[1:]:  
        for j in column_yideal[1:]:
            y = df_ideal[j]
            y_bar = df_train[z]
            summation = 0  
            n = len(y) 
            for i in range (1,n): 
                d = int(float(y[i])) - int(float(y_bar[i]))  
                sd = d**2 
                summation = summation + sd  
            MSE = [summation/n,j,z]    
            list_of_mse.append(MSE)
            print ("The Mean Square Error is: " , MSE, j, z)           
    sort_orders = sorted(list_of_mse, key=lambda x: x[0])
    sorted_fun =[]
    for i in sort_orders[:4]:
        sorted_fun.append(i)
    return sorted_fun
             
     
""" 
Processing the 4 best ideal functions
Input parameters : test table, ideal_table, output of ideal_fifty function
Returns : four ideal functions in our prefered format to map with the test data
    
"""

def processing_data(df_test,df_ideal,temp1):
    four_ideal_fun =[]
    for i in temp1:
        four_ideal_fun.append(df_ideal[i[1]])  
    return four_ideal_fun
       

""" 
Function to Map test data with training dataset
Inout parameters : The output of processing_data, here we have used inheritance concept
Returns : final output test file in data frame.
    
"""
def test(processing_data):
    test_set =[]
    test_dict = dict(zip(df_test.x, df_test.y))
    four_ideal_func = pd.DataFrame(temp2)
    four_ideal_func = np.transpose(four_ideal_func)
    x_ideal = pd.DataFrame(df_ideal["x"])
    list_of_ideal_cols = [*four_ideal_func]
    four_ideal_func.insert(0,"x",x_ideal)
    ideal_dict_y = four_ideal_func.set_index('x').T.to_dict('list')
    for key in test_dict:
        least_value =[]
        for i in range(len(ideal_dict_y[key])):
            test_set_1=float(test_dict[key]) - float(ideal_dict_y[key][i])
            test_set_1 = test_set_1**2
            test_set_1 = [key,float(test_dict[key]),test_set_1,list_of_ideal_cols[i]]
            least_value.append(test_set_1) 
            print(test_set_1)
            print ("The Mean Square Error is: " ,float(ideal_dict_y[key][i]))
            print ("The Mean Square Error is: " ,float(test_dict[key]))
        least = min(least_value, key=lambda x: x[2])
        test_set.append(least)
    df_last = pd.DataFrame (test_set, columns = ['x','y','y_ideal','No of ideal func'])
    return df_last



if __name__ == "__main__":
    
    database_name = input("enter the database name :")
    #database_name = "dataset.db" 
    csv_sql() 
    connection = connection(database_name)
    df_train = pd.read_sql_table('train', connection, index_col=False).astype(float)
    df_test = pd.read_sql_table('test', connection, index_col=False).astype(float)
    df_ideal = pd.read_sql_table('ideal', connection, index_col=False).astype(float)
    temp1 = ideal_fifty(df_train, df_ideal)
    temp2 = processing_data(df_test,df_ideal,temp1)
    temp3 = test(processing_data)
    df_sql(temp3)
      

"""  Using Bokeh visualization comparing test data with sorted four ideal functions"""

x = df_train["x"]
test_x = df_test["x"]
test_y = df_test["y"]
y1 = df_train["y1"]
y2 = df_train["y2"]
y3 = df_train["y3"]
y4 = df_train["y4"]
iy43 = df_ideal["y43"]
iy13 = df_ideal["y13"]
iy20 = df_ideal["y20"]
iy47 = df_ideal["y47"]
 
graph = figure(title = "Test vs 4 best ideal functions")

graph.circle(x,iy43, color='red', legend_label='training Function y43')
graph.circle(x,iy47, color='black', legend_label='Ideal Function y47')
graph.circle(x,iy20, color='yellow', legend_label='training Function y20')
graph.circle(x,iy13, color='green', legend_label='training Function y13')
graph.circle(test_x,test_y, color='blue', legend_label='testing')

show(graph)


""" Pre-processing coding"""


""" 
Input parameter: datatable 
Returns : List of outliers

"""
   
def find_outlier(data):
    q1, q3 = np.percentile(sorted(data), [25, 75])
    iqr = q3 - q1
    lower_quantile = q1 - (1.5 * iqr)
    upper_quantile = q3 + (1.5 * iqr)
    outliers = [x for x in data if x <= lower_quantile or x >= upper_quantile]
    print(outliers)
    
    

""" Prints the missing values - Data preprocessing"""   
print(df_train.isnull().sum().sum())
print(df_test.isnull().sum().sum())
print(df_ideal.isnull().sum().sum())
    
    
""" Scaling - Data preprocessing """
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaling = df_train[['x', 'y1','y2','y3','y4']]
scaling = ss.fit_transform(scaling)
print(scaling)

""" 
Input parameters :  training table
Returns : duplicate values in the data table 

"""
duplicate_values = df_train[df_train.duplicated()]
print("Duplicate_rows :")
print(duplicate_values)


""" Outliers """

outlier = figure(title = "Bokeh Scatter Graph")
# plotting the graph
outlier.scatter(x,y1, color = 'red' , legend_label='training Function y1' )
outlier.scatter(x,y2, color = 'yellow' , legend_label='training Function y2')
outlier.scatter(x,y3, color = 'blue' , legend_label='training Function y3')
outlier.scatter(x,y4, color = 'green' , legend_label='training Function y4')

# displaying the model
show(outlier)

find_outlier(df_train['y2'])
print(df_train.describe())
print(df_ideal.describe())
print(df_train.summary())

