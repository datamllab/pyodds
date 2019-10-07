import taos
import pandas as pd
import time

def connect_server(host,user,password):
    # Connect to TDengine server.
    #
    # parameters:
    # @host     : TDengine server IP address
    # @user     : Username used to connect to TDengine server
    # @password : Password
    # @database : Database to use when connecting to TDengine server
    # @config   : Configuration directory
    conn = taos.connect(host,user,password,config="/etc/taos")
    cursor = conn.cursor()
    return conn,cursor

@profile
def query_demo_data(conn,consur,database,table):

    # use database
    try:
        consur.execute('use %s' %database)
    except Exception as err:
        conn.close()
        raise (err)

    current_time = time.clock()

    # insert data
    cursor.execute("select * from %s.%s where %s >= \'%s\' and %s <= \'%s\' " % (
    database, table, 'ts', '2018-08-01 00:00:00', 'ts', '2018-08-15 00:00:00'))

    # Column names are in c1.description list
    cols = cursor.description
    # Use fetchall to fetch data in a list
    data = cursor.fetchall()

    cursor.execute("select * from %s.%s where %s >= \'%s\' and %s <= \'%s\' " % (
    database, table, 'ts', '2018-08-01 00:00:00', 'ts', '2018-08-15 00:00:00'))

    print ('Total query cost: %.6f s' %(time.clock() - current_time))

if __name__ == '__main__':
    conn,cursor=connect_server('127.0.0.1','yli','0906')
    query_demo_data(conn,cursor,'rtdb','rttable')