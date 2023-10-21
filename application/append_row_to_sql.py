import mysql.connector
import numpy as np

def insert_row_into_db(arr, table_name, db_config):
  if arr.shape != (5,):
      raise ValueError("The array must have shape (5,)")
  
  # Connect to MySQL database
  mydb = mysql.connector.connect(
      host=db_config['host'],
      user=db_config['user'],
      password=db_config['password'],
      database=db_config['database']
  )
  
  mycursor = mydb.cursor()
  
  # Insert data
  sql = f"INSERT INTO {table_name} (col1, col2, col3, col4, col5) VALUES (%s, %s, %s, %s, %s)"
  val = tuple(arr.tolist())
  mycursor.execute(sql, val)
  
  # Commit the transaction
  mydb.commit()

  print(f"1 record inserted, ID: {mycursor.lastrowid}")
    
if __name__ == "__main__":
  """
  python your_script.py --array 1 2 3 4 5 --table your_table_name --host localhost --user username --password your_password --database your_database
  """
  import argparse
  parser = argparse.ArgumentParser(
    description="Insert a row of 5 numbers into a MySQL table.")
  
  parser.add_argument(
    "--array", type=list, nargs=5, required=True, help="NumPy array of shape (5,)")
  parser.add_argument(
    "--table", type=str, required=True, help="MySQL table name")
  parser.add_argument(
    "--host", type=str, required=True, help="MySQL host")
  parser.add_argument(
    "--user", type=str, required=True, help="MySQL user")
  parser.add_argument(
    "--password", type=str, required=True, help="MySQL password")
  parser.add_argument(
    "--database", type=str, required=True, help="MySQL database name")

  args = parser.parse_args()

  arr = np.array(args.array)
  table_name = args.table
  db_config = {
      'host': args.host,
      'user': args.user,
      'password': args.password,
      'database': args.database
  }

  insert_row_into_db(arr, table_name, db_config)