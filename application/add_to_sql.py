import argparse
import pandas as pd
from sqlalchemy import create_engine

def read_txt_file(file_path):
    # Read space-delimited text file into a Pandas DataFrame
    df = pd.read_csv(
      file_path, delim_whitespace=True)
    return df

def insert_into_mysql(df, table_name, db_connection_string):
    # Create a database engine
    engine = create_engine(db_connection_string)

    # Insert DataFrame into MySQL table
    df.to_sql(table_name, engine, if_exists='append', index=False)

if __name__ == "__main__":
  """
  python your_script.py --file your_file.txt --table your_table --db-connection 'mysql+mysqlconnector://username:password@localhost/db_name'
  """
  parser = argparse.ArgumentParser(description="Insert a space-delimited text file into a MySQL table.")

  parser.add_argument("--file", type=str, required=True, help="Path to the space-delimited text file.")
  parser.add_argument("--table", type=str, required=True, help="Name of the MySQL table.")
  parser.add_argument("--db-connection", type=str, required=True, help="MySQL database connection string.")

  args = parser.parse_args()

  # Read text file into DataFrame
  df = read_txt_file(args.file)

  # Insert DataFrame into MySQL table
  insert_into_mysql(
    df, args.table, args.db_connection)

