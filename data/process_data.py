import sys
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Reads in two CSV's and merges them on the ID column.
    Returns the merged dataframe.'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df
    

def clean_data(df):
    '''Splits dataframe categories on ";" and creates a sparse matrix of 
    numbers. Drops duplicates and returns new dataframe'''
    categories = df['categories'].str.split(pat=";", expand=True)
    category_colnames = categories.iloc[0].str.split(pat="-", expand=True)[0]
    categories.columns = category_colnames
    for i in tqdm(range (0, len(categories))):
        categories.iloc[i] = categories.iloc[i].str.split(pat="-", expand=True)[1]
    categories = categories.astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''Saves dataframe to sql using sqlalchemy'''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_table', engine, index=False)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)   
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()