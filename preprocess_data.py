#import dependencies
import pytry
import pandas as pd
import numpy as np
import pickle
import os

        
#Function for calculating rolling mean and adding it to the dataframe
def add_roll_mean(df):
    #from dataframe, get the rewards for each trial
    rewards_over_eps = df['episodes ']

    #create empty lists for recording new values
    Roll_mean=[]
    lst=[]

    #for each trial
    for i in range(len(rewards_over_eps)):
        #get rewards for trial i
        x = rewards_over_eps[i]
        #the data was saved as a string so convert it to an array
        x = x.replace('[','')
        x = x.replace(']','')
        array = np.fromstring(x, dtype=float, sep=",")
        #convert array to dataframe
        array_df = pd.DataFrame(array)

        #calculate rolling mean and add to one of the lists
        Roll_mean.append(array_df[array_df.columns[0]].rolling(100).mean())

        #copy the rolling mean data and convert to an array
        a = np.asarray(Roll_mean[i]).copy()
        #add the new array to the second list
        lst.append(a)

    #create new column with rolling mean data
    df['roll_mean'] = lst

    #return dataframe
    return df

#Function for fetching data from the txt files containing the data from the NNI experiments
def txt_to_dataFrame(directory):
    i = 0
    #from the directory, grab each data file
    for filename in os.listdir(directory):
        #get path to data file
        filepath = os.path.join(directory, filename)

        #Read the data as a csv, separating the data names from the data 
        #Data format: 'dataname = data'
        data = pd.read_csv(filepath, sep='=')

        #Store the data in list
        vals = data.iloc[:18,1].tolist()

        #To initialise the dataframe
        if i == 0:

            #make sure the first file has all the data needed
            if len(vals) == 18:

                #Store the data names in a list
                head = data.iloc[:18,0].tolist()

                #Create a pandas data frame
                df = pd.DataFrame(vals)
                df = df.T

                #Set the column names
                df.columns = head

                i += 1

            #if the first file doesn't have all the data needed
            else:

                #print 'incomplete data' if the file is missing something
                #script will then move on to the next file
                print('Incomplete data in:',filepath)

        #for the rest of the data files, add the data to the dataframe
        else:
            try:
                df.loc[len(df.index)] = vals

            #if there is data missing, print 'incomplete data' and move to the next file
            except ValueError:
                print('incomplete data in:',filepath)

    #Add a column with the rolling mean reward to the dataframe
    df = add_roll_mean(df)

    return df
    
## Run the preprocessing ##
folder = input("Enter path to NNI data folder ")
print('path to folder:', folder)

for foldername in os.listdir(folder):
    #set path to data folder
    directory = os.path.join(folder, foldername)
    
    #Get useful information from the file path
    title = directory.split('\\')[3] #filename
    task = title.split('_')[0] #the RL task done
    rule = title.split('_')[1] #the TD rule used
    rep = title.split('_')[2] #the method for representing the state information
    
    #Convert the data in this folder to a pandas data frame
    print('Creating dataframe from:',  directory)
    df = txt_to_dataFrame(directory)
    
    #Add the TD rule and the method for representing the state to the data frame
    df['rule '][:] = rule
    df['rep '][:] = rep
    
    df_name = title+'_dataframe'
    
    #Save the data frame
    print('Saving data frame:', df_name)
    df.to_pickle('.\\data\\'+df_name)