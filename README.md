# Fast_RL_with_Bio_Based_Reps
Companion repository to ICCM paper "Fast Online Reinforcement Learning with Biologically-Based State Representations".

Contributors: Dr. M. Bartlett, Dr. T. Stewart & Dr. J. Orchard
Affiliation: University of Waterloo, ON, Canada

Repository to accompany Bartlett, Stewart & Orchard (2022) "Fast Online Reinforcement Learning with Biologically-Based State Representations" ICCM Paper (LINK).

## Requirements:

You will need to have Jupyter Notebook installed in order to run these scripts. Recommended to install [Anaconda](https://www.anaconda.com/products/individual). 

* Python 3.5+
* OpenAI Gym
* Gym MiniGrid
* PyTry
* Nengo
* NNI
* Numpy
* Pandas
* Pickle
* Matplotlib
* Seaborn
* Plotly.express
* Pathlib
* Sys
* OS

## Scripts and How-To Guide:

Download repository as a zip file or clone repository by clicking the "Code" button on the top right. <br>

### Run the Network

You can run the network from the command line using ac_network_run.py - in the terminal, navigate to the directory containing these scripts and run the command "python ac_network_run.py". 

Running this command without any changes to the file will result in you first being prompted to provide a directory for saving the data and figures. The AC network will then try to solve the MiniGrid task using the TD($\lambda$) learning rule and Spatial Semantic Pointers (SSPs) to represent the state. The preset parameter values were identified as being one of the best performing parameter sets for this configuration. 
A plot showing the total reward received in each of the 500 learning trials will be presented once the AC network has finished, and the plot will also be saved as ep_rewards.pdf in the directory you provided. 

### Experiment

The python files named "config..." and "exp..." are the scripts needed to re-run the NNI experiments which perform a search of the hyperparameter space for 8 configurations of the AC network. 
These configurations vary in terms of the learning rule used (TD(0) vs. TD($\lambda$)) and the method for representing the state information (baseline vs. one hot vs. random SSPs vs. grid cells). 
In order to replicate the experiment reported in Bartlett, Stewart & Orchard (2022), you simply need to run each "config...." file from the command line. For example: 

```
nnictl --config config_MGTD0Baseline_long.yml
```

This will start the NNI parameter search experiment which will run for 12 hours. <br>
Once the experiment starts running, a localhost url will be provided in the terminal which you can go to in order to monitor the progress of the experiment. 

**Important: you must change the data_dir in each "exp..." file before running these NNI experiments. Change to a unique data directory where you would like the data from the experiment to be saved.** 
We recommend using the following format for naming the data_dir: 

Path('../DATA_FOLDER/'TASK_RULE_REP), 

where:

* DATA_FOLDER = the parent folder where you want everything to be saved. Set this to whatever you want
* TASK = the task being learned:
    * 'MG' for MiniGrid
* RULE = the rule used:
    * 'TD0' for TD(0)
    * 'TDLam' for TD($\lambda$)
* REP = the representation used:
    * 'Baseline' for one hot with no neurons 
    * '1H' for one hot
    * 'SSP' for SSP
    * 'Grid' for grid cells

### Data Preprocessing:

Before running the analysis you will need to run a pre-processing script on the data in order to convert it from text files into pandas dataframes. You can do this either from the command line or in a Jupyter notebook.

The relevant scripts for this contain two main functions. 
The *txt_to_dataFrame* function will retrieve the text files saved by the NNI experiments and convert them to pandas data frames. 
Then the *add_roll_mean* function will add a column to the data frame that contains the rolling average reward gained for each run (the average reward over each 100 learning trials). 

#### Command Line:

To run the pre-processing from the command line, simply run:

```
python preprocess_data.py
```

You will be prompted to enter the path to the relevant data folder.

#### Notebook: 

Alternatively, you can step through the pre-process by running Preprocess_Data.ipynb from a Jupyter notebook. 

You will need to manually set the destination folder where you would like the data to be saved in cell 4:

```python
folder = '.\\data\\main_experiment'
```

### Analysis:

There are two analysis scripts. One that can be run from the command line, and one that is a Jupyter notebook. 

#### Command Line:

The analysis_main.py script can be run from the command line in order to produce pdf's of the figures presented in Bartlett, Stewart & Orchard (2022). 

```
python analysis_main.py
```

You will be prompted to provide the path to the folder containing your pandas data frames. 
The script will then create plots of:

*  the optimization curves for each NNI experiment
*  the number of trials it took to reach the goal rolling mean of 0.95 for the top 2% of parameter combinations

These plots will then be saved in a new "figures" folder. 

#### Notebook: 

To get a more detailed look at the results from the NNI experiments we recommend you run the Jupyter notebook script Analysis_Main.ipynb. 

First you will need to enter the path to the folder containing the pandas data frames with your data from the NNI experiments in cell 6:

```python
data_folder = Path('./data/')
```

The notebook is then split into 3 main parts. 
The first 2 sections divide the analysis between the two learning rules TD(0) and TD($\lambda$).

For each of these sections, the first step is to calculate, for each run, the number of trials that were needed to reach the goal rolling mean reward of 0.95. 
This data gets added to each dataframe in a new column named 'goal_reached'. 
Once this is done, and the data contained in the data frames is converted to numeric (where applicable), plots of the optimization curves for each NNI experiment are produced. 
For each learning rule, a total of 4 optimization plots are provided, one for each method of representing the state (baseline, one hot, SSPs and grid cells). 
We also print the minimum number of trials that were needed to reach the goal, which gives an idea of what the best combination of parameter values was able to achieve. 

We then take a closer look at the 'best' parameter combinations. 
First we figure out what the top 2% of runs were by getting those runs that reached the goal rolling mean in the fewest number of trials. 
For each representation method we print out the parameter values for this top 2%, and then produce a table showing the minimum and maximum value for each parameter within this top 2%. 
This table gives an idea of the stability of the values identified as being 'best' -- smaller ranges would indicate that the NNI experiment did identify a region of the parameter space that worked well, rather than just getting lucky a bunch of times. 

In order to further explore the stability of these values, we then get all of the NNI runs whose parameter values all fell within these ranges. 
Plotting the number of trials to reach goal for all of these NNI runs that used these 'best' parameter combinations illustrates whether using values within these ranges reliably produced good performance. 

Finally, for the best 2% of experiments, we plot the reward schedule across all learning trials in each run to make sure that once the goal rolling reward was reached, the network was able to maintain this good performance. 

The final section in this notebook provides the combined plots presented in the published paper. Namely, the optimization plots for all 8 experiments, and the trials-to-goal plots for the best 2% of runs in all 8 experiments. 

## Citation:

Please use this bibtex to reference the paper: 

<pre>
 @inproceedings{bartlett2022_RLNNI,
  author = {Bartlett, Madeleine and Stewart, Terrence C and Orchard, Jeff},
  title = {Fast Online Reinforcement Learning with Biologically-Based State Representations},
  year = {2022},
  booktitle={In-Person MathPsych/ICCM 2022.},
  url={mathpsych.org/presentation/838.}
 }
</pre>
