#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:08:01 2019

@author: Jingjing Tang
"""
import os
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src
import src.SupportFunctions as sf
import sys
import warnings

warnings.filterwarnings("ignore")


###############################################################################
parser = argparse.ArgumentParser(description='This program produces a set of figures for fluscore error analysis. \n \
The minimum input files required to run the program are: \n \
\t [--group1] 1) A csv file of fluscore information of group 1 with separator \',\'. \n \
\t [--group2] 2) A csv file of fluscore information of group 2 with separator \',\'. \n \
The program will output figures of the analysis result [--outputpath]. \n \
\t [--name1]  The name of group 1. \n \
\t [--name2]  The name of group 2. \n \
\t [--targets] A target or multiple targets(use \' \ \' to connect them) of interest \n \
Example: \n \
\t$python main.py --group1 example/Delphi-Epicast2.csv --group2 example/Delphi-Stat.csv \n \
Please see the documentation for more details.',formatter_class=RawTextHelpFormatter)

#INPUT ARGUMENTS
grpInput = parser.add_argument_group('Input')
grpInput.add_argument('--group1', type=str, dest='dirgroup1',default= "", help='Enter the path and name of fluscore file for group 1.')
grpInput.add_argument('--group2', type=str, dest='dirgroup2',default= "", help='Enter the path and name of fluscore file for group 1.')
grpInput.add_argument('--name1', type=str, dest='name1',default= "Group 1", help='Enter the path and name of fluscore file for group 1.')
grpInput.add_argument('--name2', type=str, dest='name2',default= "Group 2", help='Enter the path and name of fluscore file for group 1.')
grpInput.add_argument('--targets', type=str, dest='targets', default= "Season_peak_week\Season_onset\Season_peak_percentage\\1_wk_ahead\\2_wk_ahead\\3_wk_ahead\\4_wk_ahead",
                      help='Enter the targets(Season_peak_week\Season_onset\Season_peak_percentage\\1_wk_ahead\\2_wk_ahead\\3_wk_ahead\\4_wk_ahead) of interest.')
grpInput.add_argument('--competition_week', type=str, dest='compweek', default = "",help='Enter the competition weeks of interest.')
grpInput.add_argument('--location', type=str, dest='location', default= "", help='Enter the locations of interest.')
grpInput.add_argument('--scoretype', type=str, dest='scoretype', default= "logscore", help='Enter the score type of interest(logscore or score).')
grpInput.add_argument('--xlabel', type=str, dest='xlabel', default = "",help='Enter the xlabel(location or competition_week) of interest when considering summary plot.')
grpInput.add_argument('--summarytype', type=str, dest='summarytype', default = "",help='Enter the summarytype(win_rate or similar_rate) of interest when considering summary plot.')
grpInput.add_argument('--similarrate', type=float, dest='rate', default= 0.05, help='The threshold to define whether the performance of two groups are similar.')
grpInput.add_argument('--annot', type=str, dest='annot', default= 'False', help='The threshold to define whether the performance of two groups are similar.')


#OUTPUT ARGUMENTS
grpOutput = parser.add_argument_group('Output')
grpOutput.add_argument('--outputpath', type=str, default="", dest='outputpath', help='Enter the path to sore the output figures.')

args = parser.parse_args()
###############################################################################

if len(sys.argv)==1:
    parser.print_help()
    sys.stderr.write("\nNo arguments were supplied to the program. Please see the usage information above to determine what to pass to the program.\n")
    sys.exit(1)

#args.dirgroup1 = 'example/Delphi-Epicast2.csv'
#args.dirgroup2 = 'example/Delphi-Stat.csv'

#args.dirgroup1 = 'example/Delphi-Stat.csv'
#args.dirgroup2 = '../data/2018/natreg/LANL-Dante.csv'

dirgroup1 = args.dirgroup1 
dirgroup2 = args.dirgroup2
names = [args.name1, args.name2]
scoretype = args.scoretype
rate = args.rate
annot = args.annot
targets = args.targets.split('\\')


xlabels = ['competition_week', 'location']

iMode = -1
###############################################################################
# Mode 0: Any problem exits with the input

dfs = []
datatype = []
for path in [dirgroup1, dirgroup2]:
    if not src.check_file(path):
        sys.stderr.write( path + " does not exist. Please check the path to this file." )
        iMode = 0
    else:
        check_data_result = src.check_data(path)
        if type(check_data_result) != int:
            data = check_data_result
            data = src.check_targets(data)
            datatype.append(src.check_locations(data))
            dfs.append(data)
        else:
            if check_data_result == -2:
                sys.stderr.write( path + ' dose not have a log-score column')
                iMode = 0
            else:
                sys.stderr.write( path+ ' does not have a %s column'%check_data_result)
                iMode = 0
            
if np.unique(datatype).shape[0] > 1:
    iMode = 0
else:
    datatype = datatype[0]


# The output folder name is better related to both the names of the two groups
# and the data type    
if args.outputpath == '':
    outputpath = '%s_%s_%s'%(names[0], names[1], datatype)
else:
    outputpath = args.outputpath
#Check or create the folder for output figures
diroutput = src.check_create_dir( outputpath )
###############################################################################
# Choose program mode based on files supplied by the user. 
if iMode != 0:   
    if args.compweek!="" or args.location!="":
        iMode = 2
    else:
        iMode = 1

###############################################################################
# Mode 1: Generate all of the analysis figure
if iMode == 1:
    
    # Season onset is not availabel for states and hospitalization data now
    # might be changed later
    if datatype != 'Region':
        targets.remove('Season_onset')
        
    df1 = dfs[0]
    df2 = dfs[1]
    differencetables = {}
    for i in range(len(targets)):
        targets[i] = targets[i].replace('_', ' ')
        target = targets[i]
        dirtarget = src.check_create_dir( diroutput + os.sep + target)
        
        dfmin = np.min(df1[scoretype] - df2[scoretype])
        dfmax = np.max(df1[scoretype] - df2[scoretype])
        
        #Create difference table, only create once
        differencetable = sf.create_difference_df(df1, df2, target, scoretype)
        differencetables[target] = differencetable
        
        ### Generate Difference Heatmap
        heatmap = sf.DifferenceHeatmap_Target(differencetable, dfmin, dfmax, target, names, datatype, scoretype, annot)
        plt.savefig(dirtarget + '/DifferenceHeatmap.png', bbox_inches='tight')
        
        lineplot = sf.AvgDiffLineplot(df1, df2, target, names, datatype, scoretype = 'logscore')
        plt.savefig(dirtarget + '/CompWeek_AvgDiffLinePlot.png', bbox_inches  = 'tight')
            
        
        ### Generate Different Boxenplot
        boxenplot = sf.LocationBoxenplot_Target(differencetable, target, names, datatype, scoretype = 'logscore')
        plt.savefig(dirtarget + '/BoxenPlot.png', bbox_inches='tight')
        
        ### Generate Summary Plot
        for xlabel in xlabels:
            SummaryBarPlot = sf.SummaryBarPlot_Target(differencetable, target, xlabel, names, datatype, scoretype = 'logscore', rate = 0.05)
            plt.savefig(dirtarget + '/%s_SummaryBarPlot.png'%xlabel, bbox_inches='tight')
    
    dirtargets = src.check_create_dir( diroutput + os.sep + 'Combined')
        
    if args.summarytype == '':
        summarytypes = ['win_rate', 'similar_rate']
    else:
        summarytypes = [args.summarytype]
    
    for xlabel in xlabels:
        for summarytype in summarytypes:
            SummaryDotPlot = sf.SummaryDotPlot(df1, df2, xlabel, targets, names, datatype, scoretype, summarytype, rate = 0.05)
            plt.savefig(dirtargets + '/SummaryDotPlot_%s_%s.png'%(xlabel, summarytype), bbox_inches='tight')

        
        
        
        
        
    

