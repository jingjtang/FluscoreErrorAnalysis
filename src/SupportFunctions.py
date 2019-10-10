#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:50:54 2019

@author: Jingjing Tang
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_difference_df(df1, df2, target, scoretype='logscore'):
    '''
    Helper function to create a merged table 
    with log-scores for both df1 and df2
    '''
    if target != "All":
        df1 = df1.loc[df1['target'] == target][['competition_week', 'location', scoretype]]
    #    df1 = df1.drop_duplicates()
        df2 = df2.loc[df2['target'] == target][['competition_week', 'location', scoretype]]
    #    df2 = df2.drop_duplicates()
    else:
        df1 = df1[['competition_week', 'location', scoretype]]
        df2 = df2[['competition_week', 'location', scoretype]]
    df = df1.merge(df2, how ='inner', on=['competition_week', 'location'])
    df['difference'] = df['%s_x'%scoretype] - df['%s_y'%scoretype]
    return df

#def create_diffrank_table_Target(df1, df2, target, number_of_rows, ranktype = 'better', scoretype = 'logscore'):
#    """
#    create a table to rank the difference in terms of 
#    different targets for two groups
#    """
#    df = create_difference_df(df1, df2, target, scoretype)
#    df['difference'] = df['%s_x'%scoretype] - df['%s_y'%scoretype]
#    df['abs_diff'] = abs(df['difference'])
#    if ranktype == 'better':
#        df = df.sort_values('difference', ascending=False) 
#    elif ranktype == 'worse':
#        df = df.sort_values('difference', ascending=True) 
#    else:
#        df = df.sort_values('abs_diff', ascending=False)  
#    if 'US National' in df['location']:
#        df = df.loc[df['location']!='US National']
#    print(df.head(number_of_rows,))

def DifferenceHeatmap_Target(df, dfmin, dfmax, target, names, datatype, scoretype = 'logscore', annot = 'False'):  
    '''
    Create a heatmap for the different between two groups
    in terms of a specific target
    df: difference table in terms of a specific target
    dfmin: the minimum difference value across all of the targets
    dfmax: the maximum difference value across all of the targets
    names: the groups names, names[0] for group1, names[1] for group2
    datatype: Region, State or Hospitalization
    annot: whether add annotations to the cells
    '''
    if datatype == 'State':
        xlabel = 'States'
        figuresize = (20, 20)
        column_order = np.sort(np.unique(df['location']))
    elif datatype == 'Hospitalization':
        column_order = ['0-4 yr', '5-17 yr', '18-49 yr', '50-64 yr', '65+ yr', 'Overall']
        xlabel = 'Age Groups'
        figuresize = (20, 12)
    elif datatype == 'Region':
        column_order = ['HHS Region %d'%i for i in range(1,11)] + ['US National']
        xlabel = 'Regions'
        figuresize = (20, 12)
    
    dfcenter = (0 - dfmin) / (dfmax - dfmin)

    heatmap_df = pd.DataFrame({'Competition Week':df['competition_week'],
                               xlabel: df['location'],
                               scoretype: df['difference']})
    heatmap_df = heatmap_df.pivot(columns = 'Competition Week' , index = xlabel, values = scoretype)
    heatmap_df = heatmap_df.reindex(column_order, axis=0)
    heatmap_df = heatmap_df.reindex(column_order, axis=0)
    if datatype == 'State':
        heatmap_df.loc['Average'] = heatmap_df.mean(axis = 0).values
    else:
        heatmap_df.loc['Average'] = heatmap_df.iloc[:-1,:].mean(axis = 0).values
    heatmap_df['Avg'] = heatmap_df.mean(axis = 1).values
    row_order = ['Avg'] + list(heatmap_df.columns[:-1])    
    heatmap_df = heatmap_df.reindex(row_order, axis=1)
    
    
    plt.figure(figsize = figuresize)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(heatmap_df, annot=False, cmap="RdBu_r",
                     center = 0, cbar=True, vmin = dfmin,  vmax = dfmax,
                     cbar_kws={"orientation": "horizontal", "shrink": 1, 
                               "aspect":40, "label": "Flu-%s Difference: Flu_%s(%s) - Flu_%s(%s)"%(scoretype, scoretype, names[0], scoretype, names[1])})
    ax.set_yticklabels(heatmap_df.index,rotation=0, fontsize = 11)
    ax.set_xticklabels(heatmap_df.columns,fontsize = 11, rotation = 0)
#    ax.xaxis.tick_top() # x axis on top
#    ax.xaxis.set_label_position('top')
    ax.set_xlabel(heatmap_df.columns.name, fontsize = 13)
    ax.set_ylabel(heatmap_df.index.name, fontsize = 13)
    plt.axvline(1, color = 'k')
    plt.axhline(heatmap_df.shape[0]-1, color = 'k')
    plt.title('Flu-%s Difference: %s vs %s \n 2018-2019 Season (%s), Target = %s\n'%(scoretype, names[0], names[1], datatype, target), 
              fontsize = 15, fontweight="bold")
    plt.annotate('%s wins'%names[1], xy=(0, 0.18),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='left',fontweight="bold" )
    plt.annotate('', xy = (0, 0.15), xytext = (dfcenter, 0.15, ), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('%s wins'%names[0], xy=(0.99, 0.18),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='right', fontweight="bold")
    plt.annotate('', xy = (1, 0.15), xytext = (dfcenter, 0.15, ), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    return ax

def AvgDiffLineplot(df1, df2, target, names, datatype, scoretype = 'logscore'):  
    '''
    Create average line plots with confidence interval
    Either for one target or for all of choices
    '''
    
    # If there is only one target
    if type(target) == str:      
        df1 = df1.loc[df1['target'] == target]
        df2 = df2.loc[df2['target'] == target]
        title = 'Flu-%s Comparision between %s and %s \n 2018-2019 Season (%s), Target = %s\n'%(scoretype, names[0], names[1], datatype, target)
    else:
        title = 'Flu-%s Comparision between %s and %s \n 2018-2019 Season (%s) over all targets\n'%(scoretype, names[0], names[1], datatype)
    
    if datatype == 'Region':
        c = 'Regions'
        df1 = df1.loc[df1['location'] != 'US National']
        df2 = df2.loc[df2['location'] != 'US National']
    elif datatype == 'Hospitalization':
        c = 'Age groups'
        df1 = df1.loc[df1['location'] != 'Overall']
        df2 = df2.loc[df2['location'] != 'Overall']
    else:
        c = 'States'
    
    
    # overall or national value could not be included in the average
    avgdiff1 = df1.groupby(by=['competition_week']).mean()[scoretype]
    avgdiff2 = df2.groupby(by=['competition_week']).mean()[scoretype]
    
    stddiff1 = df1.groupby(by=['competition_week']).std()[scoretype]
    stddiff2 = df2.groupby(by=['competition_week']).std()[scoretype]
    
    
    sns.set(style="whitegrid") 
    plt.figure(figsize = (10, 6))
    ax = sns.lineplot(x = avgdiff1.index, y = avgdiff1.values, label = 'Average %s of %s'%(scoretype, names[0]), color = 'r')
    ax = sns.lineplot(x = avgdiff2.index, y = avgdiff2.values, label = 'Average %s of %s'%(scoretype, names[1]), color = 'b')
#    ax.set(xticks=avgdiff1.index)
    ax.fill_between(avgdiff1.index, avgdiff1.values - stddiff1.values, avgdiff1.values + stddiff1.values, alpha = 0.25, color = 'r', label = 'CI of %s'%names[0])
    ax.fill_between(avgdiff2.index, avgdiff2.values - stddiff2.values, avgdiff2.values + stddiff2.values, alpha = 0.25, color = 'b', label = 'CI of %s'%names[1])
    plt.xticks(list(avgdiff2.index),fontsize = 11)
    ax.set_xlabel('Competition Weeks', fontsize = 15)
    ax.set_ylabel('Average %s over All %s'%(scoretype,c), fontsize = 15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=1)
    plt.title(title, fontsize = 15, fontweight="bold")
    return ax


def LocationBoxenplot_Target(df, target, names, datatype, scoretype = 'logscore'):
    """
    - create a table to rank the difference in terms of 
    different targets for two groups
    - sorted by median values 
    - cross different locations/age groups
    """
    if datatype == 'State' or datatype == 'Hospitalization':
#        column_order = np.sort(np.unique(df1['location']))
        if datatype == 'State':
            ylabel = 'States'
            figs = (10,20)
        else:
            figs = (10,10)  
            ylabel = 'Hospitalization'
    elif datatype == 'Region':
#        column_order = ['HHS Region %d'%i for i in range(1,11)]
        figs = (10,10)
        ylabel = 'Regions'
    
    sns.set(style="whitegrid")    
    plt.figure(figsize= figs)
#    df = df.loc[df['location']!='US National'] 
    order = df.groupby(by=["location"])["difference"].median().sort_values()[::-1].index
    dfcenter = (0 - df['difference'].min())/(df['difference'].max()-df['difference'].min())
    ax = sns.boxplot(data=df, y='location',x='difference',
                       orient="h",  order = order, palette = None)
    ax.set_xlabel(scoretype.capitalize() + ' Difference over All competition Weeks', fontsize = 15)
    ax.set_ylabel(ylabel, fontsize = 15)
    plt.title('Flu-%s Boxplot: %s vs %s \n 2018-2019 Season (%s), Target = %s\n'%(scoretype, names[0], names[1], datatype, target), 
              fontsize = 15, fontweight="bold")
    
    #Add arrows to show the comparison information
    plt.annotate('%s wins'%names[1], xy=(0, 0.04),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='left',fontweight="bold" )
   
    plt.annotate('', xy = (0, 0.02), xytext = (dfcenter,0.02), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('%s wins'%names[0], xy=(0.99, 0.04),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='right', fontweight="bold")


    plt.annotate('', xy = (1, 0.02), xytext = (dfcenter, 0.02 ), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    return ax
    

def SummaryBarPlot_Target(df, target, xlabel, names, datatype, scoretype = 'logscore', rate = 0.05):
    if xlabel == 'competition_week':
        figuresize = (12, 6)
        ylabel = 'Competition Weeks'
        if datatype == 'Region':
            c = 'Regions'            
            df = df.loc[df['location'] != 'US National']
        elif datatype == 'State':
            c = 'States'            
        else:
            c = 'Age Groups'
            df = df.loc[df['location'] != 'Overall']            
    else:
        c = 'Competition Weeks'
        if datatype == 'Region':
            ylabel = 'Regions'
            figuresize = (12, 6)
        elif datatype == 'State':
            ylabel = 'States'
            figuresize = (12, 12)
        else:
            ylabel = 'Age Groups'
            figuresize = (12, 6)
    
    index = np.unique(df[xlabel])
    
#    if 'US National' in df['location']:
#        df = df.loc[df['location']!='US National']
    compweekcounts = df.groupby(xlabel).count()['difference'].reindex(index, fill_value = 0)
    winratecounts = df[df['difference'] >= 0].groupby(xlabel).count()['difference'].reindex(index, fill_value = 0)
    winrate = winratecounts/compweekcounts
    winrateceil = np.ceil(winrate.max() * 10)/10
#    elta = rate * df['difference'].max()
    elta = rate
    similarcounts = df[np.abs(df['difference']) <= elta].groupby(xlabel).count()['difference'].reindex(index, fill_value = 0)
    overlapindex = np.array(df['difference']>= 0) & np.array(df['difference'] <= elta)
    overlapcounts = df[overlapindex].groupby(xlabel).count()['difference'].reindex(index, fill_value = 0)
    winratefloor = np.floor(winrate.min() * 10)/10-1
    xmin = min(-0.5, winratefloor)
    xmax = max(0.5, winrateceil)
    dfcenter = -xmin/(xmax - xmin)
    xticks = np.arange(xmin, xmax, 0.1).round(decimals=1)
    plotdf = pd.DataFrame({c: index,
                           'Win-Rate': winrate,
                           'Lose-Rate': winrate-1, 
                           'Overlap-win-Rate': overlapcounts/compweekcounts,
                           'Overlap-lose-Rate': (overlapcounts - similarcounts)/compweekcounts})
    plotdf = plotdf.sort_values('Win-Rate', ascending = False)
    
    
    sns.set(style="whitegrid")
    plt.figure(figsize=figuresize)
    
    ax = sns.set_color_codes("pastel")
    ax = sns.barplot(x="Lose-Rate", y=c, data=plotdf,
                label="$0 < Fluscore(%s)-Fluscore(%s) < %.2f$"%(names[1], names[0], rate), color="b", orient = 'h',
                order = plotdf[c])
    
    ax = sns.set_color_codes("muted")
    ax = sns.barplot(x="Overlap-lose-Rate", y=c, data=plotdf,
                label="$Fluscore(%s) \geq Fluscore(%s)$"%(names[1], names[0]), color="b",
                orient = 'h', order = plotdf[c])

    ax = sns.set_color_codes("pastel")
    ax = sns.barplot(x="Win-Rate", y=c, data=plotdf,
                label="$Fluscore(%s) \geq Fluscore(%s)$"%(names[0], names[1]), color="r",
                orient = 'h', order = plotdf[c])
    

    ax = sns.set_color_codes("muted")
    ax = sns.barplot(x="Overlap-win-Rate", y=c, data=plotdf,
                label='$0 < Fluscore(%s)-Fluscore(%s) < %.2f$'%(names[0], names[1], rate), color="r",
                orient = 'h', order = plotdf[c])
       
    # Add a legend and informative axis label
#    ax.legend(ncol =1, loc="upper right", frameon=True, fontsize = 15)
    ax.set_ylabel(xlabel.capitalize(), fontsize = 15)
    ax.axvline(0.5, color = 'r', linestyle = '--')
    ax.axvline(-0.5, color = 'b', linestyle = '--')
#    ax.axhline(-0.5, color = 'r', linestyle = '--')
    ax.set_xlabel('Summary Rates over All %s'%c, fontsize = 15)
    ax.set_ylabel(ylabel, fontsize = 15)
    plt.yticks(rotation = 0 , fontsize = 12)
    plt.xticks(xticks, abs(xticks), fontsize = 12)
    sns.despine(left=True, bottom=True)
    
#    dfcenter = -ax.get_xticks()[0]/(ax.get_xticks()[-1]-ax.get_xticks()[0])
    plt.title('Flu-%s Summary: %s vs %s \n 2018-2019 Season (%s), Target = %s\n'%(scoretype, names[0], names[1], datatype, target), 
              fontsize = 15, fontweight="bold")
    
    #Add arrows to show the comparison information
    plt.annotate('%s wins'%names[1], xy=(0, 0.2),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='left',fontweight="bold" )
   
    plt.annotate('', xy = (0, 0.18), xytext = (dfcenter,0.18), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('%s wins'%names[0], xy=(0.99, 0.2),  xycoords=('axes fraction', 'figure fraction'),
                 textcoords='offset points', size=12, ha='right', fontweight="bold")


    plt.annotate('', xy = (1, 0.18), xytext = (dfcenter, 0.18), xycoords=('axes fraction', 'figure fraction'),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=1)

    return ax


def SummaryDotPlot(df1, df2, xlabel, targets, names, datatype, scoretype, summarytype = 'win_rate', rate = 0.05):
    if xlabel == 'competition_week':
        if datatype == 'State':
            c = 'States'
        elif datatype == 'Region':
            c = 'Regions'
        else:
            c = 'Age Groups'
    else:
        c = 'Competition Weeks'
            
            
    if summarytype == 'win_rate':
        
        plottitle = 'Flu-%s Summary: %s vs %s \n 2018-2019 Season (%s), win-rate = P((Difference > 0) over all %s'%(scoretype, names[0], names[1], datatype, c)
    else:
        plottitle = 'Flu-%s Summary: %s vs %s \n 2018-2019 Season (%s), similar-rate = P((|Difference| <= %.2f) over all %s'%(scoretype, names[0], names[1], datatype, rate, c)
    
    sns.set(style="whitegrid")
    if type(targets) == str:
        targets = [targets]
    
    plotdf = pd.DataFrame({xlabel: np.unique(df1[xlabel])})
    for i in range(len(targets)):
        df = create_difference_df(df1, df2, targets[i], 'logscore')
        
#        if 'US National' in df['location']:
#            df = df.loc[df['location']!='US National']
        if summarytype == 'win_rate':
            figdf = df[df['difference'] >= 0].groupby(xlabel).count()['difference'] / df.groupby(xlabel).count()['difference']
            figdf = figdf.fillna(0)
        elif summarytype == 'similar_rate':
            elta = rate 
            figdf = df[np.abs(df['difference']) <= elta].groupby(xlabel).count()['difference'] / df.groupby(xlabel).count()['difference']
            figdf = figdf.fillna(0)
        tempt = []
        for k in range(plotdf.shape[0]):
            if plotdf[xlabel][k] in figdf.index:
                tempt.append(figdf.values[k])
            else:
                tempt.append(np.nan)
        plotdf[targets[i]] = tempt
    
    plotdf['Average'] = plotdf.iloc[:, 1:].mean(axis = 1)
    plotdf['Std'] = plotdf.iloc[:, 1:].std(axis = 1)
    
    if xlabel == 'location':
        plotdf = plotdf.sort_values('Average', ascending = False)
    plotdf.loc['Average'] = ['Avg'] + list(plotdf.iloc[:, 1:-1].mean(axis = 0)) + [np.nan]
    plotdf.loc['Std'] = ['Std'] + list(plotdf.iloc[:, 1:-1].std(axis = 0)) + [np.nan]
    
    
    sns.set(style="whitegrid")
    plt.figure(figsize = (8, 8))
    g = sns.PairGrid(plotdf,
                 x_vars=plotdf.columns[1:], y_vars=plotdf.columns[0],
                 height=10, aspect=.25)
    g.map(sns.stripplot, size=10, orient="h",
           palette="ch:s=1,r=-.1,h=1_r", linewidth=1, edgecolor="w")
    
    ceil = plotdf.iloc[:, 1:].max().values.max() + 0.1
    floor = plotdf.iloc[:, 1:].min().values.min() - 0.1
    g.set(xlim=(floor, ceil), xlabel=summarytype, ylabel="")
    titles = np.array(plotdf.columns[1:])
    for ax, title in zip(g.axes.flat, titles):
        ax.set(title=title)
    
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
#    g = g.add_legend(title = plottitle, fontsize = 15, position = 'top')
    g.fig.suptitle(plottitle, fontsize = 15, fontweight="bold")
    g.fig.subplots_adjust(top=.9)

    return g





