# FluscoreErrorAnalysis


Automatically do error analysis using flu-score data of two groups 

$python main.py --group1 example/Delphi-Epicast2.csv --group2 example/Delphi-Stat.csv --name1 Delphi-Epicast --name2 Delphi-Stat

The minimum input files required to run the program are: 

[--group1] A csv file of flu-score information of group 1 with separator ','. 

[--group2] A csv file of flu-score information of group 2 with separator ','. 
	 

Optional Arguments:

[--name1]  The name of group 1. 

[--name2]  The name of group 2. 

[--targets] A target or multiple targets(use ' \ ' to connect them) of interest.Choose from
            (‘Season_peak_week\Season_onset\Season_peak_percentage\\1_wk_ahead\\2_wk_ahead\\3_wk_ahead\\4_wk_ahead’)

[—-scoretype] Choose from (‘logscore’, ‘score’) to decide which kind of value is used for comparison.

[—-datatype] Choose from (‘Region’, ‘State’, ‘Hospitalization’) to inform which type of fluscore data is used.

[--outputpath]. The defualt is 'name1_name2_datatype'.

[—-summarytype] Choose from (‘win_rate’, ‘similar_rate’) to decide which kind of summarty type to use.
		
		win_rate: The proportion that Group 1 performance better than Group 2 in terms of
		          a specific location or competition_week.
					
		similar_rate: The proportion that the absolute flu-score difference between Group 1 
		              and Group 2 is smaller than a threshold in terms of a specific location or competition_week.

[—-similarrate] The threshold used to measure whether the performance of Group 1 and Group 2 is similar enough.

[—-xlabel] Choose from (‘location’, ‘competition_week’) to decide what the x-axis is when considering summary rate(win_rate, similar_rate).

[—-competition_week] A specific competition_week or a specific range of competition_week of interest.

[—-location] A specific location or a list of locations of interest.

[—-annot] Choose from (‘True’, ‘False’) to decide whether to add annotations to the heatmap cells.




Visualization Methods:	

— Heatmap for the performance difference in terms of a specific target. (competition_week, location, difference)
- Boxplot for the performance difference in terms of a specific target. (location, distribution of difference over 
  all competition weeks)  
- Twoside barplot for Summary Rate in terms of a specific target sorted by the win-rate of group 1(summary rate 
  over all competition weeks)
- Dotplot for Summary Rate for all of the targets (sorted by average of difference for different targets if xlabel
  is chosen to be locations)


Limitations:

- This version cannot deal with more than two groups.
- It takes several seconds to run the system.
- The system allows different formats for targets, but it should at least includes strings such as “onset”, ‘peak week’, ‘pkwk’, ‘percentage’, ‘pkpr’
  ‘1’, ‘2’, ‘3’, ‘4’. You can change it in src/__init__.py/check_targets.
- The system could automatically check the data type if there is no datatype fed in. But it requires ‘us’/‘us national’ for “Region” data, ‘overall’ for 
  ‘Hospitalization’ data, otherwise, datatype is required for the system.
- Only these types of visualizations. New ideas are welcomed!

		
	 
      

		
