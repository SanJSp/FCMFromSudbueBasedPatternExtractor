# Tool chain for fCM model generation from event logs

The project and research is decribed in more detail in our [paper](paper.pdf)

This implementation is a semi-automated tool chain to create generate [fragment-based case models](https://www.researchgate.net/publication/307585126_A_Hybrid_Approach_for_Flexible_Case_Modeling_and_Execution) from your event logs. For this, we implement a parser from .xes event logs to the [subdue algorithm](https://github.com/holderlb/Subdue/tree/master/testing) input and then return subdues found patterns and the extracted virtual data objects.

The process works as follows:
![process](/process.png)



### Input data creation
To create your own input data, you need a .csv or .xes file of your event log. If it is a .csv, use [ProM](http://promtools.org/doku.php?id=prom611) (tested on version 6.11) to create an .xes file from it. 

Next, you need to create a .pnml file from your log. For this, use your .xes file and load it into ProM. Then use the inductive visual miner action ("Mine with inductive visual miner") by Leemans to create a petri net. 

To export it, click the view of the model, select export model in the bar to the right and then choose petri net. From the created files in your workspace, chose the "Petri net of XES Event Log" and export it to your disk.

### Executing [the pattern extractor](./main.py)

To run the main file, you need to provide the paths to the .xes and .pnml file of your log. Then you can execute the script.

### Outputs

In the console, you will receive updates on the stages of subdue and the discovered patterns. From there you can extract in the lines at the end of the output the found patterns, the node belongings (which node belongs to which pattern) and the virtual data objects.
This enables you to start with the manual steps of pattern post-processing, data object verification and fragment construction.

### Executing the [pattern verifier](./dependency_verifier.py)
Specify the paths in the main file to your .csv and pass the two activities you want to verify. Unfortunately we cannot provide a dataset to test this in detail.

### Contact

You are welcome to create an issue if questions arise.
