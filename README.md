# markov-decision-processes

This is the code for assignment 4 of CS 7641.

To run the code, first install burlap from https://github.com/jmacglashan/burlap

Replace the ValueIteration.java and PolicyIteration.java in the burlap code
The directories are:
burlap.behavior.singleagent.planning.stochastic.valueiteration
burlap.behavior.singleagent.planning.stochastic.policyiteration

Then, in the root directory, compile using
$mvn compile

The code to run the grid world experiments is BasicBehavior.java
to run it, just run in the root directory (from terminal). Make sure you compile it first...
$mvn exec:java -Dexec.mainClass="BasicBehavior"

For the block dude:
$mvn exec:java -Dexec.mainClass="RunBlockDude"

The code will output .txt files which you can then plot with plot_all.ipynb (it will save plots as images)
Make sure the file names are the same. You can change the names of the output .txt files in BaiscBehavior.java, RunBlockDude.java, as well as ValueIteration.java and PolicyIteration.java
It should be obvious where to do so




