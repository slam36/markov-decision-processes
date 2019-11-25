import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;


import java.io.*;
import java.awt.*;
import java.util.List;

public class BasicBehavior {

	GridWorldDomain gwdg;
	OOSADomain domain;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;


	public BasicBehavior(){
		
		int [][] map = new int[][]{
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,1,0,0,0,0,0,1,0,0},
			{0,0,1,0,0,1,0,0,1,0,0},
			{0,0,0,0,0,1,0,0,1,1,1},
			{0,0,0,0,0,0,0,0,0,1,0},
			{1,0,0,1,1,0,1,1,0,1,1},
			{0,0,0,0,1,0,0,1,0,0,0},
			{0,0,0,0,1,0,0,0,1,0,0},
			{0,1,0,0,0,0,0,0,1,0,0},
			{0,0,1,0,1,0,0,0,1,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
		};
		gwdg = new GridWorldDomain(11, 11);
		//gwdg = new GridWorldDomain(map);
		gwdg.setMapToFourRooms();
		tf = new GridWorldTerminalFunction(10, 10);
		gwdg.setTf(tf);
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();

		initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);


		//VisualActionObserver observer = new VisualActionObserver(domain, 
		//	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
		//observer.initGUI();
		//env.addObservers(observer);
	}


	public void visualize(String outputpath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}

	public void BFSExample(String outputPath){

		DeterministicPlanner planner = new BFS(domain, goalCondition, hashingFactory);
		Policy p = planner.planFromState(initialState);
		PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "bfs");

	}

	public void DFSExample(String outputPath){

		DeterministicPlanner planner = new DFS(domain, goalCondition, hashingFactory);
		Policy p = planner.planFromState(initialState);
		PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "dfs");

	}

	public void AStarExample(String outputPath){

		Heuristic mdistHeuristic = new Heuristic() {

			public double h(State s) {
				GridAgent a = ((GridWorldState)s).agent;
				double mdist = Math.abs(a.x-10) + Math.abs(a.y-10);

				return -mdist;
			}
		};

		DeterministicPlanner planner = new AStar(domain, goalCondition, 
			hashingFactory, mdistHeuristic);
		Policy p = planner.planFromState(initialState);

		PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "astar");

	}

	public void valueIterationExample(String outputPath) throws FileNotFoundException{

		PrintWriter writer = new PrintWriter("gridWorldRewardsVI.txt");
		int iters = 30;
		double cumulativeTime = 0.;
		long startTime = System.nanoTime();
		double deltaTime = 0.;
		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, -1, 1);
		//ValueIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxIterations)
		Policy p = planner.planFromState(initialState);

		// PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
		deltaTime = System.nanoTime() - startTime;
		cumulativeTime += deltaTime;
		Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());

		writer.println(episode.numTimeSteps());
		writer.println(findSumWithoutUsingStream(episode.rewardSequence));
		writer.println(deltaTime);
		for(int i = 2; i < iters; i++) {
			startTime = System.nanoTime();
			planner = new ValueIteration(domain, 0.99, hashingFactory, -1, i);
			//ValueIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxIterations)
			p = planner.planFromState(initialState);

			// PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
			episode = PolicyUtils.rollout(p, initialState, domain.getModel());
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(episode.numTimeSteps());
			writer.println(findSumWithoutUsingStream(episode.rewardSequence));
			writer.println(deltaTime);
		}
		writer.close();
		
		simpleValueFunctionVis((ValueFunction)planner, p);
		//manualValueFunctionVis((ValueFunction)planner, p);



	}

	//https://www.baeldung.com/java-array-sum-average
	public static double findSumWithoutUsingStream(List<Double> array) {
	    double sum = 0;
	    for (double value : array) {
	        sum += value;
	    }
	    return sum;
	}

	public void policyIterationExample(String outputPath) throws FileNotFoundException{

		double cumulativeTime = 0.;
		long startTime = System.nanoTime();
		double deltaTime = 0.;
		PrintWriter writer = new PrintWriter("gridWorldRewardsPI.txt");
		int iters = 30;
		Planner planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 1, 1);
		//PolicyIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
		Policy p = planner.planFromState(initialState);
		//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");
		Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 5000);
		deltaTime = System.nanoTime() - startTime;
		cumulativeTime += deltaTime;
		writer.println(episode.numTimeSteps());
		writer.println(findSumWithoutUsingStream(episode.rewardSequence));
		writer.println(deltaTime);

		
		for(int i = 2; i < iters; i++) {
			startTime = System.nanoTime();
			planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 1, i);
			//PolicyIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
			p = planner.planFromState(initialState);

			//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");
			episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 5000);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(episode.numTimeSteps());
			writer.println(findSumWithoutUsingStream(episode.rewardSequence));
			writer.println(deltaTime);
			// writer.println(episode.rewardSequence);
			

		}
		writer.close();
		simpleValueFunctionVis((ValueFunction)planner, p);


		cumulativeTime = 0.;
		startTime = System.nanoTime();
		deltaTime = 0.;
		writer = new PrintWriter("gridWorldRewardsPI_20visteps.txt");
		iters = 30;
		planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 20, 1);
		//PolicyIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
		p = planner.planFromState(initialState);
		//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");
		episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 5000);
		deltaTime = System.nanoTime() - startTime;
		cumulativeTime += deltaTime;
		writer.println(episode.numTimeSteps());
		writer.println(findSumWithoutUsingStream(episode.rewardSequence));
		writer.println(deltaTime);

		
		for(int i = 2; i < iters; i++) {
			startTime = System.nanoTime();
			planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 20, i);
			//PolicyIteration(SADomain domain, double gamma, HashableStateFactory hashingFactory, double maxDelta, int maxEvaluationIterations, int maxPolicyIterations)
			p = planner.planFromState(initialState);

			//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");
			episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 5000);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(episode.numTimeSteps());
			writer.println(findSumWithoutUsingStream(episode.rewardSequence));
			writer.println(deltaTime);
			// writer.println(episode.rewardSequence);
			

		}
		writer.close();
		simpleValueFunctionVis((ValueFunction)planner, p);
	}


	public void qLearningExample(String outputPath) throws FileNotFoundException{

		PrintWriter writer = new PrintWriter("gridWorldQlearn_lr1.txt");
		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);


		double cumulativeTime = 0.;
		double deltaTime = 0.;
		for(int i = 0; i < 100; i++){
			System.out.println(i);
			long startTime = System.nanoTime();
			Episode e = agent.runLearningEpisode(env);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(cumulativeTime);

			e.write(outputPath + "ql_" + i);
			writer.println(e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

		writer.close();


		writer = new PrintWriter("gridWorldQlearn_lr05.txt");
		agent = new QLearning(domain, 0.99, hashingFactory, 0., 0.5);


		cumulativeTime = 0.;
		deltaTime = 0.;
		for(int i = 0; i < 100; i++){
			long startTime = System.nanoTime();
			Episode e = agent.runLearningEpisode(env);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(cumulativeTime);

			e.write(outputPath + "ql_" + i);
			writer.println(e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

		writer.close();

	}


	public void sarsaLearningExample(String outputPath){

		LearningAgent agent = new SarsaLam(domain, 0.99, hashingFactory, 0., 0.5, 0.3);

		//run learning for 50 episodes
		for(int i = 0; i < 50; i++){
			Episode e = agent.runLearningEpisode(env);

			e.write(outputPath + "sarsa_" + i);
			System.out.println(i + ": " + e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(
			initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, 11, 11, valueFunction, p);
		gui.initGUI();

	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(
			initialState, domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, 
		//specifying which attributes correspond to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYKeys("agent:x", "agent:y", 
			new VariableDomain(0, 11), new VariableDomain(0, 11), 
			1, 1);

		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
			allStates, svp, valueFunction);

		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), 
			new VariableDomain(0, 11), 
			1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);

		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();



	}


	public void experimentAndPlotter(){

		//different reward function for more structured performance plots
		//((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 1.0, -1));

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "LR = 1";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0., 1.);
			}
		};

		LearningAgentFactory qLearningFactory2 = new LearningAgentFactory() {

			public String getAgentName() {
				return "LR = 0.5";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0., 0.5);
			}
		};


		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
			env, 10, 50, qLearningFactory, qLearningFactory2);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.STEPS_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}


	public static void main(String[] args) {

		BasicBehavior example = new BasicBehavior();
		String outputPath = "output/";

		//example.BFSExample(outputPath);
		//example.DFSExample(outputPath);
		//example.AStarExample(outputPath);
		// try {
		// 	example.valueIterationExample(outputPath);
		// } catch (FileNotFoundException ex) {
		// 	System.out.println();
		// }
		// try {
		// 	example.policyIterationExample(outputPath);
		// } catch (FileNotFoundException ex) {
		// 	System.out.println();
		// }
		try {
			example.qLearningExample(outputPath);
		} catch (FileNotFoundException ex) {
			System.out.println("by no means!");
		}
		//example.sarsaLearningExample(outputPath);

		//example.experimentAndPlotter();

		//example.visualize(outputPath);

	}

}