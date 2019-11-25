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
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeModel;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.blockdude.state.BlockDudeAgent;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.SADomain;

import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.mdp.singleagent.oo.OOSADomain;


import java.io.*;
import java.awt.*;
import java.util.List;



public class RunBlockDude {
	
	BlockDude block;
	TerminalFunction tf;
	OOSADomain domain;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;


	public RunBlockDude(){
		block = new BlockDude();
		domain = block.generateDomain();
		// initialState = new BlockDudeState();
		initialState = BlockDudeLevelConstructor.getLevel3(domain);
		hashingFactory = new SimpleHashableStateFactory();
		tf = new BlockDudeTF();
		block.setTf(tf);
		goalCondition = new TFGoalCondition(tf);

		env = new SimulatedEnvironment(domain, initialState);
	}

	// public void visualize(String outputPath){
	// 	Visualizer v = BlockDudeVisualizer.getVisualizer(25, 25);
	// 	new EpisodeSequenceVisualizer(v, domain, outputPath);
	// }

	public void valueIterationExample(String outputPath) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter("level3blockDudeRewardsVI.txt");
		int iters = 200;
		double cumulativeTime = 0.;
		long startTime = 0;
		Planner planner = null;
		Policy p = null;
		Episode episode = null;
		double deltaTime = 0.;

		for(int i = 1; i <= iters; i+=5) {
			startTime = System.nanoTime();
			planner = new ValueIteration(domain, 0.99, hashingFactory, -1, i);
			p = planner.planFromState(initialState);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
			episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 1000);
			writer.println(episode.numTimeSteps());
			writer.println(deltaTime);
		}
		writer.close();
		
	}

	public void policyIterationExample(String outputPath) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter("level3blockDudeRewardsPI_1.txt");
		int iters = 100;
		double cumulativeTime = 0.;
		long startTime = 0;
		Planner planner = null;
		Policy p = null;
		Episode episode = null;
		double deltaTime = 0.;

		// for(int i = 1; i <= iters; i+=5) {
		// 	startTime = System.nanoTime();
		// 	planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 1, i);
		// 	p = planner.planFromState(initialState);
		// 	deltaTime = System.nanoTime() - startTime;
		// 	cumulativeTime += deltaTime;
		// 	//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
		// 	episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 2000);
		// 	writer.println(episode.numTimeSteps());
		// 	writer.println(deltaTime);
		// }
		// writer.close();

		writer = new PrintWriter("level3blockDudeRewardsPI_20.txt");
		for(int i = 1; i <= iters; i+=10) {
			startTime = System.nanoTime();
			planner = new PolicyIteration(domain, 0.99, hashingFactory, -1, 20, i);
			p = planner.planFromState(initialState);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			//PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");
			episode = PolicyUtils.rollout(p, initialState, domain.getModel(), 2000);
			writer.println(episode.numTimeSteps());
			writer.println(deltaTime);
		}
		writer.close();
	}

	public void qLearningExample(String outputPath) throws FileNotFoundException {
		PrintWriter writer = new PrintWriter("level3blockDudeQlearn_lr1.txt");
		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);


		double cumulativeTime = 0.;
		double deltaTime = 0.;
		for(int i = 0; i < 10000; i++){
			System.out.println(i);
			long startTime = System.nanoTime();
			Episode e = agent.runLearningEpisode(env, 100000);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(cumulativeTime);

			e.write(outputPath + "ql_" + i);
			writer.println(e.maxTimeStep());
			System.out.println(e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

		writer.close();


		writer = new PrintWriter("level3blockDudeQlearn_lr05.txt");
		agent = new QLearning(domain, 0.99, hashingFactory, 0., 0.5);


		cumulativeTime = 0.;
		deltaTime = 0.;
		for(int i = 0; i < 10000; i++){
			System.out.println(i);
			long startTime = System.nanoTime();
			Episode e = agent.runLearningEpisode(env, 100000);
			deltaTime = System.nanoTime() - startTime;
			cumulativeTime += deltaTime;
			writer.println(cumulativeTime);

			e.write(outputPath + "ql_" + i);
			writer.println(e.maxTimeStep());
			System.out.println(e.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

		writer.close();
	}


	public static void main(String[] args) {
		String outputPath = "level3blockdudeoutput/";
		RunBlockDude example = new RunBlockDude();
		// try {
		// 	example.valueIterationExample(outputPath);
		// } catch (FileNotFoundException ex) {
		// 	System.out.println();
		// }
		try {
			example.policyIterationExample(outputPath);
		} catch (FileNotFoundException ex) {
			System.out.println();
		}
		// try {
		// 	example.qLearningExample(outputPath);
		// } catch (FileNotFoundException ex) {
		// 	System.out.println();
		// }

		//example.visualize(outputPath);
	}


}