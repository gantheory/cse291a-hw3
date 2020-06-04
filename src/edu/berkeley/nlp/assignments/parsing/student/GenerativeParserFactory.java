package edu.berkeley.nlp.assignments.parsing.student;

import edu.berkeley.nlp.assignments.parsing.Grammar;
import edu.berkeley.nlp.assignments.parsing.SimpleLexicon;
import edu.berkeley.nlp.assignments.parsing.TreeAnnotations;
import edu.berkeley.nlp.assignments.parsing.UnaryClosure;
import edu.berkeley.nlp.util.Counter;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.math.SloppyMath;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.assignments.parsing.Parser;
import edu.berkeley.nlp.assignments.parsing.ParserFactory;
import edu.berkeley.nlp.assignments.parsing.UnaryRule;
import edu.berkeley.nlp.assignments.parsing.BinaryRule;
import edu.berkeley.nlp.ling.Tree;
import edu.berkeley.nlp.ling.Trees.PennTreeRenderer;


class GenerativeParser implements Parser
{
	final double MIN_LOG_PROB = Double.NEGATIVE_INFINITY;
	final int SENTENCE_LENGTH = 40;
	final Tree<String> NULL_TREE = new Tree<>("ROOT", Collections.singletonList(new Tree<>("JUNK")));
	final boolean useSloppyMath = false;
	final boolean unFoldClosure = true;
	final int v = 2;
	final int h = 2;
	final double COUNT_THRESHOLD = 10.0;

	Grammar grammar;
	SimpleLexicon lexicon;
	UnaryClosure unaryClosure;
	Indexer<String> indexer;
	int rootID;
	int numOfStates;

	double[][] tagScore;
	double[][][] unaryDP;
	int[][][] unaryDPChild;
	double[][][] binaryDP;
	int[][][] binaryDPLeftChild;
	int[][][] binaryDPRightChild;
	int[][][] binaryDPSplit;

	Counter<String> stateCounter = new Counter<>();

	public Tree<String> getBestParse(List<String> sentence) {
		for (int state = 0; state < numOfStates; ++state) {
			for (int i = 0; i < sentence.size(); ++i) {
			  tagScore[state][i] = MIN_LOG_PROB;
		  	for (int j = i; j < sentence.size(); ++j) {
					unaryDP[state][i][j] = MIN_LOG_PROB;
					unaryDPChild[state][i][j] = -1;
					binaryDP[state][i][j] = MIN_LOG_PROB;
					binaryDPLeftChild[state][i][j] = -1;
					binaryDPRightChild[state][i][j] = -1;
					binaryDPSplit[state][i][j] = -1;
				}
			}
		}

		for (int state = 0; state < numOfStates; ++state) {
			String tag = indexer.get(state);
			for (int i = 0; i < sentence.size(); ++i) {
			  // score could be NaN
				double score = lexicon.scoreTagging(sentence.get(i), tag);
				if (!isValidScore(score)) continue;
				tagScore[state][i] = score;
			}
		}

		for (int state = 0; state < numOfStates; ++state) {
		  for (int i = 0; i < sentence.size(); ++i) {
				for (UnaryRule rule : unaryClosure.getClosedUnaryRulesByParent(state)) {
					double logRuleScore = rule.getScore();
					if (!isValidScore(logRuleScore)) continue;
					double logTerminalScore = tagScore[rule.getChild()][i];
					if (!isValidScore(logTerminalScore)) continue;
					double logScore = myLogAdd(logRuleScore, logTerminalScore);
					if ((unaryDP[state][i][i] == MIN_LOG_PROB) || (logScore > unaryDP[state][i][i])) {
						unaryDP[state][i][i] = logScore;
						unaryDPChild[state][i][i] = rule.getChild();
					}
				}
			}
		}

		for (int length = 2; length <= sentence.size(); ++length) {
			for (int state = 0; state < numOfStates; ++state) {
				for (int i = 0; i < sentence.size() - length + 1; ++i) {
					int j = i + length - 1;
					for (BinaryRule rule : grammar.getBinaryRulesByParent(state)) {
						double logRuleScore = rule.getScore();
						if (!isValidScore(logRuleScore)) continue;
						for (int k = i; k < j; ++k) {
							double logLeftChildScore = unaryDP[rule.getLeftChild()][i][k];
							if (!isValidScore(logLeftChildScore)) continue;
							double logRightChildScore = unaryDP[rule.getRightChild()][k + 1][j];
							if (!isValidScore(logRightChildScore)) continue;
							double logScore = myLogAdd(myLogAdd(logRuleScore, logLeftChildScore), logRightChildScore);
							if ((binaryDP[state][i][j] == MIN_LOG_PROB) || (logScore > binaryDP[state][i][j])) {
								binaryDP[state][i][j] = logScore;
								binaryDPLeftChild[state][i][j] = rule.getLeftChild();
								binaryDPRightChild[state][i][j] = rule.getRightChild();
								binaryDPSplit[state][i][j] = k;
							}
						}
					}
				}
			}
			for (int state = 0; state < numOfStates; ++state) {
				for (int i = 0; i < sentence.size() - length + 1; ++i) {
					int j = i + length - 1;
					for (UnaryRule rule : unaryClosure.getClosedUnaryRulesByParent(state)) {
			  		double logRuleScore = rule.getScore();
			  		if (!isValidScore(logRuleScore)) continue;
						double logChildScore = binaryDP[rule.getChild()][i][j];
						if (!isValidScore(logChildScore)) continue;
			  		double logScore = myLogAdd(logRuleScore, logChildScore);
						if ((unaryDP[state][i][j] == MIN_LOG_PROB) || (logScore > unaryDP[state][i][j])) {
			  			unaryDP[state][i][j] = logScore;
			  			unaryDPChild[state][i][j] = rule.getChild();
						}
					}
				}
			}
		}

		if (unaryDP[rootID][0][sentence.size() - 1] == MIN_LOG_PROB) return NULL_TREE;
		Tree<String> parseTree = buildUnaryTree(rootID, 0, sentence.size() - 1, sentence);
		return TreeAnnotations.unAnnotateTree(parseTree);
	}

	private Tree<String> buildUnaryTree(int parent, int l, int r, List<String> sentence) {
		if (l == r) {
			int preTerminal = unaryDPChild[parent][l][r];
			Tree<String> treeWithTerminal = new Tree<>(indexer.get(preTerminal), Collections.singletonList(new Tree<>(sentence.get(l))));

			if (parent == preTerminal) return treeWithTerminal;

			if (unFoldClosure) {
				List<Tree<String>> trees = new ArrayList<>();
				UnaryRule rule = new UnaryRule(parent, preTerminal);
				List<Integer> closurePath = unaryClosure.getPath(rule);
				for (int i = 0; i < closurePath.size(); ++i) {
					trees.add(new Tree<>(indexer.get(closurePath.get(i))));
				}
				trees.add(new Tree<>(sentence.get(l)));
				for (int i = 0; i < trees.size() - 1; ++i) {
					trees.get(i).setChildren(Collections.singletonList(trees.get(i + 1)));
				}
				return trees.get(0);
			} else {
				return new Tree<>(indexer.get(parent), Collections.singletonList(treeWithTerminal));
			}
		}

		int child = unaryDPChild[parent][l][r];
		if (parent == child) return buildBinaryTree(child, l, r, sentence);

		if (unFoldClosure) {
		  List<Tree<String>> trees = new ArrayList<>();
			UnaryRule rule = new UnaryRule(parent, child);
			List<Integer> closurePath = unaryClosure.getPath(rule);
			for (int i = 0; i < closurePath.size() - 1; ++i) {
				trees.add(new Tree<>(indexer.get(closurePath.get(i))));
			}
			trees.add(buildBinaryTree(closurePath.get(closurePath.size() - 1), l, r, sentence));
			for (int i = 0; i < closurePath.size() - 1; ++i) {
				trees.get(i).setChildren(Collections.singletonList(trees.get(i + 1)));
			}
			return trees.get(0);
		} else {
			Tree<String> childTree = buildBinaryTree(child, l, r, sentence);
			Tree<String> node = new Tree<>(indexer.get(parent), Collections.singletonList(childTree));
			return node;
		}
	}

	private Tree<String> buildBinaryTree(int parent, int l, int r, List<String> sentence) {
		if (l == r) return buildUnaryTree(parent, l, r, sentence);
		Tree<String> node = new Tree<>(indexer.get(parent));
		int mid = binaryDPSplit[parent][l][r];
		int leftChild = binaryDPLeftChild[parent][l][r];
		int rightChild = binaryDPRightChild[parent][l][r];
	  Tree<String> leftTree = buildUnaryTree(leftChild, l, mid, sentence);
	  Tree<String> rightTree = buildUnaryTree(rightChild, mid + 1, r, sentence);
	  node.setChildren(new ArrayList<>(List.of(leftTree, rightTree)));
		return node;
	}

	public GenerativeParser(List<Tree<String>> trainTrees) {
		System.out.print("Annotating / binarizing training trees ... ");
		List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
		System.out.println(PennTreeRenderer.render(annotatedTrainTrees.get(0)));

		System.out.println("done.");
		System.out.print("Building grammar ... ");
		grammar = Grammar.generativeGrammarFromTrees(annotatedTrainTrees);
		System.out.println("done. (" + grammar.getLabelIndexer().size() + " states)");

		lexicon = new SimpleLexicon(annotatedTrainTrees);

		unaryClosure = new UnaryClosure(grammar.getLabelIndexer(), grammar.getUnaryRules());

		indexer = grammar.getLabelIndexer();

		numOfStates = indexer.size();
		rootID = indexer.indexOf("ROOT");

		tagScore = new double[numOfStates][SENTENCE_LENGTH];
		unaryDP = new double[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
		unaryDPChild = new int[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
		binaryDP = new double[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
		binaryDPLeftChild = new int[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
		binaryDPRightChild = new int[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
		binaryDPSplit = new int[numOfStates][SENTENCE_LENGTH][SENTENCE_LENGTH];
	}

	private List<Tree<String>> annotateTrees(List<Tree<String>> trees) {
		List<Tree<String>> annotatedTrees = new ArrayList<Tree<String>>();
		for (Tree<String> tree : trees) {
			Tree<String> binarizedTree = binarization(tree);
			annotatedTrees.add(binarizedTree);
			countState(binarizedTree);
		}
		for (Tree<String> tree : annotatedTrees) {
			collapseHorizontal(tree);
		}
		return annotatedTrees;
	}

	private Tree<String> binarization(Tree<String> tree) {
		List<String> path = new ArrayList<>();
		Tree<String> verticalizedTree = verticalMarkovization(tree, path, tree);
		Tree<String> horizontalizedTree = horizontalMarkovication(verticalizedTree, tree);
		return horizontalizedTree;
	}

	private Tree<String> verticalMarkovization(Tree<String> tree, List<String> path, Tree<String> originalTree) {
		if (tree.isLeaf()) return tree;
		path.add(tree.getLabel());
		List<Tree<String>> newSubtrees = new ArrayList<>();
		for (Tree<String> subtree : tree.getChildren()) {
		  newSubtrees.add(verticalMarkovization(subtree, path, originalTree));
		}
		path.remove(path.size() - 1);
		StringBuilder nowLabel = new StringBuilder(tree.getLabel());
		for (int i = 0; i < v - 1; ++i) {
			if (path.size() - 1 - i < 0) break;
			String label = path.get(path.size() - 1 - i);
			nowLabel.append(String.format("^%s", label));
		}
		if (!tree.getLabel().equals("ROOT") && tree.getChildren().size() == 1) nowLabel.append("-U");
		if (originalTree.getChildren().size() == 1 && !originalTree.isPreTerminal()) {
		  String subLabel = originalTree.getChildren().get(0).getLabel();
			if (subLabel.equals("DT")) {
				newSubtrees.get(0).setLabel(subLabel + "^U");
			}
		  if (subLabel.equals("RB")) {
		    newSubtrees.get(0).setLabel(subLabel + "^U");
			}
		}
		return new Tree<>(nowLabel.toString(), newSubtrees);
	}

	private Tree<String> horizontalMarkovication(Tree<String> tree, Tree<String> originalTree) {
	  if (tree.isLeaf()) return tree;
		List<Tree<String>> oldSubtrees = tree.getChildren();
		List<Tree<String>> newSubtrees = new ArrayList<>();
		for (int i = 0; i < oldSubtrees.size(); ++i) {
			newSubtrees.add(horizontalMarkovication(oldSubtrees.get(i), originalTree.getChildren().get(i)));
		}
		List<Tree<String>> newRoots = new ArrayList<>();
		newRoots.add(new Tree<>(tree.getLabel()));
		for (int i = 0; i < newSubtrees.size() - 1; ++i) {
			StringBuilder nowLabel = new StringBuilder("@");
			nowLabel.append(tree.getLabel());
			nowLabel.append("[");
			if (i - (h - 1) > 0) nowLabel.append("...");
			for (int j = 0; j < h; ++j) {
				int idx = i - (h - 1) + j;
				if (idx < 0) continue;
        nowLabel.append("_");
				nowLabel.append(originalTree.getChildren().get(idx).getLabel());
			}
			nowLabel.append("]");
			newRoots.add(new Tree<>(nowLabel.toString()));
		}
		for (int i = 0; i < newRoots.size(); ++i) {
			if (i + 1 < newRoots.size()) {
				newRoots.get(i).setChildren(new ArrayList<>(List.of(newSubtrees.get(i), newRoots.get(i + 1))));
			} else {
				newRoots.get(i).setChildren(Collections.singletonList(newSubtrees.get(i)));
			}
		}
		return newRoots.get(0);
	}

	private void countState(Tree<String> tree) {
		if (tree.isLeaf()) return;
		List<Tree<String>> subtrees = tree.getChildren();
	  for (Tree<String> subtree : subtrees) countState(subtree);
		String key = tree.getLabel();
		if (!stateCounter.containsKey(key)) {
			stateCounter.setCount(key, 0.0);
		}
		stateCounter.incrementCount(key, 1.0);
	}

	private void collapseHorizontal(Tree<String> tree) {
	  if (tree.isLeaf()) return;
		for (Tree<String> subtree : tree.getChildren()) {
			collapseHorizontal(subtree);
		}
	  String nowLabel = tree.getLabel();
		double count = stateCounter.getCount(nowLabel);
		if (count < COUNT_THRESHOLD) {
			int begP = nowLabel.indexOf('[');
			if (begP == -1) return;
			int endP = nowLabel.lastIndexOf(']');
			String[] labels = nowLabel.substring(begP + 1, endP).split("_");
			if (labels.length < 2) return;
			StringBuilder newLabel = new StringBuilder(nowLabel.substring(0, begP));
			newLabel.append("[...");
			for (int i = 1; i < labels.length; ++i)newLabel.append(String.format("_%s", labels[i]));
			newLabel.append(nowLabel.substring(endP));
			tree.setLabel(newLabel.toString());
		}
	}

	private double myLogAdd(double a, double b) {
		if (useSloppyMath) return SloppyMath.logAdd(a, b);
		return a + b;
	}

	private boolean isValidScore(double a) {
		return !Double.isInfinite(a) && !Double.isNaN(a) && Double.isFinite(a);
	}
}

public class GenerativeParserFactory implements ParserFactory {
	public Parser getParser(List<Tree<String>> trainTrees) {
		return new GenerativeParser(trainTrees);
	}
}
