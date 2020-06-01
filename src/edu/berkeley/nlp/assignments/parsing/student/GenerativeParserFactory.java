package edu.berkeley.nlp.assignments.parsing.student;

import edu.berkeley.nlp.assignments.parsing.Grammar;
import edu.berkeley.nlp.assignments.parsing.SimpleLexicon;
import edu.berkeley.nlp.assignments.parsing.TreeAnnotations;
import edu.berkeley.nlp.assignments.parsing.UnaryClosure;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.math.SloppyMath;
import java.lang.reflect.Array;
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

	public static class GenerativeParserFactory implements ParserFactory {

		public Parser getParser(List<Tree<String>> trainTrees) {
			return new GenerativeParser(trainTrees);
		}
	}

	final boolean debug = false;
	final double EPSILON = 1e-9;
	final double MIN_LOG_PROB = Double.NEGATIVE_INFINITY; // Math.log(EPSILON);
	final int SENTENCE_LENGTH = 40;
	final Tree<String> NULL_TREE = new Tree<String>("ROOT", Collections.singletonList(new Tree<String>("JUNK")));
	final boolean useSloppyMath = false;

	CounterMap<List<String>, Tree<String>> knownParses;

	CounterMap<Integer, String> spanToCategories;

	Grammar grammar;

	SimpleLexicon lexicon;

	UnaryClosure unaryClosure;

	Indexer<String> indexer;

	double[][] tagScore;
	double[][][] unaryDP;
	int[][][] unaryDPChild;
	double[][][] binaryDP;
	int[][][] binaryDPLeftChild;
	int[][][] binaryDPRightChild;
	int[][][] binaryDPSplit;


	public Tree<String> getBestParse(List<String> sentence) {
		int numOfStates = indexer.size();
		for (int state = 0; state < numOfStates; ++state) {
			for (int i = 0; i < sentence.size(); ++i) {
			  tagScore[state][i] = MIN_LOG_PROB;
		  	for (int j = 0; j < sentence.size(); ++j) {
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
				if (score != MIN_LOG_PROB && Double.isFinite(score)) tagScore[state][i] = score;
			}
		}

		for (int state = 0; state < numOfStates; ++state) {
		  for (int i = 0; i < sentence.size(); ++i) {
				for (UnaryRule rule : unaryClosure.getClosedUnaryRulesByParent(state)) {
					double logRuleScore = rule.getScore();
					double logTerminalScore = tagScore[rule.getChild()][i];
					if (logRuleScore == MIN_LOG_PROB || logTerminalScore == MIN_LOG_PROB) continue;
					double logScore = myLogAdd(logRuleScore, logTerminalScore);
          unaryDP[state][i][i] = logScore;
					unaryDPChild[state][i][i] = rule.getChild();
				}
			}
		}

		for (int length = 2; length <= sentence.size(); ++length) {
			for (int state = 0; state < numOfStates; ++state) {
				for (int i = 0; i < sentence.size() - length + 1; ++i) {
					int j = i + length - 1;
					for (BinaryRule rule : grammar.getBinaryRulesByParent(state)) {
						double logRuleScore = rule.getScore();
						if (logRuleScore == MIN_LOG_PROB) continue;
						for (int k = i; k < j; ++k) {
							double logLeftChildScore = unaryDP[rule.getLeftChild()][i][k];
							double logRightChildScore = unaryDP[rule.getRightChild()][k + 1][j];
							if ((logLeftChildScore == MIN_LOG_PROB) || (logRightChildScore == MIN_LOG_PROB)) continue;
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
					boolean sameChildRule = false;
					for (UnaryRule rule : unaryClosure.getClosedUnaryRulesByParent(state)) {
					  if (rule.getChild() == state) sameChildRule = true;
			  		double logRuleScore = rule.getScore();
			  		if (logRuleScore == MIN_LOG_PROB) continue;
						double logChildScore = binaryDP[rule.getChild()][i][j];
			  		if (logChildScore == MIN_LOG_PROB) continue;
			  		double logScore = myLogAdd(logRuleScore, logChildScore);
			  		if ((unaryDP[state][i][j] == MIN_LOG_PROB) || (logScore > unaryDP[state][i][j])) {
			  			unaryDP[state][i][j] = logScore;
			  			unaryDPChild[state][i][j] = rule.getChild();
						}
					}
				}
			}
		}

    int rootID = indexer.indexOf("ROOT");
		if (unaryDP[rootID][0][sentence.size() - 1] == MIN_LOG_PROB) return NULL_TREE;
		Tree<String> parseTree = buildUnaryTree(rootID, 0, sentence.size() - 1, sentence);

		if (debug) {
			System.out.println("Success!");
			System.out.println(PennTreeRenderer.render(parseTree));
			System.out.println(PennTreeRenderer.render(TreeAnnotations.unAnnotateTree(parseTree)));
		}

		return TreeAnnotations.unAnnotateTree(parseTree);
	}

	private Tree<String> buildUnaryTree(int parent, int l, int r, List<String> sentence) {
		if (l == r) {
			int preTerminal = unaryDPChild[parent][l][r];
			Tree<String> treeWithTerminal = new Tree<String>(indexer.get(preTerminal), Collections.singletonList(new Tree<String>(sentence.get(l))));
			if (parent == preTerminal) return treeWithTerminal;
			return new Tree<String>(indexer.get(parent), Collections.singletonList(treeWithTerminal));
		}
		int child = unaryDPChild[parent][l][r];
		if (parent == child) return buildBinaryTree(child, l, r, sentence);

		Tree<String> node = new Tree<String>(indexer.get(parent));

		UnaryRule rule = new UnaryRule(parent, child);
		List<Integer> closurePath = unaryClosure.getPath(rule);
		if (closurePath.size() > 2) {
			Tree<String> prev = node;
			for (int i = 1; i < closurePath.size() - 1; ++i) {
				Tree<String> now = new Tree<String>(indexer.get(closurePath.get(i)));
				prev.setChildren(Collections.singletonList(now));
				prev = prev.getChildren().get(0);
			}
			Tree<String> childTree = buildBinaryTree(child, l, r, sentence);
			prev.setChildren(Collections.singletonList(childTree));
			return node;
		} else {
			Tree<String> childTree = buildBinaryTree(child, l, r, sentence);
			node.setChildren(Collections.singletonList(childTree));
			return node;
		}
	}

	private Tree<String> buildBinaryTree(int parent, int l, int r, List<String> sentence) {
		if (l == r) return buildUnaryTree(parent, l, r, sentence);
		Tree<String> node = new Tree<String>(indexer.get(parent));
		int mid = binaryDPSplit[parent][l][r];
		int leftChild = binaryDPLeftChild[parent][l][r];
		int rightChild = binaryDPRightChild[parent][l][r];
	  Tree<String> leftTree = buildUnaryTree(leftChild, l, mid, sentence);
	  Tree<String> rightTree = buildUnaryTree(rightChild, mid + 1, r, sentence);
	  node.setChildren(new ArrayList<Tree<String>>(List.of(leftTree, rightTree)));
		return node;
	}

	public GenerativeParser(List<Tree<String>> trainTrees) {
		System.out.print("Annotating / binarizing training trees ... ");
		List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);

		System.out.println("done.");
		System.out.print("Building grammar ... ");
		grammar = Grammar.generativeGrammarFromTrees(annotatedTrainTrees);
		System.out.println("done. (" + grammar.getLabelIndexer().size() + " states)");

		lexicon = new SimpleLexicon(annotatedTrainTrees);

		unaryClosure = new UnaryClosure(grammar.getLabelIndexer(), grammar.getUnaryRules());

		indexer = grammar.getLabelIndexer();

		int numOfStates = grammar.getLabelIndexer().size();

		if (debug) {
			System.out.println("Binary rules");
			for (BinaryRule rule : grammar.getBinaryRules()) {
				System.out.println(indexer.get(rule.getParent()) + " -> (" + indexer.get(rule.getLeftChild()) + ", " + indexer.get(rule.getRightChild()) + ") ==> " + Math.exp(rule.getScore()));
			}

			System.out.println("Unary rules");
			for (UnaryRule rule : grammar.getUnaryRules()) {
				System.out.println(indexer.get(rule.getParent()) + " -> " + indexer.get(rule.getChild()) + " ==> " + Math.exp(rule.getScore()));
			}

			System.out.println("All lexicon");
			for (String s : lexicon.getAllTags()) {
				System.out.println(s);
			}
			System.out.println();

			System.out.println("All states");
			for (int i = 0; i < numOfStates; ++i) {
				System.out.println(i + ": " + grammar.getLabelIndexer().get(i));
			}
			System.out.println();
		}

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
			annotatedTrees.add(TreeAnnotations.annotateTreeLosslessBinarization(tree));
		}
		return annotatedTrees;
	}

	private double myLogAdd(double a, double b) {
		if (useSloppyMath) return SloppyMath.logAdd(a, b);
		return a + b;
	}
}

public class GenerativeParserFactory implements ParserFactory {
	public Parser getParser(List<Tree<String>> trainTrees) {
		return new GenerativeParser(trainTrees);
	}
}
