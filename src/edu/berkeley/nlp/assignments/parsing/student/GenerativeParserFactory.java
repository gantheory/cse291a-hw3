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

	List<List<Double>> tagScore;
	List<List<List<Double>>> unaryDP;
	List<List<List<Integer>>> unaryDPChild;
	List<List<List<Double>>> binaryDP;
	List<List<List<Integer>>> binaryDPLeftChild;
	List<List<List<Integer>>> binaryDPRightChild;
	List<List<List<Integer>>> binaryDPSplit;

	public Tree<String> getBestParse(List<String> sentence) {
//		System.out.println("Sentence");
//		sentence.set(0, "COTTON");
//		sentence.set(1, ":");
//		while (sentence.size() > 2) sentence.remove(sentence.size() - 1);
//		for (String word : sentence) {
//			System.out.print(word + " ");
//		}
//		System.out.println();
		int numOfStates = indexer.size();
		for (int state = 0; state < numOfStates; ++state) {
			for (int i = 0; i < sentence.size(); ++i) {
			  tagScore.get(state).set(i, MIN_LOG_PROB);
		  	for (int j = 0; j < sentence.size(); ++j) {
					unaryDP.get(state).get(i).set(j, MIN_LOG_PROB);
					unaryDPChild.get(state).get(i).set(j, -1);
					binaryDP.get(state).get(i).set(j, MIN_LOG_PROB);
					binaryDPLeftChild.get(state).get(i).set(j, -1);
					binaryDPRightChild.get(state).get(i).set(j, -1);
					binaryDPSplit.get(state).get(i).set(j, -1);
				}
			}
		}

		for (int state = 0; state < numOfStates; ++state) {
			String tag = indexer.get(state);
			for (int i = 0; i < sentence.size(); ++i) {
			  if (!lexicon.isKnown(sentence.get(i))) continue;
			  // score could be NaN
				double score = lexicon.scoreTagging(sentence.get(i), tag);
				if (score != MIN_LOG_PROB && Double.isFinite(score)) tagScore.get(state).set(i, score);
			}
		}
		if (debug) {
				for (int state = 0; state < numOfStates; ++state) {
					String tag = indexer.get(state);
					System.out.print(tag + ": ");
					for (int i = 0; i < sentence.size(); ++i)
						System.out.print(Math.exp(tagScore.get(state).get(i)) + " ");
					System.out.println();
				}
		}
		for (int state = 0; state < numOfStates; ++state) {
		  for (int i = 0; i < sentence.size(); ++i) {
				for (UnaryRule rule : unaryClosure.getClosedUnaryRulesByParent(state)) {
					double logRuleScore = rule.getScore();
					double logTerminalScore = tagScore.get(rule.getChild()).get(i);
					if (logRuleScore == MIN_LOG_PROB || logTerminalScore == MIN_LOG_PROB) continue;
					double logScore = myLogAdd(logRuleScore, logTerminalScore);
					unaryDP.get(state).get(i).set(i, logScore);
					unaryDPChild.get(state).get(i).set(i, rule.getChild());
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
							double logLeftChildScore = unaryDP.get(rule.getLeftChild()).get(i).get(k);
							double logRightChildScore = unaryDP.get(rule.getRightChild()).get(k + 1).get(j);
							if ((logLeftChildScore == MIN_LOG_PROB) || (logRightChildScore == MIN_LOG_PROB)) continue;
							double logScore = myLogAdd(myLogAdd(logRuleScore, logLeftChildScore), logRightChildScore);
							if ((binaryDP.get(state).get(i).get(j) == MIN_LOG_PROB) || (logScore > binaryDP.get(state).get(i).get(j))) {
								binaryDP.get(state).get(i).set(j, logScore);
								binaryDPLeftChild.get(state).get(i).set(j, rule.getLeftChild());
								binaryDPRightChild.get(state).get(i).set(j, rule.getRightChild());
								binaryDPSplit.get(state).get(i).set(j, k);
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
			  		double logChildScore = binaryDP.get(rule.getChild()).get(i).get(j);
			  		if (logChildScore == MIN_LOG_PROB) continue;
			  		double logScore = myLogAdd(logRuleScore, logChildScore);
			  		if ((unaryDP.get(state).get(i).get(j) == MIN_LOG_PROB) || (logScore > unaryDP.get(state).get(i).get(j))) {
			  		  unaryDP.get(state).get(i).set(j, logScore);
			  		  unaryDPChild.get(state).get(i).set(j, rule.getChild());
						}
					}
					if (!sameChildRule) {
						double logScore = binaryDP.get(state).get(i).get(j);
						if (logScore == MIN_LOG_PROB) continue;
						if ((unaryDP.get(state).get(i).get(j) == MIN_LOG_PROB) && (logScore > unaryDP.get(state).get(i).get(j))) {
							unaryDP.get(state).get(i).set(j, logScore);
							unaryDPChild.get(state).get(i).set(j, state);
						}
					}
				}
			}
		}
		if (debug) {
			System.out.println("Unary => ");
			for (int state = 0; state < numOfStates; ++state) {
				System.out.println(indexer.get(state) + ": ");
				for (int i = 0; i < sentence.size(); ++i) {
					for (int j = 0; j < sentence.size(); ++j) {
						System.out.print(Math.exp(unaryDP.get(state).get(i).get(j)) + " ");
					}
					System.out.println();
				}
			}
			System.out.println("Binary ==> ");
			for (int state = 0; state < numOfStates; ++state) {
				System.out.println(indexer.get(state) + ": ");
				for (int i = 0; i < sentence.size(); ++i) {
					for (int j = 0; j < sentence.size(); ++j) {
						System.out.print(Math.exp(binaryDP.get(state).get(i).get(j)) + " ");
					}
					System.out.println();
				}
			}
		}

    int rootID = indexer.indexOf("ROOT");
		if (unaryDP.get(rootID).get(0).get(sentence.size() - 1) == MIN_LOG_PROB) return NULL_TREE;
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
			int preTerminal = unaryDPChild.get(parent).get(l).get(r);
			Tree<String> treeWithTerminal = new Tree<String>(indexer.get(preTerminal), Collections.singletonList(new Tree<String>(sentence.get(l))));
			if (parent == preTerminal) return treeWithTerminal;
			return new Tree<String>(indexer.get(parent), Collections.singletonList(treeWithTerminal));
		}
		int child = unaryDPChild.get(parent).get(l).get(r);
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
	  int mid = binaryDPSplit.get(parent).get(l).get(r);
	  int leftChild = binaryDPLeftChild.get(parent).get(l).get(r);
	  int rightChild = binaryDPRightChild.get(parent).get(l).get(r);
	  Tree<String> leftTree = buildUnaryTree(leftChild, l, mid, sentence);
	  Tree<String> rightTree = buildUnaryTree(rightChild, mid + 1, r, sentence);
	  node.setChildren(new ArrayList<Tree<String>>(List.of(leftTree, rightTree)));
		return node;
	}

	public GenerativeParser(List<Tree<String>> trainTrees) {
//	  while (trainTrees.size() > 1) trainTrees.remove(trainTrees.size() - 1);
//		System.out.println(PennTreeRenderer.render(trainTrees.get(0)));
		System.out.print("Annotating / binarizing training trees ... ");
		List<Tree<String>> annotatedTrainTrees = annotateTrees(trainTrees);
//		System.out.println(PennTreeRenderer.render(annotatedTrainTrees.get(0)));

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

		tagScore = new ArrayList<>();
		for (int i = 0; i < numOfStates; ++i) {
		  tagScore.add(new ArrayList<>());
		  for (int j = 0; j < SENTENCE_LENGTH; ++j)
		  	tagScore.get(i).add(MIN_LOG_PROB);
		}

		unaryDP = new ArrayList<>();
		unaryDPChild = new ArrayList<>();
		binaryDP = new ArrayList<>();
		binaryDPLeftChild = new ArrayList<>();
		binaryDPRightChild = new ArrayList<>();
		binaryDPSplit = new ArrayList<>();
		for (int i = 0; i < numOfStates; ++i) {
			unaryDP.add(new ArrayList<>());
			unaryDPChild.add(new ArrayList<>());
			binaryDP.add(new ArrayList<>());
			binaryDPLeftChild.add(new ArrayList<>());
			binaryDPRightChild.add(new ArrayList<>());
			binaryDPSplit.add(new ArrayList<>());
			for (int j = 0; j < SENTENCE_LENGTH; ++j) {
				unaryDP.get(i).add(new ArrayList<>());
				unaryDPChild.get(i).add(new ArrayList<>());
				binaryDP.get(i).add(new ArrayList<>());
				binaryDPLeftChild.get(i).add(new ArrayList<>());
				binaryDPRightChild.get(i).add(new ArrayList<>());
				binaryDPSplit.get(i).add(new ArrayList<>());
				for (int k = 0; k < SENTENCE_LENGTH; ++k) {
					unaryDP.get(i).get(j).add(MIN_LOG_PROB);
					unaryDPChild.get(i).get(j).add(-1);
					binaryDP.get(i).get(j).add(MIN_LOG_PROB);
					binaryDPLeftChild.get(i).get(j).add(-1);
					binaryDPRightChild.get(i).get(j).add(-1);
					binaryDPSplit.get(i).get(j).add(-1);
				}
			}
		}
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
