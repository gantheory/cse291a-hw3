#!/bin/bash
ant -f ./build_assign_parsing.xml
# java -cp assign_parsing.jar:assign_parsing-submit.jar -server -mx2000m edu.berkeley.nlp.assignments.parsing.PCFGParserTester -path ./wsj -parserType BASELINE -maxTrainLength 15 -maxTestLength 15 -quiet
java -cp assign_parsing.jar:assign_parsing-submit.jar -server -mx2000m edu.berkeley.nlp.assignments.parsing.PCFGParserTester -path ./wsj -parserType GENERATIVE -maxTrainLength 15 -maxTestLength 15 -quiet
# java -cp assign_parsing.jar:assign_parsing-submit.jar -server -mx2000m edu.berkeley.nlp.assignments.parsing.PCFGParserTester -path ./wsj -parserType GENERATIVE -maxTrainLength 3 -maxTestLength 3
