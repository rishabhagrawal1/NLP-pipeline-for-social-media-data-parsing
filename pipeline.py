import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import StanfordTokenizer
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import GenericStanfordParser
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.tag.stanford import StanfordPOSTagger, StanfordNERTagger
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.wordnet import wordnet
from nltk.tree import Tree
from nltk import compat
from nltk import sent_tokenize
import re, collections
import csv
import sys
import tempfile
import os
import traceback

posTaggerPath = ".:/home/rishabh/posTagger/stanford-postagger-full-2015-12-09"
os.environ['CLASSPATH'] = posTaggerPath

sys.path.insert(0, r'/home//rishabh//script')
sys.path.insert(0, r'/home//rishabh//ner')
sys.path.insert(0, r'/home//rishabh//AmalGram//pysupersensetagger-master')
sys.path.insert(0, r'/home/rishabh/nltk_data/corpora/wordnet')
sys.path.insert(0, r'//home//rahool//dependency-parsing//tweebo-parser//TweeboParser//ark-tweet-nlp-0.3.2')
sys.path.insert(0, r'/home/rishabh/happyTockenizer')
sys.path.insert(0, r'/home/rishabh/wnut2015')
sys.path.insert(0, r'/home/rishabh/softwares')


##Python files to be used
from tweebo_twokenize import tokenizeRawTweetText
#from DependencyParse import process_stan_output
from SVOExtract import process_stan_output_svo
from happierfuntokenizing import *
from customNorm import *
from AmalG import runAmalGram
from nltk.tokenize import word_tokenize
#import mysql.connector

##TweeboTagger Python wrapper
## Reference is https://github.com/ianozsvald/ark-tweet-nlp-python

import subprocess
import shlex

# NOTE this command is directly lifted from runTagger.sh
#RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar /home/rahool/dependency-parsing/tweebo-parser/TweeboParser/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar --model /home/rishabh/softwares/model.ritter_ptb_alldata_fixed.20130723"
RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar /home/rahool/dependency-parsing/tweebo-parser/TweeboParser/ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar"
def _split_results(rows):
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                #yield tokens, tags, confidence
                ## We don't need confidence to take so dropping here
                yield tokens, tags

def subProcessFactoryTagger(isInputAmalgram, run_tagger_cmd=RUN_TAGGER_CMD):
    # build a list of args
    args = shlex.split(run_tagger_cmd)          
    if(isInputAmalgram == False):
        args.append('--model')
        args.append('/home/rishabh/softwares/model.ritter_ptb_alldata_fixed.20130723')
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return po

def _call_runtagger(tweets, isInputAmalgram, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    result = subProcessFactoryTagger(isInputAmalgram, run_tagger_cmd).communicate(tweets)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')
    pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, isInputAmalgram, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, isInputAmalgram, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    result = ""
    if(isInputAmalgram == True):
        for a in  pos_result[0]:
            result += a[0]
            result += '\t'
            result += a[1]
            result += '\n'
    else:
        result = []
        for a in  pos_result[0]:
            result.append(a[0] + '/'+ a[1])
    return result

##Stanford Parser
STANFORD_PARSER_PATH = '/home/rahool/dependency-parsing/stanford-parser/stanford-parser-full-2015-12-09/'
RUN_PARSER_CMD = "java -mx768m -cp \""+STANFORD_PARSER_PATH+"*:\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat wordsAndTags,typedDependencies"
ENCODING='utf8'              

def parseSentence(fileName, isTockenzied, isTagged, run_parser_cmd=RUN_PARSER_CMD):
    args = shlex.split(run_parser_cmd)
    if(isTagged == True):
        tagArgs = shlex.split("-tokenized -tagSeparator / -tokenizerFactory  edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerMethod newCoreLabelTokenizerFactory")
 
        for arg in tagArgs:
            args.append(arg)
    elif(isTockenzied == True):
        args.append("-tokenized")
    
    args.append("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    args.append(fileName)
    so = subprocess.check_output(args, stdin=subprocess.PIPE,  stderr=subprocess.PIPE)
    return so
 
def runStanfordParser(tweet, isTockenzied, isTagged, run_parser_cmd=RUN_PARSER_CMD):
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as input_file:
            # Write the actual sentences to the temporary input file
            input_file.write(' '.join(tweet))
            #input_file.write(tweet)
            input_file.flush()
            result = parseSentence(input_file.name, isTockenzied, isTagged)
    os.unlink(input_file.name)
    return result

def removeEmoticons(line):
    dictE = dict()
    ec = ")(:;'\"o-+=|\/$!"
    for i in range(len(ec)):
        dictE[ec[i]] = 1

    i = 0    
    while(i < len(line)-1):
        if line[i] in dictE and line[i+1] in dictE and line[i+1] != line[i]:
            line = line[:i] + line[(i+2):]
        i = i+1 
    return line    


def preProcessingPunctuation(line):
    dictP = dict()
    pc = ".!?-"
    for i in range(len(pc)):
        dictP[pc[i]] = 1
    #line = line.replace('\"', '')
    #line = line.replace('\'', '')
    #line = line.replace('\\', '')
    i = 0    
    while(i < len(line)-1):
        while(line[i] in dictP and line[i+1] == line[i]):
            line = line[:i] + line[(i+1):]
            if(i >= len(line) - 1):
                return line
        i = i+1
    #print line
    return line    


def addLineEnding(line):
    line = line.rstrip()
    dictS = dict()
    sc = ".!?"
    for i in range(len(sc)):
        dictS[sc[i]] = 1
    #print len(line) -1   
    #print line[len(line)-1] 
    if(len(line) > 0 and line[len(line)-1] not in dictS):
        line = line + '.'
    else:
        line = line + '.'
        #line = line + "\n"
    return line


def parse_trees_output(output_):
    res = []
    cur_lines = []
    cur_trees = []
    blank = False
    for line in output_.splitlines(False):
        if line == '':
            if blank:
                res.append(iter(cur_trees))
                cur_trees = []
                blank = False
            elif self._DOUBLE_SPACED_OUTPUT:
                cur_trees.append(self._make_tree('\n'.join(cur_lines)))
                cur_lines = []
                blank = True
            else:
                res.append(iter([self._make_tree('\n'.join(cur_lines))]))
                cur_lines = []
        else:
            cur_lines.append(line)
            blank = False
    return iter(res)



tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
def nltkTwitterTockenizer(line):
    return tknzr.tokenize(line)
    #parseStanford(tokens)
#line = "A rare black squirrel has become a regular visitor to a suburban garden"
#nltkTwitterTockenizer(line)



def tweeboTockenizer(line):
    return tokenizeRawTweetText(line)
    #parseStanford(tokens)
#line = "A rare black squirrel has become a regular visitor to a suburban garden"
#nltkTwitterTockenizer(line)


lmtzr = WordNetLemmatizer()
def doLemmatize(tokens, lemmatizeIndex):
    result = []
    i = 0
    for t in tokens:
        l = lmtzr.lemmatize(t)
        if(l != t):
            lemmatizeIndex[i] = t
        result.append(l) 
        i += 1
    return result


##NER 
## Reference https://pythonprogramming.net/named-entity-recognition-stanford-ner-tagger/
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
st = StanfordNERTagger('/home/rishabh/ner/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz',
                        '/home/rishabh/ner/stanford-ner-2015-12-09/stanford-ner.jar', encoding='utf-8')

def checkInNER(text):
    #tokenized_text = nltkTwitterTockenizer(text)
    #classified_text  = st.tag(tokenized_text)
    classified_text  = st.tag(text)
    if(classified_text[0][1] != 'O'):
        return True
    return False


##Handling hash tag
def extract_hash_tags(s):
    result = ""
    for part in s.split():
        if part.startswith('#'):
            if(len(part) > 1 and (wordnet.synsets(part[1:]) or checkInNER(part[1:]))):
                result = result + ' ' + part[1:]
        else:
            result = result + ' ' +part
    return result


##Model Learning
def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(file('.//ner//big.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

##Word corrector
def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

dictPunct = {}
punct = ")(:;'\"o-+=|\/$!,.?@#%^&~<>1234567890"
for i in range(len(punct)):
    dictPunct[punct[i]] = 1

def doCorrect(tokens, correctIndex):
    i = 0
    cTokens = []
    for token in tokens:
        token = token.lower()
        if(token not in dictPunct and checkInNER(token) != True):
            cToken = correct(token)
            cTokens.append(cToken)
            if(cToken != token):
                correctIndex[i] = token
        else:
            cTokens.append(token)
        i= i+1
    return cTokens

# In[ ]:

## get the original sentence back from current tokens list and saved dictionaries.
def getBackOriginal(tokens):
    text = ""
    
    ##Change token to original value
    if(correctIndex):
        for index, val in correctIndex.items():
            tokens[index] = val
    
    if(lemmatizeIndex):
        for index, val in lemmatizeIndex.items():
            tokens[index] = val

    ##Add more dictionaries in order here
    
    
    
    ##Create original sentence after all the modifications done
    for t in tokens:
        text = text + t + " "
    text = text[:(len(text) - 1)]
    
    ##Return original sentence
    return text

#correct[1] = "value"
#tokens = ['I', 'really', 'wana', 'move','soon','!']
#getBackOriginal(tokens)

##Nltk Sentence tockenizer
def sentenceTockenizer(message):
    sents = nltk.sent_tokenize(message)
    return sents


#inputType = "txtFile"
inputType = "csvFile"
#inputType = "mysqlDB"

skipHeaderCSV = True
idColumn = 0
userIdColumn = 1
msgColumn = 2

##sql Options
host = "db4free.net"
user = "rishabh"
password = "mittal123"
database = "nlp523" 
table = "uMessage"
idColumnDB = "msgId"
msgColumnDB = "msg"

## Paths
InputDir = "//home//rishabh//data//"  #can change address here
InputDir = "//shared-projects//"
#InputFile = "sample_fb.txt"                      #can change filename here
#InputFile = "sample_fb_original.txt"
#InputFile = "sampleFbCsv.csv"
InputFile = "fb22-1000.csv"
InputPath = InputDir + InputFile

OutputFile = "result"
OutputDir = "//home//rishabh//output//"  #can change address here
OutputPath = OutputDir + OutputFile
#print OutputPath

OutputFileCSV = "resultCSV"
OutputPathCSV = OutputDir + OutputFileCSV

##Processing Stats
OutputProcessStatsFile = "processedStats.txt"
OutputProcessStatsPath = OutputDir + OutputProcessStatsFile
#print OutputPath

##output Files
OutputSentenceTokenizerFile = "resultSentenceTokenizer"
OutputwordTokenizerFile = "resultwordTokenizer"
OutputwordNormalizedTokenizerFile = "resultNormalizedwordTokenizer"
OutputTweeboTaggerFile = "resulttweeboTagger"
OutputStanfordParserFile = "resulttStanfordParser"

##Output sub Folders
SentenceTokenizerFolder = OutputDir + "outputSentenceTok"+"//" 
WordTokenizerFolder = OutputDir + "outputWordTok"+"//"
NormalizedFolder = OutputDir + "outputNormalizer"+"//"
TweeboTaggerFolder = OutputDir + "outputTweeboTagger"+"//"
StanfordParserFolder = OutputDir + "outputSatnfordParser"+"//"
OutputFolder = OutputDir + "outputResult"+"//"

##Variables
count = 0
iterationCycle = 500
extentionCSV = ".csv"
extentionTXT = ".txt"
listSentences = []

##Dictionaries
correctIndex = {}
lemmatizeIndex = {}

## To choose tockenizer
#tockenizer = 'tweebo'
#tockenizer = 'nltktwitter'
#tockenizer = 'stanford'
tockenizer = 'happy'

## To select tweebo tockenizer(True) vs stanford default tagger(False)
isTagged = True
tokens = []
isTaggerAmalgram = True

##Need to get original sentence back
needOriginal = False

##Need Intermediate output
isNeedIntermediateData = False

##Which Normalizer to use
isExtNormalizer = True
numProcessed = 0
wr = 0
wrST = 0
wrWT = 0
wrNWT = 0
wrTT = 0
wrSP = 0

def convert_stan_format(line):
    if line is None:
        return ""
    list_pairs = line.split("\n\n")
    if list_pairs is None or len(list_pairs) <= 1:
        return ""
    return list_pairs[1].split('\n')
 
#print convert_stan_format(stan_sample)

def append_msg_id(sentence, msg_id):
    sentence = sentence.split('\n')
    for i in range(len(sentence)):
        if sentence[i] is None or len(sentence[i]) == 0:
            continue
        sentence[i] += '\t' + str(msg_id) + '\t' + str(msg_id)
    return ('\n').join(sentence)

#print append_msg_id(open("original").read(),12)

##Parse a single sentence
def doParseSentence(lineId, userId, sentence, listTokens, listNormalized, listTagger, listParser):   
    try:
        global correctIndex
        sentence = extract_hash_tags(sentence)
        
	##Tockenization phase
        if(tockenizer == 'tweebo'):
            tokens = tweeboTockenizer(sentence)
        elif(tockenizer == 'nltkTwitter'):
            tokens = nltkTwitterTockenizer(sentence)
        elif(tockenizer == 'happy'):
            tok = Tokenizer(preserve_case=False)
            tokens = tok.tokenize(sentence)
        elif(tockenizer == 'stanford'):
            tokens = StanfordTokenizer().tokenize(sentence)
        ##Write Intermediate result in file
        if(isNeedIntermediateData):
            listTokens.append(tokens)
	
	#print tokens
        ##Nomalization phase
        if(isExtNormalizer == True):
            ##Calling external normalizer
            tokens = testNorm(tokens)
        else:
	    ris = 1
            #tokens = doCorrect(tokens, correctIndex) 
            ##Lemmatizer or Stemmer
            #tokens = doLemmatize(tokens, lemmatizeIndex)
        if(isNeedIntermediateData):
            listNormalized.append(tokens)

        ##Selecting tagger
        if(isTagged):
            if(isTaggerAmalgram):
                tokens = runtagger_parse(' '.join(tokens), True)
                lineUserId = lineId + "__" + userId
                result = append_msg_id(tokens, lineUserId)
                #print "appended output is", result
                return result+"\n", listTokens,listNormalized, listTagger,listParser
            else:
                tokens = runtagger_parse(' '.join(tokens), False)
                if(isNeedIntermediateData):
                    listTagger.append(tokens)
            ##Parse data now
            result = runStanfordParser(tokens, True, True)
        else:
            ##Parse data now
            result = runStanfordParser(tokens, True, False)

        if(isNeedIntermediateData):
            listParser += " "+str(convert_stan_format(result))
    
        ##dependency processing code from aman start
        result = process_stan_output_svo(result)
        ##dependency processing code from aman end
    
        ##If need to get original text back start
        if(needOriginal and isExtNormalizer):
            originalText = getBackOriginal(tokens)
            correctIndex = {}
            lemmatizeIndex = {}
            print originalText
        ##If need to get original text back end    
        return result, listTokens,listNormalized, listTagger,listParser
    except:
        print(traceback.format_exc())


def createResultList(lineId, userId, result, listResult):
    listResult.append(lineId)
    listResult.append(userId)
    listResult.append(result)
    return listResult

def initLists(listTokens, listNormalized, listTagger,listParser):
    listTokens = []
    listTagger = []
    listParser = ""
    listNormalized = []
    return listTokens, listNormalized, listTagger,listParser

def writeLists(lineId, userId, sentences, listTokens, listNormalized, listTagger, listParser):
    global wr
    global wrST
    global wrWT
    global wrNWT
    global wrTT
    global wrSP
    listResult = []
    listResult = createResultList(lineId, userId, sentences, listResult)
    wrST.writerow(listResult)
    listResult = []
    listResult = createResultList(lineId, userId, listTokens, listResult)
    wrWT.writerow(listResult)
    listResult = []
    listResult = createResultList(lineId, userId, listNormalized, listResult)
    wrNWT.writerow(listResult)
    listResult = []
    listResult = createResultList(lineId, userId, listTagger, listResult)
    wrTT.writerow(listResult)
    listResult = []
    listResult = createResultList(lineId, userId, listParser, listResult)    
    wrSP.writerow(listResult)

## Parse a line. May contain multiple sentences
def doParseLine(lineId, userId, line, lineResult):
    try:
        ##Remove emoticons
        line = removeEmoticons(line)
        ##Fix extra punctuation marks
        line = preProcessingPunctuation(line)
        ##Fix line endings
        line = addLineEnding(line)
        ##Sentence Tockenizer
        sentences = sentenceTockenizer(line)
	#print "Sentence tokenized", sentences
        listTokens = [] 
        listNormalized = []
        listTagger =[]
        listParser = ""
        
        ##Now parse each sentence seprately
        for sentence in sentences:
	    #print "before", sentence
	    #sentence = sentence.decode('unicode_escape').encode('ascii','ignore')
	    #print "after", sentence
            ##Call for individual sentence parsing
            result, listTokens, listNormalized, listTagger,listParser = doParseSentence(lineId, \
                        userId, sentence, listTokens, listNormalized, listTagger, listParser)
            ##Create output result set with appending sentence parsed output$$Start
	    if(isTaggerAmalgram == False):
	        if(len(lineResult) > 0 and len(result) > 0):
                    lineResult += "**"
                result = '**'.join(result)
	    lineResult += result
        ##Create output result set with appending sentence parsed output$$End

        ##Write Intermediate results in file start
        if(isNeedIntermediateData):
            writeLists(lineId, userId, sentences, listTokens, listNormalized, listTagger, listParser)
            listTokens, listNormalized, listTagger,listParser = initLists(listTokens, listNormalized, \
                                                                          listTagger,listParser)
        ##Write Intermediate results in file end
        return lineResult
    except Exception,e:
        print "Error in line with id:",lineId, "Line is:", line
        print "Original error returned was:", str(e)
	print(traceback.format_exc())

def createOutputFilePath(dirName, fileName, iteration, extention):
    return dirName + fileName + "-" + str(iteration) + extention

def createIntermediateFolders(folder1, folder2, folder3):
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    if not os.path.exists(folder3):
        os.makedirs(folder3) 

##Logic for txt file parsing
def operateOnTxtFile(InputPath, OutputPath):
    try:
        with open(InputPath) as infile:
            parsedResult = []
            lineId = 0
            with open(OutputPath, 'w') as outFile:
                for line in infile:
		    print line
                    lineResult = ''
                    ##Add line number in start of list
                    lineResult += str(lineId)
		    lineResult += '##'
                    ##Call for individual line parsing. May contain multiple sentences
                    lineResult = doParseLine(lineId, 0, line, lineResult)
                    outFile.write(lineResult)
		    outFile.write('\n')	
                    parsedResult.append(lineResult)
                    lineId += 1
                    #print "lineResult is:", lineResult
    except Exception,e:
        print "error occured:", str(e)
	print(traceback.format_exc())
	traceback.print_exc()
        #sys.exit()
    print "parsedResult is:", parsedResult
    return parsedResult

##Logic for CSv Parsing
def operateOnCsvFile(InputPath, OutputPath, idColumn, userIdColumn, msgColumn):
    try:
        if not os.path.exists(OutputDir):
            os.makedirs(OutputDir)
        if(isTaggerAmalgram == False):
            global wr
            global wrST
            global wrWT
            global wrNWT
            global wrTT
            global wrSP
            global numProcessed    
            createIntermediateFolders(SentenceTokenizerFolder, WordTokenizerFolder, NormalizedFolder)
            createIntermediateFolders(TweeboTaggerFolder, StanfordParserFolder, OutputFolder)
	outStatsFile = open(OutputProcessStatsPath, 'w')
	count = numProcessed
        numRows = 0
        csvfile = open(InputPath)
        readCSV = csv.reader(csvfile, delimiter=',')
        for count1,line1 in enumerate(readCSV,0):
            numRows = numRows + 1
        csvfile.close()
        with open(InputPath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',',  skipinitialspace = True)
            #if(skipHeaderCSV):
            #    next(readCSV)
            while(numProcessed):
		next(readCSV)
                numProcessed -= 1
	    inputAmalgram = ""
            for line in readCSV:
		print line
                ##To be removed after check end  
                if(count % iterationCycle == 0):
                    if(isTaggerAmalgram == False):               
                        OutputPath = createOutputFilePath(OutputFolder, OutputFile, int(count/iterationCycle), extentionCSV)
                        outFile = open(OutputPath, 'w')
                        wr = csv.writer(outFile, quoting=csv.QUOTE_ALL)
                        OutputSTPath = createOutputFilePath(SentenceTokenizerFolder, OutputSentenceTokenizerFile, int(count/iterationCycle), extentionCSV)
                        outSTFile = open(OutputSTPath, 'w')
                        wrST = csv.writer(outSTFile, quoting=csv.QUOTE_ALL)
                        OutputWTPath = createOutputFilePath(WordTokenizerFolder, OutputwordTokenizerFile, int(count/iterationCycle), extentionCSV)
                        outWTFile = open(OutputWTPath, 'w')
                        wrWT = csv.writer(outWTFile, quoting=csv.QUOTE_ALL)
                        OutputNWTPath = createOutputFilePath(NormalizedFolder, OutputwordNormalizedTokenizerFile, int(count/iterationCycle), extentionCSV)
                        outNWTFile = open(OutputNWTPath, 'w')
                        wrNWT = csv.writer(outNWTFile, quoting=csv.QUOTE_ALL)
                        OutputTTPath = createOutputFilePath(TweeboTaggerFolder, OutputTweeboTaggerFile, int(count/iterationCycle), extentionCSV)
                        outTTFile = open(OutputTTPath, 'w')
                        wrTT = csv.writer(outTTFile, quoting=csv.QUOTE_ALL)
                        OutputSPPath = createOutputFilePath(StanfordParserFolder, OutputStanfordParserFile, int(count/iterationCycle), extentionCSV)
                        outSPFile = open(OutputSPPath, 'w')
                        wrSP = csv.writer(outSPFile, quoting=csv.QUOTE_ALL)
                    else:
                        OutputPath = createOutputFilePath(OutputDir, OutputFile, int(count/iterationCycle), extentionTXT)
                        outFile = open(OutputPath, 'w') 
                        OutputPathCSV = createOutputFilePath(OutputDir, OutputFileCSV, int(count/iterationCycle), extentionCSV)
                        outFileCSV = open(OutputPathCSV, 'w')
                        wr = csv.writer(outFileCSV, quoting=csv.QUOTE_ALL)    

                ##Call for individual line parsing. May contain multiple sentences
                lineResult = ""
                if(len(line[msgColumn]) == 0):
		    continue
		line[msgColumn] = line[msgColumn].replace('\"', '')
	    line[msgColumn] = line[msgColumn].replace('\'', '')
		line[msgColumn] = line[msgColumn].replace('\\', '')
		line[msgColumn] = line[msgColumn].decode('unicode_escape').encode('ascii','ignore')
		lineResult = doParseLine(line[idColumn], line[userIdColumn], line[msgColumn], lineResult)
	
		if(isTaggerAmalgram == True):
    		    inputAmalgram += lineResult
    	            inputAmalgram += "\n"
    	            if((count % iterationCycle == iterationCycle -1) or (count == numRows-1)):
    		        amalgramResult = ""
    			#print "Final input for Amalgram is\n",inputAmalgram
			try:
    			    amalgramDictFull, amalgramDictPartial = runAmalGram(inputAmalgram)
                            for key, value in amalgramDictFull.items():
    			        amalgramResultCSV = []
    			        msgId__userId = key.split("__")
    			        amalgramResult += msgId__userId[0] + "\t" + msgId__userId[1]+"\t"+value+"\n"
			        amalgramResultCSV = createResultList(msgId__userId[0], msgId__userId[1], amalgramDictPartial[key], amalgramResultCSV)
    			        wr.writerow(amalgramResultCSV)
    			    outFile.write(amalgramResult)
			    outStatsFile.write(str(count) + "\t" + str(line[idColumn]) + "\t" + str(line[userIdColumn]))
			except:
			    print "error occured"
			    print(traceback.format_exc())
    		        inputAmalgram = ""
                        outFile.close()
                        outFileCSV.close()
		else:
                    listResult = []
                    listResult = createResultList(line[idColumn], line[userIdColumn], lineResult, listResult)    
                    wr.writerow(listResult)
		    if((count % iterationCycle == iterationCycle -1) or (count == numRows-1)):
                        outStatsFile.write(str(count) + "\t" + str(line[idColumn]) + "\t" + str(line[userIdColumn]))
                        outFile.close()
                        outSTFile.close()
                        outWTFile.close()
                        outNWTFile.close()
                        outTTFile.close()
                        outSPFile.close()
                count += 1
	outStatsFile.write(str(count) + "\t" + str(line[idColumn]) + "\t" + str(line[userIdColumn]))
	outStatsFile.close()
    except Exception,e:
        print "error occured:", str(e)
        print(traceback.format_exc())
	    #sys.exit()

##Logic for Database data Parsing
def connectMysqlDB(host, user, password, database):
    conn = mysql.connector.Connect(host=host, user=user,\
                        password=password, database=database)
    c = conn.cursor()
    return conn,c

def operateOnMysqlData(host, user, password, database, table, idColumn, userIdColumn, msgColumn):
    try:
        conn,c = connectMysqlDB(host, user, password, database)
        query = "select * from " + table +";"
        c.execute(query)
        
        parsedResult = []
        with open(OutputPath, 'w') as outFile:
            for line in c:
                print line
                lineResult = ''
                ##Assuming message id or row id is in first column of 
                lineResult += line[idColumn] ## for appending msg id
                lineResult += "##" 
                lineResult += line[userIdColumn] ## for appending user id
                lineResult += "##"
                ##Call for individual line parsing. May contain multiple sentences
                lineResult = doParseLine(line[idColumn], line[userIdColumn], line[msgColumn], lineResult)
                outFile.write(lineResult)
                outFile.write('\n')
                parsedResult.append(lineResult)
                print "lineResult is:", lineResult
                print "\n"
        #print "parsedResult is",parsedResult
        return parsedResult

        c.close()
    except Exception,e:
        print "error occured:", str(e)
        #sys.exit() 

if(inputType == "txtFile"):
    operateOnTxtFile(InputPath, OutputPath)
elif(inputType == "csvFile"):
    operateOnCsvFile(InputPath, OutputPath, idColumn, userIdColumn, msgColumn)
#elif(inputType == "mysqlDB"):
   # operateOnMysqlData(host, user, password, database, table, idColumn, userIdColumn, msgColumn)

