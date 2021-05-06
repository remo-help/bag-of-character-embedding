from nltk.tokenize import RegexpTokenizer

##########
###### This script preprocesses the data for the CountVectorizer
###########

tokenizer = RegexpTokenizer(r'\w+')

training= ['training//eng.txt','training//ger.txt','training//fre.txt']
test = ['test//test_eng.txt','test//test_ger.txt','test//test2_ger.txt','test//test_fre.txt','test//test2_fre.txt']


###########################################################
#creating the file to train classifier
############################################################
tokenlist=[]
train_list_tokens=[]
for filestring in training:
	f=open(filestring,'r', encoding='utf-8')      # read in the file, make sure to mark as utf8
	file = f.read()
	f.close()
	tag = "\t" + filestring[-7:-4] ###adding the language tag that we want
	tokens= tokenizer.tokenize(file)
	tokens = tokens[400:-5000] #clearing out the Gutenberg parts  
	for token in tokens:
		train_list_tokens.append(token)
		token =token.lower()
		token =token+tag
		tokenlist.append(token)
		

train_tokens = tokenlist #[*tokenlist[0],*tokenlist[1],*tokenlist[2]]

trainfile= open('count_train_tokens.txt','w+', encoding='utf-8')
for token in train_tokens:
	for i in token:
		trainfile.write(i)
	trainfile.write("\n")
trainfile.close()

###########################################################
#creating the file to test the classifier
############################################################
tokenlist=[]
c=0
for filestring in test:
	f=open(filestring,'r', encoding='utf-8')      # read in the file, make sure to mark as utf8
	file = f.read()
	f.close()
	tag = "\t" + filestring[-7:-4] ###adding the language tag that we want
	tokens= tokenizer.tokenize(file)
	tokens = tokens[400:-5000] #clearing out the Gutenberg parts 
	for token in tokens:
		if token in train_list_tokens:
			print("this aint it")
			print(c)
			c +=1
			pass
		else:
			token =token.lower()
			token =token+tag
			tokenlist.append(token)
			print("found it")
			print(c)
			c +=1

test_tokens = tokenlist #[*tokenlist[0],*tokenlist[1],*tokenlist[2]]

trainfile= open('count_test_tokens.txt','w+', encoding='utf-8')
for token in test_tokens:
	for i in token:
		trainfile.write(i)
	trainfile.write("\n")
trainfile.close()

