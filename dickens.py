###############################
## This is the preprocessing script for the glove_embeddings tutorial. I do not recommend you run this, as it takes quite a while
###############################

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

training= ['training//eng.txt','training//ger.txt','training//fre.txt']
test = ['test//test_eng.txt','test//test_ger.txt','test//test2_ger.txt','test//test_fre.txt','test//test2_fre.txt']


###########################################################
#creating the file to train our vectors
############################################################
tokenlist = []
for file in training: #there are 3 files in this list
	f=open(file,'r', encoding='utf-8')      # read in the file, make sure to mark as utf8
	file = f.read()
	f.close()
	tokens= tokenizer.tokenize(file)
	tokens = tokens[400:-5000] #clearing out the Gutenberg parts
	tokenlist.append(tokens)
tokens = [*tokenlist[0],*tokenlist[1],*tokenlist[2]]
#print(tokens)
tokenfile= open('vector_tokens.txt','w+', encoding='utf-8')
for token in tokens:
	token = [i+" " for i in token.lower()]
	for i in token:
		tokenfile.write(i)
	tokenfile.write("\n")
tokenfile.close()



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
		train_list_tokens.append(token) #saving these to weed out the test items later
		token = [i+" " for i in token.lower()]
		token.append(tag)
		tokenlist.append(token)

train_tokens = tokenlist #[*tokenlist[0],*tokenlist[1],*tokenlist[2]]

trainfile= open('train_tokens.txt','w+', encoding='utf-8')
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
			temp = [i+" " for i in token.lower()]
			temp.append(tag)
			tokenlist.append(temp)
			print("found it")
			print(c)
			c +=1

test_tokens = tokenlist #[*tokenlist[0],*tokenlist[1],*tokenlist[2]]

trainfile= open('test_tokens.txt','w+', encoding='utf-8')
for token in test_tokens:
	for i in token:
		trainfile.write(i)
	trainfile.write("\n")
trainfile.close()



###########################################################
#cleaning the test file to only contain unknown words
############################################################

#testing = open('test_tokens.txt','w+', encoding='utf-8')
#testfile = testing.read()

#training= open('train_tokens.txt','r', encoding='utf-8')
#trainfile=training.read()
#training.close()

#test_data = testfile.split("\n")
#train_data = trainfile.split("\n")

#for item in test_data:
#	if item in train_data:
#		print("fond it")
#		testing.write(item)
#		testing.write("\n")
#testing.close()
