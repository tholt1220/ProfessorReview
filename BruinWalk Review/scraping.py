import bs4 
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup 
import numpy as np
from string import punctuation
from sklearn.svm import SVC

def scrape():
	m_url = input("Please enter a bruinwalk URL: ")
	if m_url == "":
		print("Whoops, you forgot to enter a URL! We'll just use Gene Block")
		m_url = "https://www.bruinwalk.com/professors/gene-block/physci-199/"

	uClient = uReq(m_url) #download 
	m_html = uClient.read() #dump content into variable
	uClient.close()

	my_page = soup(m_html, 'html.parser')

	prof_name = my_page.find('div', {'class': 'professor-info'}).get_text()
	prof_name = prof_name.strip('\n\r\t ')
	# print(prof_name)
	if prof_name == '':
		print("Sorry, we couldn't find anything for this URL, try again!")
		exit()
	else:
		print("\nOkay, Scraping Reviews for", prof_name, "\b.")

	review_body = my_page.findAll('div', {'class' : 'review bruinwalk-card'})

	filename = 'test_label.txt'
	f = open(filename, 'w')
	for i in range(len(review_body)):
		body = review_body[i].findAll('p')
		for x in body:
			f.write(x.get_text())

	f.close()

	print('All done!')
	# print(review_text)


def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(infile):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    with open(infile, 'rU') as fid :
    	lines = fid.readlines() #read file line by line 
    	for input_string in lines:
    		for c in punctuation:
    			input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    index = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        lines = fid.readlines() #read file line by line 
        for line in lines:
        	words = extract_words(line) #get list of words using extract_words
        	for word in words: #update dictionay if we have encountered new words
        		if word not in word_list.keys():
        			word_list.update({word : index})
        			index += 1


        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors2(words, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = 1
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    for word in words:
    	if word in word_list:
    		idx = word_list[word]
    	feature_matrix[0][idx] = 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    index = 0
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        lines = fid.readlines() #read file line by line 
        for line in lines:
        	words = extract_words(line) #get list of words using extract_words
        	for word in words: #update dictionay if we have encountered new words
        		if word not in word_list.keys():
        			word_list.update({word : index})
        			index += 1


        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        lines = fid.readlines()
        for i in range(num_lines):
        	words = extract_words(lines[i]) #get list of words using extract_words
        	for word in words:
        		idx = word_list[word]
        		feature_matrix[i][idx] = 1
        ### ========== TODO : END ========== ###
        
    return feature_matrix



def main() :
    scrape()
    
    # read the tweets and its labels   
    words = extract_words('./test_label.txt')
    dictionary = extract_dictionary('./trainingX.txt')
    train_X = extract_feature_vectors('./trainingX.txt', dictionary)
    train_y = read_vector_file('./trainingy.txt')

    # print(words)
    vector = extract_feature_vectors2(words, dictionary)
    Best_Gamma = 0.03125
    Best_C =  1.0
    ### ========== TODO : END ========== ###
    clf = SVC(kernel = 'rbf', C = Best_C, gamma = Best_Gamma)
    clf.fit(train_X, train_y)
    prediction = clf.predict(vector)
    print(prediction)

if __name__ == "__main__" :
    main()