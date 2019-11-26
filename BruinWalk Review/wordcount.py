import re 
import operator
import time
from collections import Counter
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords

import bs4 
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup 

def scrape():
	m_url = input("Please enter a bruinwalk URL: ")
	if m_url == "":
		print("Whoops, you forgot to enter a URL! We'll just use Professor Smallberg")
		m_url = "https://www.bruinwalk.com/professors/david-a-smallberg/com-sci-31/"

	uClient = uReq(m_url) #download 
	m_html = uClient.read() #dump content into variable
	uClient.close()

	my_page = soup(m_html, 'html.parser')

	prof_name = my_page.find('div', {'class': 'professor-info'}).get_text()
	prof_name = prof_name.strip('\n\r\t ')
	class_name = my_page.find('div', {'class': 'hide-for-small-only profile-header row'})
	class_name = class_name.find('h2').get_text()
	#print(class_name)
	# print(prof_name)
	if prof_name == '':
		print("Sorry, we couldn't find anything for this URL, try again!")
		exit()
	else:
		print("\nOkay, Scraping Reviews for", class_name, "taught by", prof_name, "\b....")
	
	review_body = my_page.findAll('div', {'class' : 'review bruinwalk-card'})

	filename = 'test_label.txt'
	f = open(filename, 'w')
	for i in range(len(review_body)):
		body = review_body[i].findAll('p')
		for x in body:
			f.write(x.get_text())

	f.close()

	print('All done!\n')
	# print(review_text)
	return prof_name, class_name




def filter(unfiltered_list, filtered_list):
	filter_words = { 'THE', 'AND', 'YOU', 'IS', 'ARE', 'TO', 'I', 'HE', 'SHE', 'IN', 'CLASS', 'SO', 'OF', 'THAT', 'A', 'FOR', 'IT', 'IS', 'HIS', 'GET', 'AN', 'BUT', 'WITH', 'IF' }
	# stopWords = set(stopwords.words('english'))
	for word in unfiltered_list:
		if word[0] not in filter_words:
			filtered_list.append(word)

def filter2(unfiltered_list, good_list, bad_list):
	good_words = {'SMART', 'GOOD', 'AMAZING', 'BRILLIANT', 'FUN', 'FUNNY', 'EASY', 'GREAT', 'BEST', 'ENGAGING' 'LOVE'}
	bad_words = {'BAD', 'WORST', 'WORSE', 'HARD', 'HARDEST', 'DIFFICULT', 'ANNOYING', 'HATE', 'DISLIKE', 'BORING', 'UGLY', 'UNCLEAR'}
	for word in unfiltered_list:
		if word[0] in good_words:
			good_list.append(word)
		elif word[0] in bad_words:
			bad_list.append(word)
def sum(word_list):
	sum = 0
	for word in word_list:
		sum += word[1]
	return sum


def main():
	prof_name, class_name = scrape()
	# open file, read words into cap_words
	with open('test_label.txt') as f:
		passage = f.read()
	words = re.findall(r'\w+', passage)
	cap_words = [word.upper() for word in words]

	# assign word counts
	word_counts = Counter(cap_words)
	# print(word_counts)

	sorted_word_counts = sorted(word_counts.items(), key = operator.itemgetter(1), reverse=True)
	filtered_swc = []
	good_swc = []
	bad_swc= []
	filter(sorted_word_counts, filtered_swc)
	filter2(sorted_word_counts, good_swc, bad_swc)

	# for i in range(len(filtered_swc)):
	# 	# word = entry[0]
	# 	entry = filtered_swc[i]
	# 	print(entry[0], entry[1])
	margin = 12
	num_gw = sum(good_swc)
	num_bw = sum(bad_swc)

	print('Good words:')
	for gw in good_swc:
		print(gw[0], gw[1])
	print("-" * margin)
	print("TOTAL:", num_gw, "\n")

	print('\nBad words:')
	for bw in bad_swc:
		print(bw[0], bw[1])

	print("-" * margin)
	print("TOTAL:", num_bw, "\n")
	#print("")

	if num_gw > num_bw:
		print(prof_name, "is a GOOD professor for", class_name, "\b.")
	elif num_gw < num_bw:
		print(prof_name, "is a BAD professor for", class_name, "\b.")
	else:
		print("People have mixed feelings about", prof_name, "for", class_name, "\b.")


if __name__ == "__main__":
	main()