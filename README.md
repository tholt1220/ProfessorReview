# ProfessorReview
Uses web scraping across different teacher rating websites (BruinWalk, RateMyProfessors) to find the most common tags for UCLA professors, including difficulty, kindness, and course value

## Installation

This project relies on [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) and [urllib](https://docs.python.org/3/library/urllib.request.html#module-urllib.request) for webscraping 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
pip install bs4 urllib
```

## Usage

In the terminal, run the following command to start the ProfessorReview script:
```bash
python3 wordcount.py
```

The terminal will prompt the user to enter a valid [BruinWalk URL](https://www.bruinwalk.com/professors/david-a-smallberg/com-sci-31/).

Sample Response: 
```
Okay, Scraping Reviews for Introduction to Computer Science I taught by David A Smallberg....
All done! 

Good words:
GOOD 7
EASY 3
GREAT 3
FUNNY 3
------------------
TOTAL: 14

Bad words:
DIFFICULT 4
BORING 3
HARD 2
BAD 2
------------------
TOTAL: 11

David A Smallberg is a GOOD professor for Introduction to Computer Science I.
```
