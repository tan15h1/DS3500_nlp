'''
filename: hw7_codeblue_app.py
'''

'''
script sources
house: https://clinic-duty.livejournal.com/385.html
greys anatomy: https://script-pdf.s3.amazonaws.com/greys-anatomy-pilot-1-tv-script-pdf.pdf
the resident: https://scripts.tv-calling.com/script/fox-the-resident-1x01-pilot/
the good doctor: https://scripts.tv-calling.com/script/abc-good-doctor-1x01-pilot/
the mindy project: https://scripts.tv-calling.com/script/fox-the-mindy-project-1x01/
scrubs: https://www.scriptslug.com/script/scrubs-101-my-first-day-2001
'''
from hw7_codeblue import CodeBlue
import codeblue_parsers as cdp
from pprint import pprint  # Importing pprint for pretty printing
from pdf_to_txt import pdf_convert

def main():
    cb = CodeBlue()

    # convert pdfs to txt files
    pdf_convert('greys_anatomy.pdf', 'greys_anatomy.txt')
    pdf_convert('good_doctor.pdf', 'good_doctor.txt')
    pdf_convert('resident.pdf', 'resident.txt')
    pdf_convert('scrubs.pdf', 'scrubs.txt')
    pdf_convert('mindy_project.pdf', 'mindy_project.txt')

    #load stop words
    stopwords = cb.load_stop_words('stop_words.txt')

    # load text data for each script
    cb.load_text('greys_anatomy.txt', label='greys_anatomy',
                 parser=lambda filename: cdp.script_parser(filename, greys=True, stopwords=stopwords))
    cb.load_text('good_doctor.txt', label='good_doctor', parser=cdp.script_parser, stopwords=stopwords)
    cb.load_text('scrubs.txt', label='scrubs',
                 parser=lambda filename: cdp.script_parser(filename, scrubs=True, stopwords=stopwords))
    cb.load_text('house.txt', label='house', parser=cdp.house_parser, stopwords=stopwords)
    cb.load_text('mindy_project.txt', label='mindy_project',
                 parser=lambda filename: cdp.script_parser(filename, mindy=True, stopwords=stopwords))
    cb.load_text('resident.txt', label='resident',
                 parser=lambda filename: cdp.script_parser(filename, resident=True, stopwords=stopwords))

    # create Sankey diagram
    cb.wordcount_sankey(k=10)

    # analyze and plot sentiment analysis
    cb.sentiment_analysis()
    cb.plot_sentiments()

    # create cosine similarity
    cosine_similarities, labels = cb.cosine_similarity()
    cb.cosine_heatmap(cosine_similarities, labels)

    # print statements
    #pprint(cb.data['greys_anatomy'])
    #pprint(cb.data['good_doctor'])
    #pprint(cb.data['scrubs'])
    #pprint(cb.data['house'])
    #pprint(cb.data['mindy_project'])
    #pprint(cb.data['resident'])

if __name__ == '__main__':
    main()

