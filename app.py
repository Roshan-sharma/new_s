import streamlit as st
import spacy
from textblob import TextBlob
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sunny(d):
	parser=PlaintextParser.from_string(d,tokenizer("english"))
	l_s=LexRankSummarizer()
	summary=l_s(parser.document,3)
	s_l=[str(s) for s in summary]
	result=''.join(s_l)
	return result
def text_analyzer(new_text):
	nlp=spacy.load("en_core_web_sm")
	document=nlp(new_text)
	tokens=[token.text for token in document]
	data=[('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_)) for token in document]
	return data
def entity_analyzer(new_text):
	nlp=spacy.load("en_core_web_sm")
	document=nlp(new_text)
	tokens=[token.text for token in document]
	entities=[(entity.text,entity.label_) for entity in document.ents]
	data=[('"Tokens":{},\n"Entities":{}'.format(tokens,entities))]
	return data
def main():
	st.title("Streamlit NLP")
	st.subheader("NLP")
	if st.checkbox("Token and lemma"):
		message=st.text_area("Message","Enter message")
		if st.button("Analyze"):
			result=text_analyzer(message)
			st.json(result)
	if st.checkbox("Name entity"):
		message=st.text_area("Message","Enter message")
		if st.button("Extract"):
			result=entity_analyzer(message)
			st.json(result)
	if st.checkbox("Sentiment"):
		message=st.text_area("Message","Enter message")
		if st.button("Analyze"):
			blob=TextBlob(message)
			result=blob.sentiment
			st.success(result)
	if st.checkbox("Summarize"):
		message=st.text_area("Message","Enter message")
		selected_option=st.selectbox("Choose",("gensim","sumy"))
		if st.button("Analyze"):
			if selected_option=="gensim":
				result=summarize(message)
			elif selected_option=="sumy":
				result=sunny(message)
			st.success(result)



if __name__=="__main__":
	main()