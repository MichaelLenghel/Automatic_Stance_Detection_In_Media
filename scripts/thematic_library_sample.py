import spacy
import speech_to_text_speech_recognition as sr

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

recording_name = 'harvard.wav'

# text = sr.transcribe_recording(recording_name)

text = 'the stale smell of old beer lingers it takes heat to bring'\
            'out the odour a cold dip restores health exist a salt pickle'\
            'taste fine with him because of pasta are my favourite exist for'\
            'food is the hot cross bun'

doc = nlp(text)

# Perform analysis on the text
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)