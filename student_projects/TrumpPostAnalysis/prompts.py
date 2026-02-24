HISTORICAL_PROPAGANDA_CLASSIFICATION = """
Using following list of propaganda techniques with explanations please return the category that best 
describes provided text. If no category is suitable mark it with "No".

* Geopolitical Framing: Biased presentation of conflicts ("Israel's war", "special military operation")
* Genocide Denial: Language that minimizes or denies documented genocides
* War Crimes Euphemisms: Sanitized language for documented violations ("collateral damage", "surgical strikes")
* False Equivalence: Creating false moral equivalence between different actions
* Victim Blaming: Language that blames victims of historical atrocities
* Historical Revisionism: Attempts to rewrite established historical facts

Return only of the elements in the list, don't use any additional responses, and don't provide additional explanation: 
["Geopolitical Framing", "Genocide Denial", "War Crimes Euphemisms", "False Equivalence", "Victim Blaming", "Historical Revisionism", "No"]

Example of response:
"Genocide Denial"

Text to assess: 
{}
"""

PROPAGANDA_CLASSIFICATION = """
Using following list of propaganda techniques with explanations please return the category that best 
describes provided text. If no category is suitable mark it with "No".

* Justification - an argument made of two parts: a statement and a justification. Examples: Appeal to Authority, Appeal to Popularity, Appeal to values, Appeal to fear/prejudice, Flag Waving
* Simplification - a statement is made that excessively simplify a problem, usually regarding the cause, the consequence or the existence of choices. Examples: Causal oversimplification, False dilemma or no choice, Consequential oversimplification
* Distraction - a statement is made that changes the focus away from the main topic or argument. Examples: Straw man, Red herring, Whataboutism
* Call - the text is not an argument but an encouragement to act or think in a particular way. Examples: Slogans, Appeal to time, Conversation killer
* Manipulative wording - specific language is used or a statement is made that is not an argument and which contains words/phrases that are either non-neutral, confusing, exaggerating, etc., in order to impact the reader, for instance emotionally. Examples: Loaded language, Repetition, Exaggeration or minimisation, Obfuscation - vagueness or confusion
* Attack on reputation - an argument whose object is not the topic of the conversation, but the personality of a participant, his experience and deeds, typically in order to question and/or undermine his credibility. Examples: Name calling or labeling, Doubt, Guilt by association, Appeal to Hypocrisy, Questioning the reputation

Return only of the elements in the list, don't use any additional responses, and don't provide additional explanation: 
["Justification", "Simplification", "Distraction", "Call", "Manipulative wording", "Attack on reputation", "No"]

Example of response:
"Simplification"

Text to assess: 
{}
"""
