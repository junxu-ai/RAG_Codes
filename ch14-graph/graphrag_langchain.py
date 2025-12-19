from langchain.indexes import GraphIndexCreator
from langchain_openai import ChatOpenAI
# from langchain_community.graphs.index_creator import GraphIndexCreator
from langchain.chains import GraphQAChain
import os

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",api_key=os.getenv("OPENAI_API_KEY"))
index_creator = GraphIndexCreator(llm=llm)

text = """
Harry Potter is an orphaned wizard raised by Muggles who discovers, at age eleven, that he is famous in the magical world for surviving Lord Voldemort’s attack that killed his parents, James and Lily Potter. At Hogwarts School of Witchcraft and Wizardry he forms an inseparable trio with Ron Weasley—sixth son in a warm-hearted wizarding family—and Hermione Granger, a brilliant Muggle-born witch. Under headmaster Albus Dumbledore’s mentorship, the trio repeatedly uncovers fragments of Voldemort’s plan to regain power.

Voldemort, born Tom Riddle, is linked to Harry by prophecy and by the piece of his own soul embedded in Harry’s scar. His chief followers, the Death Eaters, include the fanatical Bellatrix Lestrange and the conflicted Severus Snape. Snape, outwardly Voldemort’s ally, is secretly loyal to Dumbledore because of his unrequited love for Lily Potter; his double role shapes the war’s outcome.

Harry’s godfather Sirius Black and former teacher Remus Lupin, both members of the Order of the Phoenix, become surrogate family figures, while gamekeeper Rubeus Hagrid acts as guardian and friend from Harry’s first day. Ron’s sister Ginny evolves from schoolgirl admirer to Harry’s partner, tying Harry permanently to the Weasley clan. Rivalry with Draco Malfoy, heir to a pure-blood supremacist line, mirrors the wider ideological divide in wizarding society.

The story culminates in a hunt for Voldemort’s Horcruxes—objects anchoring his fragmented soul—leading to the Battle of Hogwarts. When the final Horcrux inside himself is destroyed, Harry willingly sacrifices, is briefly “killed,” and returns to defeat Voldemort, freeing the wizarding world. Nineteen years later, the next generation boards the Hogwarts Express, symbolizing hard-won peace and the enduring power of chosen bonds over bloodline dogma.
"""

graph = index_creator.from_text(text)

chain = GraphQAChain.from_llm(llm, graph=graph, verbose=True)
chain.run("What is the relationship between Harry Potter and Voldemort") 
result = index.query(query)
print(result)
