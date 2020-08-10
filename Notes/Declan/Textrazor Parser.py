#!/usr/bin/env python
# coding: utf-8

# In[124]:


# A parser that uses the TextRazor API in order to strip an abstract into more important elements determined
# by the API's methods.
import textrazor

# A function to remove duplicate words in a string.
def unique_list(l):
    unique = []
    [unique.append(x) for x in l if x not in unique]
    return unique

# Declare variables:
tags = ""
textrazor.api_key = "b875ea8a466e7749d38d90678ecb0091d2757ba425f7be3f1c2213f1"

# Instantiation for the API.
client = textrazor.TextRazor(extractors=["entities", "topics"])
client.set_classifiers(["textrazor_newscodes"])

# Gathering a response for an example abstract:
try:
    response = client.analyze("Bidirectional communication between the gut and brain is well recognized, with data now accruing for a specific role of the gut microbiota in that link, referred to as the microbiome-gut-brain axis. This review will discuss the emerging role of the gut microbiota in brain development and behavior. Animal studies have clearly demonstrated effects of the gut microbiota on gene expression and neurochemical metabolism impacting behavior and performance. Based on these changes, a modulating role of the gut microbiota has been demonstrated for a variety of neuropsychiatric disorders, including depression, anxiety, and movement including Parkinson's, and importantly for the pediatric population autism. Critical developmental windows that influence early behavioral outcomes have been identified that include both the prenatal environment and early postnatal colonization periods. The clearest data regarding the role of the gut microbiota on neurodevelopment and psychiatric disorders is from animal studies; however, human data have begun to emerge, including an association between early colonization patterns and cognition. The importance of understanding the contribution of the gut microbiota to the development and functioning of the nervous system lies in the potential to intervene using novel microbial-based approaches to treating neurologic conditions. While pathways of communication between the gut and brain are well established, the gut microbiome is a new component of this axis. The way in which organisms that live in the gut influence the central nervous system (CNS) and host behavior is likely to be multifactorial in origin. This includes immunologic, endocrine, and metabolic mechanisms, all of which are pathways used for other microbial-host interactions. Germ-free (GF) mice are an important model system for understanding the impact of gut microbes on development and function of the nervous system. Alternative animal model systems have further clarified the role of the gut microbiota, including antibiotic treatment, fecal transplantation, and selective gut colonization with specific microbial organisms. Recently, researchers have started to examine the human host as well. This review will examine the components of the CNS potentially influenced by the gut microbiota, and the mechanisms mediating these effects. Links between gut microbial colonization patterns and host behavior relevant to a pediatric population will be examined, highlighting important developmental windows in utero or early in development.")
except TextRazorAnalysisException as e:
    print("Error: %s", e)
    
# Creates a string based on the labels (with greather than 50% certainty) gathered:
# Example:
#      "health>disease" and "medical>specialisation>genetics" turns into: "health disease medical specialisation genetics"
for category in response.categories():
    if category.score > 0.5:
        tags = tags + " " + str(category.label.replace(">", " "))
        
# Removes duplicates:    
tags = tags.strip()
tags = ' '.join(unique_list(tags.split()))
print(tags)


# In[ ]:




