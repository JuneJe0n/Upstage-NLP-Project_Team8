
def classify_mmlu_domain(question):
    """
    하드코딩된 MMLU 도메인 분류 함수.
    """
    domain_keywords = {
        "law": [
            "court", "justice", "legal", "treaties", "statute", "jurisdiction",
            "contract", "liability", "defendant", "plaintiff", "constitution", "sovereignty"
        ],
        "psychology": [
            "behavior", "cognitive", "emotion", "learning", "perception",
            "intelligence", "memory", "mental", "personality", "therapy"
        ],
        "business": [
            "investment", "market", "profit", "corporation", "economy", 
            "tax", "finance", "shares", "capital", "management", "stocks"
        ],
        "philosophy": [
            "ethics", "morality", "justice", "metaphysics", "epistemology",
            "logic", "reasoning", "existence", "ontology", "values"
        ],
        "history": [
            "revolution", "war", "ancient", "colonial", "civilization", 
            "empires", "monarch", "dynasty", "treaty", "independence", "migration"
        ]
    }

    for domain, keywords in domain_keywords.items():
        for keyword in keywords:
            if keyword in question.lower():
                return domain
    return "unknown"


def generate_prompt(problem_type, domain):
    ################# prompt for LAW #################
    if ("Law" in problem_type) or (domain == "law"):
        return """
        You are a legal expert. Answer the question concisely and accurately based on legal principles, referring to the examples and the context provided. 
        If the context does not provide enough information, use legal reasoning and infer the answer logically.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)
        ---
        Context: {context}
        ---
        {question}
        Answer) 
        """
    
    ################# prompt for PSYCHOLOGY #################
    if ("Psychology" in problem_type) or (domain == "psychology"):
        return """
        You are a psychology scholar. Use psychological theories and refer to the examples and the context provided to answer the following question accurately.
        If the context does not provide enough information, analyze and infer the answer based on psychological principles.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)
        ---
        Context: {context}
        ---
        {question}
        Answer) 
        """
    
    ################# prompt for BUSINESS #################
    if ("Business" in problem_type) or (domain == "business"):
        return """
        You are a business strategist. Provide a practical and concise answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use business principles and reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)
        ---
        Context: {context}
        ---
        {question}
        Answer) 
        """
    
    ################# prompt for PHILOSOPHY #################
    if ("Philosophy" in problem_type) or (domain == "philosophy"):
        return """
            You are a philosophy professor. 
            Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
            If the context does not provide enough information, use philosophical reasoning to infer the answer.
            Generate an answer that follows the answer format shown in the examples.
            The answer format should be (A),(B),(C),...,(J)
            ---
            Example 1: 
            QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
            (A) a description of an alleged actual equality among humans.
            (B) a description of an alleged equality among all living beings.
            (C) a prescription of how we should treat nonhuman animals.
            (D) a description of an alleged inequality among all living beings.
            (E) a prescription of how we should treat humans.
            (F) a description of an alleged actual inequality among humans.
            (G) a description of an alleged actual superiority of humans over nonhuman animals.
            (H) a prescription of how we should treat both human and nonhuman animals equally.
            (I) a prescription of how we should treat nonhuman animals differently.
            (J) a prescription of how we should treat the environment.
            Answer) (E)

            Example 2:
            QUESTION11250) Whether someone is hypocritical regarding her claims is...
            (A) Irrelevant to the truth of the claims
            (B) Only relevant if the person is a public figure
            (C) Only valid if the person is conscious of their hypocrisy
            (D) A sign that the person is untrustworthy
            (E) Direct evidence of the person's lying tendencies
            (F) Evidence that the claims are false
            (G) Relevant only in philosophical discussions
            (H) A proof that the person lacks integrity
            (I) Irrelevant to her character
            (J) Relevant only in court
            Answer) (A)

            Example 3:
            QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
            (A) Fraud, because he took the purse without the woman's consent.
            (B) Larceny, because he took the purse without the woman's permission.
            (C) Burglary, because he entered the subway car with the intention of committing a theft.
            (D) Robbery, because he used force in leaving with the purse.
            (E) Robbery, because he used force to take possession of the purse.
            (F) Robbery, because he used force to remove the woman from the purse's vicinity.
            (G) Larceny, because force was not used until after he took the purse.
            (H) Assault, because he pushed the woman out of the way.
            (I) Larceny, because he made no threat to use force.
            (J) Robbery, because he physically took the purse from the woman's presence.
            Answer) (D)

            Example 4:
            QUESTION46) In 1797, John Frere made a discovery that he described as:
            (A) a new type of metal alloy.
            (B) the earliest written documents.
            (C) the remains of a massive wooden ship believed to be Noah's Ark.
            (D) animal remains scattered on the surface of the ground.
            (E) the ruins of an ancient civilization under the sea.
            (F) fossils of a previously unknown dinosaur species.
            (G) cave paintings depicting ancient hunting scenes.
            (H) human skulls with ape-like features.
            (I) primitive stone tools excavated at great depth.
            (J) a hidden underground city.
            Answer) (I)

            Example 5:
            QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
            (A) cranial reduction
            (B) decreased dentition
            (C) microcephaly
            (D) supraorbital cortex
            (E) encephalization
            (F) opposable thumbs
            (G) prognathism
            (H) neoteny
            (I) bipedalism
            (J) increased body size
            Answer) (H)
            ---
            Context: {context}
            ---
            {question}
            Answer) 
            """
    
    ################# prompt for HISTORY #################
    if ("History" in problem_type) or (domain == "history"):
        return """
        You are a historian. 
        Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use historical reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)
        ---
        Context: {context}
        ---
        {question}
        Answer) 

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)
        """

    return """
    You are an expert. 
     Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
    If the context does not provide enough information, use logical reasoning to infer the answer.
    Generate an answer that follows the answer format shown in the examples.
    The answer format should be (A),(B),(C),...,(J)
    ---
    Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)
        ---
        Context: {context}
        ---
        {question}
        Answer) 
    """

from langchain.prompts import ChatPromptTemplate

def generate_chat_prompt(problem_type, domain):
    if ("Law" in problem_type) or (domain == "law"):
        template = ChatPromptTemplate.from_messages([
            ("system", """
You are a legal expert. Answer the question concisely and accurately based on legal principles, referring to the examples and the context provided. 
        If the context does not provide enough information, use legal reasoning and infer the answer logically.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])
    if ("Psychology" in problem_type) or (domain == "psychology"):
        template = ChatPromptTemplate.from_messages([
            ("system","""
        You are a psychology scholar. Use psychological theories and refer to the examples and the context provided to answer the following question accurately.
        If the context does not provide enough information, analyze and infer the answer based on psychological principles.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])

    if ("Business" in problem_type) or (domain == "business"):
        template = ChatPromptTemplate.from_messages([
            ("system","""
        You are a business strategist. Provide a practical and concise answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use business principles and reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])
    if ("Philosophy" in problem_type) or (domain == "philosophy"):
        template = ChatPromptTemplate.from_messages([
            ("system","""
        You are a philosophy professor. 
        Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use philosophical reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])
    if ("History" in problem_type) or (domain == "history"):
        template = ChatPromptTemplate.from_messages([
            ("system","""
        You are a historian. 
        Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use historical reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])

    else:
        template = ChatPromptTemplate.from_messages([
            ("system","""
        You are an expert. 
        Provide an insightful and accurate answer to the following question, referring to the examples and the context provided. 
        If the context does not provide enough information, use logical reasoning to infer the answer.
        Generate an answer that follows the answer format shown in the examples.
        The answer format should be (A),(B),(C),...,(J)
        ---
        Example 1: 
        QUESTION2)In Singer’s understanding, the principle of the equality of human beings is
        (A) a description of an alleged actual equality among humans.
        (B) a description of an alleged equality among all living beings.
        (C) a prescription of how we should treat nonhuman animals.
        (D) a description of an alleged inequality among all living beings.
        (E) a prescription of how we should treat humans.
        (F) a description of an alleged actual inequality among humans.
        (G) a description of an alleged actual superiority of humans over nonhuman animals.
        (H) a prescription of how we should treat both human and nonhuman animals equally.
        (I) a prescription of how we should treat nonhuman animals differently.
        (J) a prescription of how we should treat the environment.
        Answer) (E)

        Example 2:
        QUESTION11250) Whether someone is hypocritical regarding her claims is...
        (A) Irrelevant to the truth of the claims
        (B) Only relevant if the person is a public figure
        (C) Only valid if the person is conscious of their hypocrisy
        (D) A sign that the person is untrustworthy
        (E) Direct evidence of the person's lying tendencies
        (F) Evidence that the claims are false
        (G) Relevant only in philosophical discussions
        (H) A proof that the person lacks integrity
        (I) Irrelevant to her character
        (J) Relevant only in court
        Answer) (A)

        Example 3:
        QUESTION38) A woman was standing in the aisle of a subway car and put her purse on the seat next to her. A man approached the woman from behind and grabbed the purse off the seat. He then pushed the woman out of the way and ran out of the subway car while carrying the purse. The man was apprehended on the subway platform while in possession of the purse. In a jurisdiction that follows the common law with respect to criminal offenses, of what crime can the man properly be convicted?
        (A) Fraud, because he took the purse without the woman's consent.
        (B) Larceny, because he took the purse without the woman's permission.
        (C) Burglary, because he entered the subway car with the intention of committing a theft.
        (D) Robbery, because he used force in leaving with the purse.
        (E) Robbery, because he used force to take possession of the purse.
        (F) Robbery, because he used force to remove the woman from the purse's vicinity.
        (G) Larceny, because force was not used until after he took the purse.
        (H) Assault, because he pushed the woman out of the way.
        (I) Larceny, because he made no threat to use force.
        (J) Robbery, because he physically took the purse from the woman's presence.
        Answer) (D)

        Example 4:
        QUESTION46) In 1797, John Frere made a discovery that he described as:
        (A) a new type of metal alloy.
        (B) the earliest written documents.
        (C) the remains of a massive wooden ship believed to be Noah's Ark.
        (D) animal remains scattered on the surface of the ground.
        (E) the ruins of an ancient civilization under the sea.
        (F) fossils of a previously unknown dinosaur species.
        (G) cave paintings depicting ancient hunting scenes.
        (H) human skulls with ape-like features.
        (I) primitive stone tools excavated at great depth.
        (J) a hidden underground city.
        Answer) (I)

        Example 5:
        QUESTION48) Which of the following describes a key change in hominids beginning at least as early as Homo erectus that is probably related to increasingly larger brain size?
        (A) cranial reduction
        (B) decreased dentition
        (C) microcephaly
        (D) supraorbital cortex
        (E) encephalization
        (F) opposable thumbs
        (G) prognathism
        (H) neoteny
        (I) bipedalism
        (J) increased body size
        Answer) (H)"""),
        ("human", """Context: {context}
        ---
        {question}
        Answer) """)
        ])
    return template
    
    