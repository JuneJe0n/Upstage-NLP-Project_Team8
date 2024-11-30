
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
        ---
        Context: {context}
        ---
        {question}
        Answer) 
    """
