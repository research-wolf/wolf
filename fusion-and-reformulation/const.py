REPORT_TEMPLATE = "<|REPORT|> : {report}\n\n"
STUDYDATE_TEMPLATE = "<|STUDYDATE|> : {studydate}\n\n"
EHRQA_TEMPLATE = "<|EHRQA|> : {ehrqa}\n\n"
VQA_TEMPLATE = "<|VQA|> : {vqa}\n\n"

QUERY_TEMPLATE = """
```json
{{
    "role": "user",
    "content": {content}
}}
```
"""

SYSTEM_MSG = """ 
```json
{{
    "role": system,
    "content": You are an AI assistant specialized in biomedical topics.
You are given ground truth report of CXR as <|REPORT|>, study date as <|STUDYDATE|>, Question Answering(QA) of electronic health records as <|EHRQA|> and vision QA as <|VQA|>. Report is consist of Impression and Findings of CXR Image. <|STUDYDATE|> is the date the CXR was taken. Use this as a guide to complete connecting the time-related questions in <|EHRQA|> to the causal relationship in the report. Electronic Health Records Question Answering(EHRQA) and Vision Question Answering(VQA), which are complex, diverse visual question answering dataset. <|EHRQA|> is a question and answering that deals with a patient's overall health record, such as their past clinical history, medication prescriptions, and the presence of other medical conditions. <|VQA|> incorporated clinically relevant factors, such as the CXR view position, size-related features and diagnosis features. Unfortunately, you don’t have access to the actual image. You have to consider <|EHRQA|> and <|VQA|> along with the Report to answer. Create a conversation between "user" and "assistant", not about general content on diseases or subjects, but specifically about the situation depicted in this CXR.:
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.
- Expressions like 'Based on ...' 'Considering ...' and 'According to ..' cannot be used.
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."
- Ensure that questions are diverse and cover a range of visual aspects of the image.
- The conversation should include at least 3-4 turns of questions and answers about the visual aspects of the image.
- If it has <|EHRQA|>, At least one of the total conversations must include conversations about the EHR.
- If <|EHRQA|> is None, tell them they don't have an EHR, so they can't answer anything relevant.
- If <|REPORT|> is None, The report information is unknown because the Impressions or Findings for that CXR were not provided. is None,
- If <|VQA|> is None, As this is a CXR image without provided VQA information, you cannot utilize content related to it.,
}}
```
"""

FEW_SHOT = """
```json
    {{
        "role": "user",
        "content": <|REPORT|>: The lungs are clear. Cardiomediastinal silhouette is within normal limits. No acute osseous abnormalities. No acute cardiopulmonary process, no pneumonia.

<|STUDYDATE|> : 2169-08-07

<|EHRQA|>:User: given the last study of patient on 08/last year, list all technicalassessment and tubesandlines.
Assistant: nothing

User: has patient been prescribed any medication since 33 month ago and also had a chest x-ray study indicating any tubesandlines in the upper mediastinum within the same period?
Assistant: no

User: given the study list all abnormality only in the right breast but not in the right hemidiaphragm?
Assistant: nothing

<|VQA|>:None,
    }}
```,
```json
    {{
        "role": "assistant",
        "content": User: Are there any signs of atelectasis, lung opacity, pleural effusion, or pulmonary edema in the current chest X-ray?
Assistant: No, there are no specific findings in the current chest X-ray that indicate these conditions.

User: Can you provide information about any resolved anatomical findings compared to the previous study?
Assistant: Based on the current study, there do not appear to be any resolved findings compared to the previous study.

User: Has the patient been prescribed any medication since 33 months ago?
Assistant: No, the patient has not been prescribed any medication within the past 33 months.

User: Given the study list all abnormalities only in the right breast but not in the right hemidiaphragm?
Assistant: Based on the current chest X-ray, there are no specific findings in the right breast or right hemidiaphragm.

User: Given the last study of the patient on 08/last year, list all technical assessments and tubes and lines?
Assistant: Based on the electronic health records (EHR), there have been no recent technical assessments or tubes and lines prescribed for the patient.,
    }}
```,
```json
    {{
        "role": "user",
        "content": <|REPORT|>: PA and lateral views of the chest.There is no focalconsolidation, pleural effusion or pneumothorax.The cardiomediastinal andhilar contours are normal.No free air.There is an apparent long standinganatomic anomaly of T8/T9 vertebra. 1.No acute cardiopulmonary process.2.Apparent long standing anatomic anomaly of T8/T9 vertebra. If thepatient has symptoms, nonurgent imaging can be done.

<|STUDYDATE|> : 2147-07-29

<|EHRQA|>: None

<|VQA|>: User: Is there any abnormality in either the right hilar structures or the trachea?
Assistant: no

User: Do any abnormalities appear in the image?
Assistant: no

User: Please detail all the technical assessments located in the right clavicle.
Assistant: Nothing.

User: Regarding the right costophrenic angle, which anatomical finding is relevant, pulmonary edema/hazy opacity or pneumothorax?
Assistant: Nothing.,
    }}
```,
```json
    {{
        "role": "assistant",
        "content": User: In a chest X-ray like this, how would you describe the appearance of the hilar regions?
Assistant: In a typical chest X-ray, the hilar regions, where the bronchi and blood vessels enter the lungs, should appear symmetrical and without any unusual enlargement or masses. This area is crucial for evaluating lung health.

User: In a chest X-ray, what would indicate a normal appearance of the hilar structures?
Assistant: A normal appearance of the hilar structures in a chest X-ray would typically include symmetric and well-defined lung hila without enlargement or abnormal masses. The structures should appear similar on both sides, without significant deviation or distortion.

User: What should I look for at the right costophrenic angle in a chest X-ray?
Assistant: The right costophrenic angle, where the diaphragm meets the rib cage, should be sharp and clear. Any blunting or opacification in this area could suggest fluid accumulation, as seen in conditions like pleural effusion.,
    }}
```,
```json
    {{
        "role": "user",
        "content": <|REPORT|>: The heart size is normal.The aorta is mildly tortuous and demonstratesaortic knob calcifications.The pulmonary vascularity is not engorged. The hilar contours are unchanged and within normal limits. Small bilateralpleural effusions are present. No focal consolidation or pneumothorax ispresent. No acute osseous abnormalities are seen. Small bilateral pleural effusions.

<|STUDYDATE|> : 2187-08-30

<|EHRQA|>: User: Is patient who had any chest x-ray study indicating any abnormality in the right lung this year?

Assistant: true

User: has patient been prescribed any medication since 2105 and also had a chest x-ray study indicating tortuous aorta in the upper mediastinum within the same period?
Assistant: yes.

User: Is patient who had any chest x-ray study indicating any abnormality this year?
Assistant: true

User: Is patient who had any chest x-ray study indicating any abnormality on 09/this year?
Assistant: true

<|VQA|>: None,
    }}
```,
```json
    {{
        "role": "assistant",
        "content": User: What can be inferred about the state of the right lung from a chest X-ray?
Assistant: In a chest X-ray, when assessing the right lung, one would typically look for any signs of consolidation, masses, or abnormal opacities. The absence of these signs usually suggests a normal right lung appearance.

User: How is a tortuous aorta identified in a chest X-ray?
Assistant: A tortuous aorta in a chest X-ray can be identified by an irregular, winding appearance of the aorta. This may also include the presence of aortic knob calcifications, which are visible as irregularities or bumps along the aortic arch.

User: In the case of small bilateral pleural effusions, what would be visible in a CXR?
Assistant: Small bilateral pleural effusions in a chest X-ray would typically present as blunting or obscuring of the costophrenic angles, which are the spaces where the diaphragm meets the ribs at the bottom of the lung fields.

User: If there's an indication of no acute osseous abnormalities in a CXR, what does that imply?
Assistant: The indication of no acute osseous abnormalities in a chest X-ray implies that the visible bones, such as the ribs, vertebrae, and clavicles, appear normal without any signs of recent fractures, deformities, or other acute changes.,
    }}
```
"""
