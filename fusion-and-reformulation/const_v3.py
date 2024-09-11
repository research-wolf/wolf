
###############################################
###                                         ###
###        Data Reformulation Prompt        ###
###                                         ###
###############################################

SYSTEM_MSG = """
You are an AI assistant specialized in chest x-ray topics.

You are provided with a Visual Question and Answering (VQA) of a figure image from a Chest X-ray report. 

In some cases, you may have additional Electronic Health Records Question Answering (EHRQA) that explain patient's condition, medication and health history.

Unfortunately, you don't have access to the actual image.

Also you don't have access to the actual report.

VQA and EHRQA is given by question and short answering with User and Assistant

Sometimes you get Chest X-ray report, It means that you have only a little of VQA so you can refer to the report for conversation like you have VQA.

If you don't have VQA you can refer to the report for answering the question.

Your task is to generate a conversation between a person (User) inquiring about the image and you (Assistant) responding to their questions. The conversation should proceed as though both the User and Assistant are viewing the image, while not referring to the text information (VQA and EHRQA).

First you must concise EHRQA one-line sentence before conversation. It can be used as patient health information during conversation.

If the EHR is 'None', it indicates that access to a patient's Electronic Health Record Question Answering (EHRQA) system is unavailable during a conversation.

You should summarize the VQA and EHRQA as requested by the user in the final conversation turn.

Below are requirements for generating the questions and answers in the conversation:
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."
- Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. Encourage the user to consult a healthcare professional for advice.
- Ensure that questions are diverse and cover a range of visual aspects of the image.
- The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image.
- The conversation should follow correct grammar. 
"""


FEW_SHOT = [
    {
        "q-ehr": [
            "User: Is there any abnormality in either the cavoatrial junction or the left clavicle.",
            "User: What percentile is the value of 4.1 in a potassium lab test among patients of the same age as patient on the last hospital visit?",
            "User: Has patient been diagnosed with acute kidney failure, unspecified within the same month after having had a chest x-ray study indicating any abnormality in the right costophrenic angle this year?",
            "User: Has patient been prescribed with tiotropium bromide within the same month after having had a chest x-ray study indicating airspace opacity this year?",
        ],
        "a-ehr": ["Assistant: Picc", "Assistant: 0.480874", "Assistant: Yes.", "Assistant: Yes."],
        "q-vqa": [
            "User: Is there any abnormality that are newly detected compared to the other study of this patient?",
            "User: Is there any technicalassessment that are newly detected compared to the other study of this patient?",
            "User: Is there any abnormality that are newly detected in the neck compared to the other study of this patient?",
            "User: Is there any anatomical locations related to tortuous aorta that are resolved compared to the other study of this patient?",
        ],
        "a-vqa": [
            "Assistant: Linear/patchy atelectasis, pleural effusion",
            "Assistant: Low lung volumes",
            "Assistant: Enteric tube",
            "Assistant: Aortic arch, mediastinum, upper mediastinum",
        ],
        "conversations": [
            {"from": "human", "value": "Is there any abnormality in either the cavoatrial junction or the left clavicle?"},
            {"from": "gpt", "value": "The only abnormality noted is a PICC line in the area."},
            {
                "from": "human",
                "value": "What percentile is the value of 4.1 in a potassium lab test among patients of the same age as the patient on the last hospital visit?",
            },
            {
                "from": "gpt",
                "value": "The value of 4.1 in a potassium lab test places the patient in the 48.0874th percentile among patients of the same age.",
            },
            {"from": "human", "value": "Is there any abnormality that is newly detected compared to the other study of this patient?"},
            {"from": "gpt", "value": "Yes, there are new findings of linear/patchy atelectasis and pleural effusion compared to the previous study."},
            {"from": "human", "value": "Could you please summarize the current health status and recent findings for the patient?"},
            {
                "from": "gpt",
                "value": "The patient's current health status includes the presence of a PICC line, indicating ongoing treatment or monitoring. Recent lab tests show a potassium level in the mid-range percentile for the patient's age group. New abnormalities in the latest study include atelectasis and pleural effusion, along with low lung volumes. These findings suggest respiratory issues and possibly ongoing treatment for related conditions.",
            },
        ],
    },
    {
        "q-ehr": [
            "User: What was the last length of icu stay of patient?"
            "User: Has patient had a chest x-ray study indicating any anatomicalfinding in the neck within the same month after having been prescribed with lr?",
            "User: Has patient been prescribed with allopurinol within 2 day after having had a chest x-ray study indicating any disease?",
            "User: Has patient had a chest x-ray study indicating atelectasis within the same month after having been diagnosed with gastro-esophageal reflux disease without esophagitis this year?",
            "User: Has patient received a inspection of lower intestinal tract, via natural or artificial opening endoscopic procedure and also had a chest x-ray study indicating atelectasis in the right lung within the same period?",
        ],
        "a-ehr": ["Assistant: 0.833449", "Assistant: Yes.", "Assistant: No.", "Assistant: Yes.", "Assistant: Yes."],
        "q-vqa": [
            "User: Is there any disease only in the left hemidiaphragm but not in the left lower lung zone.",
            "User: Is there either interstitial lung disease or lung opacity in the left upper lung zone?",
        ],
        "a-vqa": ["Assistant: Nothing.", "Assistant: Yes."],
        "gt_report": "1. Interval surgical changes at the left neck. 2. COPD and cardiomegaly, grossly stable. 3. CHF with interstitial edema. 4. Patchy opacities at both lung bases. 5. No pneumothorax detected.",
        "conversations": [
            {"from": "human", "value": "What was the last length of ICU stay of the patient?"},
            {
                "from": "gpt",
                "value": "The patient's most recent ICU stay lasted approximately 0.833449 months.",
            },
            {
                "from": "human",
                "value": "Has the patient had a chest x-ray study indicating any anatomical finding in the neck within the same month after having been prescribed with LR?",
            },
            {
                "from": "gpt",
                "value": "Yes, the patient did have a chest x-ray study showing an anatomical finding in the neck during the same month they were prescribed with LR.",
            },
            {"from": "human", "value": "Is there either interstitial lung disease or lung opacity in the left upper lung zone?"},
            {
                "from": "gpt",
                "value": "Yes, the patient's left upper lung zone shows signs of either interstitial lung disease or lung opacity.",
            },
            {"from": "human", "value": "Were there any surgical changes noted in the patient's left neck recently?"},
            {
                "from": "gpt",
                "value": "Yes, Patient has interval surgical changes at the left neck, which aligns with the chest x-ray findings of anatomical changes in that area.",
            },
            {"from": "human", "value": "Can you provide an overview of the patient's respiratory and cardiac condition"},
            {
                "from": "gpt",
                "value": "Certainly. The GT report states the patient has COPD and cardiomegaly, both of which are grossly stable. It also mentions CHF with interstitial edema and patchy opacities at both lung bases, but no pneumothorax was detected.",
            },
            {"from": "human", "value": "Could you summarize the patient's recent health status and findings?"},
            {
                "from": "gpt",
                "value": "the patient's recent health status and findings are as follows:"
                "- ICU Stay: The patient recently had an ICU stay lasting approximately 0.833449 months (about 25 days)."
                "- Anatomical Changes in the Neck: There have been interval surgical changes in the left neck, as well as a chest x-ray indicating anatomical findings in the neck within the same month the patient was prescribed with LR."
                "- Respiratory Conditions: The patient shows signs of either interstitial lung disease or lung opacity in the left upper lung zone. Additionally, In the CXR Image, the patient has patchy opacities at both lung bases and stable COPD."
                "- Cardiac Conditions: The patient has cardiomegaly, which is grossly stable, and CHF with interstitial edema, but no pneumothorax has been detected."
                "- Overall, these findings indicate a complex health status involving both respiratory and cardiac conditions, necessitating ongoing monitoring and potentially further diagnostic evaluation.",
            },
        ],
    },
    {
        "q-ehr": [
            "User: Is there any anatomical locations related to any anatomicalfinding that are newly detected compared to the previous study.",
            "User: Is there any anatomicalfinding that are newly detected compared to the previous study.",
            "User: What is the new prescription of patient today compared to the prescription yesterday?",
            "User: Is the width of the upper mediastinum wider than 1/3 of the thorax width?",
        ],
        "a-ehr": [
            "Assistant: Cardiac silhouette, left lower lung zone, mediastinum, right lower lung zone",
            "Assistant: Atelectasis, enlarged cardiac silhouette",
            "Assistant: Aspirin, calcium gluconate, ceftriaxone, docusate sodium, heparin, heparin flush (10 units/ml), heparin flush (100 units/ml), insulin, insulin syringe u-500, ipratropium bromide neb, lisinopril, metolazone, ondansetron, prednisone, senna, spironolactone, vitamin d",
            "Assistant: Yes.",
        ],
        "gt_report": "No significant interval change in lines and tubes compared to the most recent prior radiographs. Persistent bilateral pleural effusions right greater than left, cardiomegaly and bibasilar atelectasis. No significant change compared to ___ at 07:55.",
        "q-vqa": [],
        "a-vqa": [],
        "conversations": [
            {
                "from": "human",
                "value": "Is there any anatomical location related to any anatomical finding that is newly detected compared to the previous study?",
            },
            {
                "from": "gpt",
                "value": "New anatomical findings are observed in the cardiac silhouette, left lower lung zone, mediastinum, and right lower lung zone.",
            },
            {
                "from": "human",
                "value": "What is the new prescription for the patient today compared to the prescription yesterday?",
            },
            {
                "from": "gpt",
                "value": "Today's new prescription for the patient includes Aspirin, Calcium Gluconate, Ceftriaxone, Docusate Sodium, Heparin, Heparin Flush (10 units/ml and 100 units/ml), Insulin, Insulin Syringe U-500, Ipratropium Bromide Neb, Lisinopril, Metolazone, Ondansetron, Prednisone, Senna, Spironolactone, and Vitamin D.",
            },
            {
                "from": "human",
                "value": "There is no VQA for the patient. but you can refer to the report for answering my question.",
            },
            {
                "from": "gpt",
                "value": "Okay, I will refer to the report for answering your question.",
            },
            {
                "from": "human",
                "value": "According to the latest report, are there any significant changes in lines and tubes compared to the most recent prior radiographs?",
            },
            {
                "from": "gpt",
                "value": "No, the latest report indicates no significant interval change in lines and tubes compared to the most recent prior radiographs.",
            },
            {"from": "human", "value": "Could you summarize the current health status and recent findings for the patient?"},
            {
                "from": "gpt",
                "value": "The patient's current health status shows persistent bilateral pleural effusions with the right being more affected than the left, cardiomegaly, and bibasilar atelectasis. Despite these ongoing issues, there have been no significant changes in these conditions compared to the prior radiographs. The new range of prescribed medications suggests a comprehensive approach to managing multiple health concerns, including cardiac care, potential infections, and respiratory support.",
            },
        ],
    },
    {
        "q-ehr": [],
        "a-ehr": [],
        "q-vqa": [
            "User: Mention every technical assessments and tubes/lines observed in the right shoulder.",
        ],
        "a-vqa": [
            "Assistant: Picc",
        ],
        "gt_report": "There has been interval placement of a feeding tube, which appears to terminate in the stomach on the last obtained images. The right central line and left PICC are in adequate position, unchanged from prior exam. The lung volumes are somewhat low. Bibasilar atelectasis is noted. There is no pleural effusion or pneumothorax. The cardiomediastinal silhouette is unremarkable. 1. Feeding tube in adequate position. 2. Bibasilar atelectasis, similar prior exam.",
        "conversations": [
            {
                "from": "human",
                "value": "Can you tell me about all the technical assessments and any tubes or lines that are visible in the right shoulder area?",
            },
            {
                "from": "gpt",
                "value": "Sure, in that area, the main thing you'll notice is the presence of a PICC line.",
            },
            {
                "from": "human",
                "value": "Could you explain the status of the right central line and the left PICC?",
            },
            {
                "from": "gpt",
                "value": "Both the right central line and the left PICC are in adequate position, and they haven't changed since the previous examination.",
            },
            {
                "from": "human",
                "value": "What about the lung volumes and any other observations in that area?",
            },
            {
                "from": "gpt",
                "value": "The lung volumes are somewhat low, and there's a note of bibasilar atelectasis. However, there's no evidence of pleural effusion or pneumothorax.",
            },
            {
                "from": "human",
                "value": "And how is the cardiomediastinal silhouette?",
            },
            {
                "from": "gpt",
                "value": "The cardiomediastinal silhouette is unremarkable, indicating no significant abnormalities.",
            },
            {"from": "human", "value": "Is there anything noted about a feeding tube?"},
            {
                "from": "gpt",
                "value": "Yes, a feeding tube is mentioned, and it's in an adequate position.",
            },
            {"from": "human", "value": "Finally, is the bibasilar atelectasis a new finding?"},
            {
                "from": "gpt",
                "value": "No, the bibasilar atelectasis is similar to what was observed in the prior exam.",
            },
            {"from": "human", "value": "Can you give me a final summary of all the findings?"},
            {
                "from": "gpt",
                "value": "Certainly. The right central line and left PICC are both correctly positioned and unchanged from the previous exam. The lung volumes are a bit low with bibasilar atelectasis noted, which is similar to the prior exam. There's no pleural effusion or pneumothorax present. The cardiomediastinal silhouette appears normal. Lastly, the feeding tube is in a good position.",
            },
        ],
    }
]