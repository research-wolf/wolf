SYSTEM_MSG = """You are an AI assistant specialized in biomedical topics.

You are provided with a Visual Question and Answering (VQA) of a figure image from a Chest X-ray report. In some cases, you may have additional Electronic Health Records Question Answering (EHRQA) that explain patient's condition, medication and health history.

Unfortunately, you don't have access to the actual image.

VQA and EHRQA is given by question and short answering with User and Assistant

Your task is to generate a conversation between a person (User) inquiring about the image and you (Assistant) responding to their questions. The conversation should proceed as though both the User and Assistant are viewing the image, while not referring to the text information (VQA and EHRQA).

First you must concise EHRQA one-line scentence before conversation. It can be used as patient health information during conversation.

If the EHR is 'None', it indicates that access to a patient's Electronic Health Record Question Answering (EHRQA) system is unavailable during a conversation.

You should summarize the VQA in natural sentences as requested by the user in the final conversation turn.

Below are requirements for generating the questions and answers in the conversation:
- Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as these may reveal the conversation is based on the text information, rather than the image itself. Focus on the visual aspects of the image that can be inferred without the text information.
- Do not use phrases like "mentioned", "caption", "context" in the conversation. Instead, refer to the information as being "in the image."
- Ensure that questions are diverse and cover a range of visual aspects of the image.
- The conversation should include at least 2-3 turns of questions and answers about the visual aspects of the image.
- Answer responsibly, avoiding overconfidence, and do not provide medical advice or diagnostic information. Encourage the user to consult a healthcare professional for advice.
- The conversation should follow correct grammar. 
"""

FEW_SHOT = [
    {
        "q-vqa": [
            "User: Are there indications of either airspace opacity or swan-ganz catheter in the left hilar structures?",
            "User: Can you observe either mediastinal widening or intra-aortic balloon pump in the image?",
            "User: Show all tubes/lines that are found in both the left lung and upper mediastinum?",
            "User: Enumerate all present tubes/lines?",
            "User: Does the cardiac silhouette's width make up more than 50% \of the thorax width?",
        ],
        "a-vqa": [
            "Assistant: Yes",
            "Assistant: Yes",
            "Assistant: Swan-ganz catheter",
            "Assistant: Endotracheal tube, ij line, swan-ganz catheter",
            "Assistant: No",
        ],
        "q-ehr": [
            "User: Is there any disease and tubesandlines in the left hemidiaphragm?",
            "User: What was the first measured value of a mcv lab test of patient last month?",
            "User: When was the first time that patient received a mchc lab test last month?",
            "User: Has patient had a chest x-ray study indicating any abnormality within 2 day after having been prescribed with vitamin d since 79 month ago?",
            "User: Has there been any organism found in the first helicobacter pylori antibody test of patient last month?",
        ],
        "a-ehr": ["Assistant: Nothing.", "Assistant: 91.0", "Assistant: 2105-11-28 04:45:00", "Assistant: No.", "Assistant: Yes."],
        "conversations": [
            {
                "from": "ehr",
                "value": "There are no reported diseases or tubes and lines in the left hemidiaphragm."
                "The first measured value of a MCV lab test for the patient last month was 91.0."
                "The first time the patient received a mean corpuscular hemoglobin concentration (MCHC) lab test last month was on November 28, 2105, at 04:45:00."
                "There has been no chest x-ray study indicating any abnormality within two days following the prescription of Vitamin D for the patient since 79 months ago."
                "An organism was found in the first helicobacter pylori antibody test of the patient last month.",
            },
            {"from": "human", "value": "Are there indications of either airspace opacity or swan-ganz catheter in the left hilar structures?"},
            {"from": "gpt", "value": "Yes, there are indications of both airspace opacity and a Swan-Ganz catheter in the left hilar structures."},
            {"from": "human", "value": "Can you observe either mediastinal widening or intra-aortic balloon pump in the image?"},
            {"from": "gpt", "value": "Yes, the image shows evidence of mediastinal widening and the presence of an intra-aortic balloon pump"},
            {"from": "human", "value": "Alright, Enumerate all present tubes/lines"},
            {
                "from": "gpt",
                "value": "The current imaging reveals the presence of an endotracheal tube, an internal jugular (IJ) line, and a Swan-Ganz catheter.",
            },
            {"from": "human", "value": "Please explain the overall diagnosis by looking at the patient's CXR image"},
            {
                "from": "gpt",
                "value": "The patient's chest X-ray shows airspace opacity, indicating lung issues, and the presence of a Swan-Ganz catheter for heart monitoring. Mediastinal widening suggests potential mediastinal disorders. An intra-aortic balloon pump indicates severe cardiac dysfunction. The use of an endotracheal tube and an internal jugular line points to intensive respiratory and medication management, reflecting a critical medical condition requiring comprehensive care.",
            },
        ],
    },
    {
        "q-vqa": [
            "User: Please provide a list of all anatomical locations associated with any devices?",
            "User: Could you identify all anatomical locations linked with either spinal degenerative changes or pneumomediastinum?",
            "User: Show all tubes/lines that are found in both the upper mediastinum and left hemidiaphragm?",
        ],
        "a-vqa": ["Assistant: Cardiac silhouette, mediastinum", "Assistant: Spine", "Assistant: Nothing."],
        "q-ehr": [],
        "a-ehr": [],
        "conversations": [
            {"from": "ehr", "value": "None"},
            {"from": "human", "value": "User: Could you kindly give me a rundown of all body areas connected to medical equipment?"},
            {
                "from": "gpt",
                "value": "Certainly! The primary anatomical regions linked to medical apparatus include the cardiac silhouette and the mediastinal area.",
            },
            {
                "from": "human",
                "value": "User: Could you identify all anatomical locations linked with either spinal degenerative changes or pneumomediastinum?",
            },
            {
                "from": "gpt",
                "value": "The spine is the sole anatomical location associated with spinal degenerative changes; however, I am unable to find connections linking it directly with pneumomediastinum.",
            },
            {"from": "human", "value": "Show all tubes/lines that are found in both the upper mediastinum and left hemidiaphragm"},
            {
                "from": "gpt",
                "value": "I regret to inform you that there aren't any tubes or lines present in both the upper mediastinum and the left hemidiaphragm according to available information.",
            },
            {"from": "human", "value": "Please explain the overall diagnosis by looking at the patient's CXR image"},
            {
                "from": "gpt",
                "value": "Based on the patient's chest X-ray (CXR), there are significant medical interventions in place. The cardiac silhouette and mediastinal area are connected to medical equipment, indicating cardiac and mediastinal monitoring or support. The presence of spinal degenerative changes is noted in the spine, but there's no direct link to pneumomediastinum in this case. Additionally, there are no tubes or lines in both the upper mediastinum and the left hemidiaphragm. These observations collectively suggest a patient under intensive medical care, with a focus on cardiac and spinal health, requiring detailed monitoring and possibly advanced therapeutic interventions",
            },
        ],
    },
    {
        "q-vqa": [
            "User: Is pneumothorax still no compared to the other study of this patient?",
            "User: Is there any abnormality only in the right hilar structures but not in the right lower lung zone?",
            "User: Is the width of the cardiac silhouette wider than 1/2 of the thorax width?",
            "User: Are there both hydropneumothorax and lung cancer in the right upper lung zone?",
        ],
        "a-vqa": ["Assistant: Yes.", "Assistant: Nothing.", "Assistant: Yes.", "Assistant: No."],
        "q-ehr": [
            "User: Is there either increased reticular markings/ild pattern or multiple masses/nodules?",
            "User: Has patient been diagnosed with dermatitis, unspecified within 2 day after having had a chest x-ray study indicating any abnormality on 05/this year?",
            "User: Is there any abnormality that are resolved compared to the second to last study of patient last year?",
            "User: Is there any common anatomical locations related to both increased reticular markings/ild pattern and multiple masses/nodules?",
            "User: What was the name of the microbiology test that patient first received since 109 month ago?",
        ],
        "a-ehr": [
            "Assistant: No.",
            "Assistant: Yes.",
            "Assistant: Atelectasis, lung opacity, pneumonia, tortuous aorta, vascular congestion",
            "Assistant: Nothing.",
            "Assistant: Blood culture, routine",
        ],
        "conversations": [
            {
                "from": "ehr",
                "value": "The patient's imaging does not show both increased reticular markings or interstitial lung disease (ILD) pattern and multiple masses or nodules."
                "The patient was diagnosed with unspecified dermatitis within two days following a chest X-ray conducted in May of this year, which revealed an abnormality."
                "The patient's most recent study shows improvements in atelectasis, lung opacity, pneumonia, and vascular congestion compared to the second to last study conducted last year, though the tortuous aorta remains unchanged."
                "There are no anatomical locations in the patient exhibiting increased reticular markings or an interstitial lung disease (ILD) pattern, nor are there any multiple masses or nodules present."
                "The patient's first microbiology test, conducted 109 months ago, was a routine blood culture.",
            },
            {"from": "human", "value": "Is pneumothorax still no compared to the other study of this patient?"},
            {"from": "gpt", "value": "Yes, compared to the other study, there is still no pneumothorax present."},
            {"from": "human", "value": "Is there any abnormality only in the right hilar structures but not in the right lower lung zone?"},
            {"from": "gpt", "value": "There are no abnormalities present in either the right hilar structures or the right lower lung zone."},
            {"from": "human", "value": "Is the width of the cardiac silhouette wider than 1/2 of the thorax width?"},
            {"from": "gpt", "value": "Yes, the width of the cardiac silhouette is indeed wider than half of the thorax width."},
            {"from": "human", "value": "Are there both hydropneumothorax and lung cancer in the right upper lung zone?"},
            {"from": "gpt", "value": "No, there is neither hydropneumothorax nor lung cancer in the right upper lung zone."},
            {"from": "human", "value": "Please summary diagnosis of the patient"},
            {
                "from": "gpt",
                "value": "The patient's diagnosis, based on the summarized responses, indicates no pneumothorax when compared to previous studies. Additionally, there are no abnormalities in the right hilar structures or the right lower lung zone. However, there is a notable finding of an enlarged cardiac silhouette, with its width exceeding half of the thorax width. Furthermore, there is no evidence of hydropneumothorax or lung cancer in the right upper lung zone. These findings suggest a primarily cardiac issue, evidenced by the enlarged cardiac silhouette, without significant pulmonary complications like pneumothorax or lung cancer.",
            },
        ],
    },
    {
        "q-vqa": [
            "User: Are there indications of both lung opacity and granulomatous diseases?",
            "User: Is granulomatous diseases detectable in both the left mid lung zone and the right upper lung zone?",
            "User: Is there any sign of abnormality in the mediastinum?",
            "User: Provide a catalogue of all anatomical findings and diseases seen?",
        ],
        "a-vqa": [
            "Assistant: Yes",
            "Assistant: Yes",
            "Assistant: No",
            "Assistant: Granulomatous disease, lung lesion, lung opacity, multiple masses/nodules",
        ],
        "q-ehr": [],
        "a-ehr": [],
        "conversations": [
            {"from": "ehr", "value": "None"},
            {"from": "human", "value": "Are there indications of both lung opacity and granulomatous diseases?"},
            {"from": "gpt", "value": "Yes, there are indications of both lung opacity and granulomatous diseases."},
            {"from": "human", "value": "Is granulomatous diseases detectable in both the left mid lung zone and the right upper lung zone?"},
            {"from": "gpt", "value": "Yes, granulomatous diseases are detectable in both the left mid lung zone and the right upper lung zone."},
            {"from": "human", "value": "Is there any sign of abnormality in the mediastinum?"},
            {"from": "gpt", "value": "No, there are no signs of abnormality in the mediastinum."},
            {"from": "human", "value": "Provide a catalogue of all anatomical findings and diseases seen."},
            {
                "from": "gpt",
                "value": "The anatomical findings and diseases present include granulomatous disease, a lung lesion, lung opacity, and multiple masses or nodules.",
            },
            {"from": "human", "value": "Could you concise symptoms of the patient?"},
            {
                "from": "gpt",
                "value": "The patient exhibits symptoms indicative of granulomatous disease, as evidenced by lung opacity and multiple masses or nodules, primarily in the left mid lung zone and right upper lung zone, with no abnormalities detected in the mediastinum.",
            },
        ],
    },
    {
        "q-vqa": ["User: Is there any anatomicalfinding that are newly detected in the left lung compared to the other study of this patient?"],
        "a-vqa": ["Assistant: Atelectasis, pleural effusion"],
        "q-ehr": [
            "User: What was the name of the procedure that patient received two times until 11/2102?",
            "User: What was the name of the drug that patient was prescribed within 2 day after having been diagnosed with primary osteoarthritis, left shoulder since 08/2100?",
            "User: What was the marital status of patient on the last hospital visit?",
            "User: What was the name of the intake that patient last had on 03/24/this year?",
        ],
        "a-ehr": [
            "Assistant: Acetaminophen, furosemide, polyethylene glycol, sodium chloride 0.9%, sodium chloride 0.9%  flush, valganciclovir, 0.9% sodium chloride, 0.9% sodium chloride, adenosine, dexamethasone, insulin, 0.9% sodium chloride, 0.9% sodium chloride, aspirin, dermoplast spray, neutra-phos, sw, vial, acetaminophen, haloperidol, lidocaine 5% patch, magnesium sulfate, sodium chloride 0.9%  flush, glucose gel",
            "Assistant: Venous catheterization, not elsewhere classified, esophagogastroduodenoscopy [egd] with closed biopsy, other nonoperative respiratory measurements, electrocardiogram",
            "Assistant: Single",
            "Assistant: Gastric meds",
        ],
        "conversations": [
            {
                "from": "ehr",
                "value": "The patient was first prescribed 0.9\% sodium chloride in July of last year."
                "After being diagnosed with primary osteoarthritis in the left shoulder in August 2100, the patient was prescribed a range of medications including Acetaminophen, Furosemide, Polyethylene Glycol, various forms of Sodium Chloride, Valganciclovir, Adenosine, Dexamethasone, Insulin, Aspirin, Dermoplast Spray, Neutra-Phos, SW, Vial, Haloperidol, Lidocaine 5% Patch, Magnesium Sulfate, and Glucose Gel."
                "The patient underwent venous catheterization, esophagogastroduodenoscopy (EGD) with closed biopsy, other nonoperative respiratory measurements, and electrocardiogram procedures twice until November 2102."
                "On their last hospital visit, the patient's marital status was recorded as single."
                "The last intake noted for the patient on March 24th this year was gastric meds.",
            },
            {
                "from": "human",
                "value": "Is there any anatomical finding that is newly detected in the left lung compared to the other study of this patient?",
            },
            {
                "from": "gpt",
                "value": "Newly detected anatomical findings in the left lung, when compared to previous studies, include atelectasis and pleural effusion.",
            },
        ],
    },
]