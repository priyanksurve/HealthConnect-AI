"""
symptom_checker.py
------------------
Comprehensive Medical Symptom Analysis, Disease Prediction, and Doctor Recommendation Engine.

Analyzes user-entered symptoms (via free text or structured selection),
evaluates clinical symptom profiles, calculates match confidence,
determines severity levels, detects red-flag emergency symptoms,
recommends the appropriate medical specialist (doctor), and suggests
clinical diagnostic tests and home precautions.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

# =========================================================
# 1. CATEGORIZED SYMPTOMS DICTIONARY
# =========================================================

CATEGORIZED_SYMPTOMS = {
    "🫁 Respiratory & Breathing": [
        ("cough", "Cough (Dry or Productive)"),
        ("productive_cough", "Cough with Phlegm / Mucus (Yellow/Green)"),
        ("chronic_cough", "Chronic Cough (> 3 weeks)"),
        ("shortness_of_breath", "Shortness of Breath (Dyspnea)"),
        ("wheezing", "Wheezing (Whistling sound when breathing)"),
        ("rapid_breathing", "Rapid / Shallow Breathing (Tachypnea)"),
        ("coughing_blood", "Coughing up Blood (Hemoptysis)"),
        ("chest_congestion", "Chest Congestion / Heavy Tightness"),
    ],
    "❤️ Chest & Cardiovascular": [
        ("chest_pain", "Chest Pain (General)"),
        ("sharp_chest_pain", "Sharp / Stabbing Chest Pain (worse with breathing)"),
        ("crushing_chest_pain", "Crushing / Squeezing Chest Pressure"),
        ("pain_radiating_arm", "Chest Pain Radiating to Left Arm / Jaw / Neck"),
        ("palpitations", "Palpitations / Rapid Heartbeat"),
        ("dizziness", "Dizziness / Lightheadedness / Fainting"),
    ],
    "🌡️ Fever & Systemic": [
        ("fever", "Fever (High Body Temperature)"),
        ("high_fever", "High-Grade Fever (> 102°F / 39°C)"),
        ("chills", "Chills & Shivering / Shakes"),
        ("night_sweats", "Night Sweats / Profuse Sweating"),
        ("fatigue", "Severe Fatigue / Weakness / Lethargy"),
        ("unexplained_weight_loss", "Unexplained Weight Loss"),
        ("loss_of_appetite", "Loss of Appetite / Poor Intake"),
    ],
    "🗣️ Throat, Nose & Head": [
        ("sore_throat", "Sore Throat / Painful Swallowing"),
        ("runny_nose", "Runny or Stuffy Nose (Rhinorrhea)"),
        ("sneezing", "Frequent Sneezing"),
        ("nasal_congestion", "Nasal Congestion / Blocked Sinuses"),
        ("facial_pressure", "Facial Pain / Sinus Pressure / Forehead Aches"),
        ("headache", "Headache (Moderate to Severe)"),
        ("loss_of_taste_smell", "Loss of Taste or Smell (Anosmia)"),
        ("swollen_lymph_nodes", "Swollen Neck Glands / Lymph Nodes"),
    ],
    "🍽️ Digestive & Acid": [
        ("heartburn", "Heartburn / Acid Reflux / Burning Chest"),
        ("acid_regurgitation", "Sour / Acidic Taste in Mouth"),
        ("nausea", "Nausea or Upset Stomach"),
        ("vomiting", "Vomiting"),
        ("abdominal_pain", "Abdominal Pain / Stomach Cramps"),
        ("diarrhea", "Watery Diarrhea / Loose Stools"),
        ("difficulty_swallowing", "Difficulty Swallowing (Dysphagia)"),
    ],
    "💪 General & Musculoskeletal": [
        ("body_ache", "Generalized Body Aches / Muscle Pain"),
        ("joint_pain", "Joint Pain / Arthralgia"),
        ("malaise", "General Feeling of Discomfort / Malaise"),
        ("bluish_lips_skin", "Bluish Tint on Lips / Fingernails (Cyanosis)"),
        ("confusion", "Confusion / Mental Fog (in elderly)"),
    ]
}

ALL_SYMPTOM_CHOICES = []
for _cat, _sym_list in CATEGORIZED_SYMPTOMS.items():
    for _key, _label in _sym_list:
        ALL_SYMPTOM_CHOICES.append((_key, _label))

SYMPTOM_LABELS_DICT = dict(ALL_SYMPTOM_CHOICES)

# =========================================================
# 2. SYMPTOM ALIASES & NLP MAPPING
# =========================================================

SYMPTOM_ALIASES = {
    # Fever & Chills
    "fever": "fever",
    "fevers": "fever",
    "febrile": "fever",
    "temperature": "fever",
    "high temperature": "high_fever",
    "high fever": "high_fever",
    "burning up": "high_fever",
    "chills": "chills",
    "shivering": "chills",
    "shakes": "chills",
    "cold chills": "chills",
    "sweating": "night_sweats",
    "sweats": "night_sweats",
    "night sweats": "night_sweats",
    
    # Cough & Respiratory
    "cough": "cough",
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "productive_cough",
    "productive cough": "productive_cough",
    "cough with phlegm": "productive_cough",
    "cough with mucus": "productive_cough",
    "yellow phlegm": "productive_cough",
    "green phlegm": "productive_cough",
    "phlegm": "productive_cough",
    "mucus": "productive_cough",
    "sputum": "productive_cough",
    "chronic cough": "chronic_cough",
    "persistent cough": "chronic_cough",
    "coughing blood": "coughing_blood",
    "blood in cough": "coughing_blood",
    "hemoptysis": "coughing_blood",
    "blood sputum": "coughing_blood",
    
    # Breathing
    "shortness of breath": "shortness_of_breath",
    "short of breath": "shortness_of_breath",
    "breathlessness": "shortness_of_breath",
    "breathing problem": "shortness_of_breath",
    "difficulty breathing": "shortness_of_breath",
    "trouble breathing": "shortness_of_breath",
    "can't breathe": "shortness_of_breath",
    "cant breathe": "shortness_of_breath",
    "dyspnea": "shortness_of_breath",
    "wheezing": "wheezing",
    "whistling breath": "wheezing",
    "rapid breathing": "rapid_breathing",
    "shallow breathing": "rapid_breathing",
    "fast breathing": "rapid_breathing",
    "chest congestion": "chest_congestion",
    "congested chest": "chest_congestion",
    "tight chest": "chest_congestion",
    
    # Chest Pain & Heart
    "chest pain": "chest_pain",
    "pain in chest": "chest_pain",
    "chest ache": "chest_pain",
    "chest hurts": "chest_pain",
    "sharp chest pain": "sharp_chest_pain",
    "stabbing chest pain": "sharp_chest_pain",
    "pleuritic chest pain": "sharp_chest_pain",
    "crushing chest pain": "crushing_chest_pain",
    "crushing pain": "crushing_chest_pain",
    "heavy chest": "crushing_chest_pain",
    "chest pressure": "crushing_chest_pain",
    "chest tightness": "crushing_chest_pain",
    "pain in left arm": "pain_radiating_arm",
    "pain radiating to arm": "pain_radiating_arm",
    "pain radiating to jaw": "pain_radiating_arm",
    "radiating chest pain": "pain_radiating_arm",
    "palpitations": "palpitations",
    "racing heart": "palpitations",
    "fast heartbeat": "palpitations",
    "rapid pulse": "palpitations",
    "dizziness": "dizziness",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "fainting": "dizziness",
    
    # Throat & Head & ENT
    "sore throat": "sore_throat",
    "throat pain": "sore_throat",
    "scratchy throat": "sore_throat",
    "pain swallowing": "sore_throat",
    "runny nose": "runny_nose",
    "running nose": "runny_nose",
    "sneezing": "sneezing",
    "sneeze": "sneezing",
    "nasal congestion": "nasal_congestion",
    "blocked nose": "nasal_congestion",
    "stuffy nose": "nasal_congestion",
    "sinus": "facial_pressure",
    "sinus pressure": "facial_pressure",
    "facial pain": "facial_pressure",
    "headache": "headache",
    "head ache": "headache",
    "head pain": "headache",
    "migraine": "headache",
    "loss of taste": "loss_of_taste_smell",
    "loss of smell": "loss_of_taste_smell",
    "anosmia": "loss_of_taste_smell",
    "swollen glands": "swollen_lymph_nodes",
    "swollen lymph nodes": "swollen_lymph_nodes",
    
    # Digestive & Acid
    "heartburn": "heartburn",
    "acid reflux": "heartburn",
    "acidity": "heartburn",
    "burning in chest": "heartburn",
    "burning sensation in chest": "heartburn",
    "sour taste": "acid_regurgitation",
    "regurgitation": "acid_regurgitation",
    "nausea": "nausea",
    "nauseous": "nausea",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "puke": "vomiting",
    "stomach pain": "abdominal_pain",
    "abdominal pain": "abdominal_pain",
    "belly ache": "abdominal_pain",
    "stomach cramps": "abdominal_pain",
    "diarrhea": "diarrhea",
    "loose motion": "diarrhea",
    "loose stools": "diarrhea",
    "difficulty swallowing": "difficulty_swallowing",
    "dysphagia": "difficulty_swallowing",
    
    # General
    "fatigue": "fatigue",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "weakness": "fatigue",
    "exhaustion": "fatigue",
    "lethargy": "fatigue",
    "body ache": "body_ache",
    "body pain": "body_ache",
    "muscle ache": "body_ache",
    "muscle pain": "body_ache",
    "myalgia": "body_ache",
    "joint pain": "joint_pain",
    "aching joints": "joint_pain",
    "weight loss": "unexplained_weight_loss",
    "lost weight": "unexplained_weight_loss",
    "loss of appetite": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "poor appetite": "loss_of_appetite",
    "blue lips": "bluish_lips_skin",
    "cyanosis": "bluish_lips_skin",
    "blue nails": "bluish_lips_skin",
    "confusion": "confusion",
    "confused": "confusion",
    "disoriented": "confusion",
    "malaise": "malaise",
    "unwell": "malaise",
}

RED_FLAG_SYMPTOMS = {
    "crushing_chest_pain": "Crushing chest pain / pressure (Possible Acute Coronary Event / Heart Attack)",
    "pain_radiating_arm": "Chest pain radiating to arm / jaw / neck (Cardiac Emergency Indicator)",
    "coughing_blood": "Coughing up blood / Hemoptysis (Serious Lung / Vascular Pathology)",
    "bluish_lips_skin": "Cyanosis / Blue lips or nails (Critical Low Blood Oxygen Level)",
    "confusion": "Sudden confusion or disorientation with respiratory infection (Severe Sepsis / Hypoxia)",
    "shortness_of_breath": "Severe acute shortness of breath (Respiratory Distress)",
}

# =========================================================
# 3. CLINICAL DISEASE KNOWLEDGE BASE
# =========================================================

DISEASE_KNOWLEDGE_BASE = {
    "Pneumonia": {
        "name": "Pneumonia (Bacterial or Viral Lung Infection)",
        "icon": "🫁",
        "category": "Lower Respiratory Infection",
        "is_pulmonary": True,
        "base_severity": "High",
        "summary": (
            "Pneumonia is an inflammatory infection of the lung air sacs (alveoli), "
            "which may fill with fluid or purulent material. It often causes productive cough, "
            "fever, chills, and breathing difficulty."
        ),
        "specialist": "Pulmonologist / Chest Physician",
        "specialist_icon": "🫁",
        "specialist_reason": (
            "A Pulmonologist specializes in lower respiratory tract infections, lung parenchyma "
            "disorders, oxygen saturation monitoring, and targeted antibiotic/antiviral therapy."
        ),
        "urgency_level": "High Priority (Consult within 12-24 hours or immediate ER if breathless)",
        "symptoms": {
            "fever": 1.5,
            "high_fever": 2.0,
            "chills": 1.8,
            "cough": 1.2,
            "productive_cough": 2.5,
            "shortness_of_breath": 2.5,
            "sharp_chest_pain": 2.2,
            "rapid_breathing": 2.0,
            "fatigue": 1.2,
            "night_sweats": 1.5,
            "bluish_lips_skin": 3.0,
            "confusion": 2.5,
            "chest_congestion": 1.6,
        },
        "tests": [
            "Chest X-ray (PA & Lateral View) — [Available in this AI Portal]",
            "High-Resolution Computed Tomography (HRCT Chest)",
            "Complete Blood Count (CBC) with Differential & ESR/CRP",
            "Sputum Gram Stain & Culture",
            "Pulse Oximetry (SpO2 Oxygen Saturation)",
            "Arterial Blood Gas (ABG) if hypoxic"
        ],
        "precautions": [
            "Rest completely and avoid any strenuous physical exertion.",
            "Stay well hydrated with warm liquids, soups, and electrolyte solutions.",
            "Monitor body temperature and oxygen saturation (SpO2) every 4-6 hours.",
            "Use a humidifier or inhale steam to loosen lung mucus.",
            "Do NOT suppress productive phlegm cough without doctor's instruction.",
            "Avoid exposure to cold air, smoke, and air pollutants."
        ],
        "red_flags": [
            "SpO2 dropping below 92% on room air",
            "Severe struggle to catch breath or inability to speak full sentences",
            "Bluish color of lips, face, or nailbeds",
            "Persistent high fever (> 103°F) unresponsive to medication"
        ]
    },

    "Acute Bronchitis": {
        "name": "Acute Bronchitis / Tracheobronchitis",
        "icon": "🌬️",
        "category": "Airway Inflammation",
        "is_pulmonary": True,
        "base_severity": "Moderate",
        "summary": (
            "Inflammation of the bronchial tubes carrying air to your lungs, causing "
            "a persistent mucus-producing cough, airway irritation, and mild chest tightness."
        ),
        "specialist": "Pulmonologist or General Physician",
        "specialist_icon": "🩺",
        "specialist_reason": (
            "A Pulmonologist or General Physician can assess bronchial inflammation, rule out "
            "pneumonia via chest imaging, and prescribe bronchodilators or anti-inflammatory inhalers."
        ),
        "urgency_level": "Moderate (Consult within 24-48 hours)",
        "symptoms": {
            "cough": 2.0,
            "productive_cough": 2.2,
            "chest_congestion": 2.0,
            "wheezing": 1.8,
            "sore_throat": 1.3,
            "fatigue": 1.2,
            "fever": 1.0,
            "body_ache": 1.1,
            "shortness_of_breath": 1.5,
        },
        "tests": [
            "Chest X-ray to rule out pneumonia or lung infiltration",
            "Spirometry / Peak Expiratory Flow Rate (PEFR)",
            "Complete Blood Count (CBC)"
        ],
        "precautions": [
            "Drink plenty of warm water, ginger tea, and clear broths.",
            "Inhale steam twice daily to soothe bronchial passages.",
            "Use honey and warm water for natural cough suppression (adults).",
            "Avoid cigarette smoke, vaping, and harsh cleaning fumes."
        ],
        "red_flags": [
            "Cough lasting longer than 3 weeks or worsening significantly",
            "Coughing up rust-colored, thick brown, or bloody sputum",
            "Onset of high fever (> 101.5°F) or severe breathlessness"
        ]
    },

    "COVID-19 / Viral Influenza": {
        "name": "COVID-19 / Influenza (Viral Respiratory Syndrome)",
        "icon": "🦠",
        "category": "Viral Systemic Infection",
        "is_pulmonary": True,
        "base_severity": "Moderate to High",
        "summary": (
            "A contagious viral respiratory illness that causes widespread systemic symptoms "
            "including sudden fever, body aches, exhaustion, dry cough, and sensory loss."
        ),
        "specialist": "Infectious Disease Specialist / Pulmonologist",
        "specialist_icon": "🔬",
        "specialist_reason": (
            "Specialized in managing viral viral loads, post-viral pulmonary complications, "
            "and prescribing targeted antiviral treatments (Paxlovid/Oseltamivir)."
        ),
        "urgency_level": "High Priority (Test & isolate, consult doctor within 24 hours)",
        "symptoms": {
            "fever": 2.0,
            "high_fever": 2.2,
            "chills": 1.8,
            "fatigue": 2.2,
            "body_ache": 2.2,
            "loss_of_taste_smell": 3.0,
            "headache": 1.8,
            "sore_throat": 1.6,
            "cough": 1.8,
            "shortness_of_breath": 2.0,
            "runny_nose": 1.2,
            "nasal_congestion": 1.2,
            "malaise": 1.8,
        },
        "tests": [
            "Rapid Antigen Test (RAT) or RT-PCR for SARS-CoV-2 / Influenza A & B",
            "Chest X-Ray / HRCT Chest if shortness of breath occurs",
            "Inflammatory Markers: CRP, Serum Ferritin, D-Dimer"
        ],
        "precautions": [
            "Isolate in a well-ventilated room to protect family members.",
            "Check oxygen level (SpO2) 3 times daily using a pulse oximeter.",
            "Maintain strict bed rest and high fluid intake.",
            "Wear a high-efficiency mask (N95/KN95) when in shared areas."
        ],
        "red_flags": [
            "Difficulty breathing or persistent chest heaviness",
            "SpO2 dropping below 94%",
            "Inability to stay awake, severe confusion, or bluish discoloration"
        ]
    },

    "Asthma / Reactive Airway Disease": {
        "name": "Bronchial Asthma / Acute Bronchospasm",
        "icon": "🫁",
        "category": "Chronic Airway Disease",
        "is_pulmonary": True,
        "base_severity": "Moderate to High",
        "summary": (
            "A condition in which airways narrow, swell, and produce extra mucus, making "
            "breathing difficult and triggering coughing, wheezing, and shortness of breath."
        ),
        "specialist": "Pulmonologist / Allergy & Asthma Specialist",
        "specialist_icon": "🫁",
        "specialist_reason": (
            "A Pulmonologist/Allergist can perform pulmonary function testing, prescribe rescue "
            "and controller inhalers, and design an Asthma Action Plan."
        ),
        "urgency_level": "Urgent (Seek immediate medical care if rescue inhaler fails)",
        "symptoms": {
            "wheezing": 3.0,
            "shortness_of_breath": 2.8,
            "chest_congestion": 2.2,
            "cough": 1.8,
            "rapid_breathing": 2.2,
            "crushing_chest_pain": 1.0,
        },
        "tests": [
            "Spirometry with Pre- and Post-Bronchodilator reversibility test",
            "Fractional Exhaled Nitric Oxide (FeNO) test",
            "Chest X-ray to rule out pneumothorax or consolidation",
            "Allergy Skin Prick Testing or IgE panel"
        ],
        "precautions": [
            "Keep prescribed rescue inhaler (e.g. Salbutamol / Albuterol) always within reach.",
            "Identify and avoid triggers (dust mites, pollen, pet dander, cold air, smoke).",
            "Sit upright and try pursed-lip breathing during mild tightness.",
            "Never ignore worsening nighttime cough or chest tightness."
        ],
        "red_flags": [
            "Severe breathlessness, inability to talk in full sentences",
            "No relief 15 minutes after using rescue inhaler",
            "Chest wall and ribs sucking inward on inhalation (retractions)",
            "Gray or blue color in lips, gums, or fingertips"
        ]
    },

    "Acute Coronary Syndrome / Angina": {
        "name": "Cardiac Angina / Acute Coronary Syndrome",
        "icon": "❤️",
        "category": "Cardiovascular Emergency",
        "is_pulmonary": False,
        "base_severity": "Critical / Emergency",
        "summary": (
            "A critical cardiovascular condition caused by reduced blood flow to the heart muscle. "
            "Presents with severe chest tightness/squeezing that may radiate to the left arm or jaw."
        ),
        "specialist": "Cardiologist / Emergency Medicine Physician",
        "specialist_icon": "❤️",
        "specialist_reason": (
            "Requires immediate emergency cardiac intervention, ECG monitoring, cardiac enzymes "
            "evaluation, and potential emergency catheterization/angioplasty."
        ),
        "urgency_level": "🚨 IMMEDIATE EMERGENCY (Call Emergency / Go to Nearest ER Now)",
        "symptoms": {
            "crushing_chest_pain": 3.5,
            "pain_radiating_arm": 3.5,
            "chest_pain": 2.5,
            "shortness_of_breath": 2.2,
            "night_sweats": 2.0,
            "dizziness": 2.2,
            "palpitations": 2.0,
            "nausea": 1.6,
            "fatigue": 1.5,
        },
        "tests": [
            "12-Lead Electrocardiogram (ECG / EKG) — Immediate",
            "High-Sensitivity Cardiac Troponin (hs-cTnI / hs-cTnT)",
            "2D Echocardiogram (Echo)",
            "Coronary Angiography (CAG)",
            "Creatine Kinase-MB (CK-MB) & Lipid Profile"
        ],
        "precautions": [
            "STOP all physical activity immediately and sit or lie down in a comfortable position.",
            "Call local emergency medical services immediately (do not drive yourself).",
            "Chew one standard Aspirin (300mg) if advised by emergency operator and not allergic.",
            "Loosen tight clothing around neck and waist and stay as calm as possible."
        ],
        "red_flags": [
            "Crushing chest pressure radiating to arm, jaw, neck or upper back",
            "Sudden cold sweats, intense nausea, or impending sense of doom",
            "Fainting, severe dizziness, or profound shortness of breath"
        ]
    }
}

# Additional Disease Profiles
DISEASE_KNOWLEDGE_BASE.update({
    "GERD / Acid Reflux": {
        "name": "GERD (Gastroesophageal Reflux Disease)",
        "icon": "🔥",
        "category": "Gastrointestinal",
        "is_pulmonary": False,
        "base_severity": "Mild to Moderate",
        "summary": (
            "Occurs when stomach acid repeatedly flows back into the tube connecting your mouth "
            "and stomach (esophagus). This backwash can irritate the lining and cause non-cardiac chest burning."
        ),
        "specialist": "Gastroenterologist / General Physician",
        "specialist_icon": "🩺",
        "specialist_reason": (
            "A Gastroenterologist evaluates acid reflux severity, rules out esophageal damage, "
            "and prescribes proton pump inhibitors (PPIs) or H2 blockers."
        ),
        "urgency_level": "Routine / Moderate (Consult within a few days)",
        "symptoms": {
            "heartburn": 3.0,
            "acid_regurgitation": 2.8,
            "chest_pain": 1.8,
            "difficulty_swallowing": 2.2,
            "cough": 1.5,
            "sore_throat": 1.4,
            "nausea": 1.3,
        },
        "tests": [
            "Upper GI Endoscopy (Esophagogastroduodenoscopy - EGD)",
            "24-Hour Esophageal pH / Impedance Monitoring",
            "Barium Swallow / Esophagram",
            "ECG to rule out cardiac origin of chest discomfort"
        ],
        "precautions": [
            "Avoid trigger foods (spicy, fatty, citrus, chocolate, caffeine, mint).",
            "Do not lie down for at least 3 hours after eating a meal.",
            "Elevate the head of your bed by 6 inches when sleeping.",
            "Eat smaller, more frequent meals rather than large heavy dinners.",
            "Avoid tight-fitting belts and clothing around the abdomen."
        ],
        "red_flags": [
            "Severe pain radiating to jaw or left arm (seek cardiac emergency care)",
            "Difficulty or pain while swallowing solid foods or liquids (dysphagia)",
            "Vomiting blood or coffee-ground material, or dark black stools"
        ]
    },

    "Common Cold / Viral URTI": {
        "name": "Common Cold / Acute Viral Rhinopharyngitis",
        "icon": "🤧",
        "category": "Upper Respiratory Tract",
        "is_pulmonary": False,
        "base_severity": "Mild",
        "summary": (
            "A mild viral infection of the nose and throat (upper respiratory tract). "
            "Usually harmless and resolves on its own within 7-10 days."
        ),
        "specialist": "General Physician / Family Doctor",
        "specialist_icon": "🩺",
        "specialist_reason": (
            "A General Physician provides symptomatic management, ensures no secondary bacterial "
            "complication has occurred, and advises on safe over-the-counter remedies."
        ),
        "urgency_level": "Mild / Routine (Self-care or routine visit if persistent)",
        "symptoms": {
            "runny_nose": 2.5,
            "sneezing": 2.5,
            "nasal_congestion": 2.2,
            "sore_throat": 2.0,
            "cough": 1.5,
            "headache": 1.2,
            "malaise": 1.2,
            "fever": 0.8,
            "body_ache": 1.0,
        },
        "tests": [
            "Routine clinical ENT examination",
            "Rapid Strep test or viral throat swab if severe sore throat occurs"
        ],
        "precautions": [
            "Get plenty of restful sleep and stay warm.",
            "Drink hot tea with honey and lemon, and warm broths.",
            "Use saline nasal drops/sprays to relieve nasal blockage.",
            "Gargle with warm salt water 2-3 times daily for sore throat relief."
        ],
        "red_flags": [
            "High fever persisting for more than 3-4 days",
            "Severe ear pain, intense facial headache, or difficulty breathing",
            "Symptoms lasting longer than 10-14 days without improvement"
        ]
    },

    "Acute Sinusitis": {
        "name": "Acute Rhinosinusitis / Sinus Infection",
        "icon": "👃",
        "category": "ENT / Paranasal Sinuses",
        "is_pulmonary": False,
        "base_severity": "Mild to Moderate",
        "summary": (
            "Inflammation of the tissue lining the sinuses, causing thick nasal mucus, "
            "facial pressure/pain around eyes and cheeks, and headache."
        ),
        "specialist": "ENT Specialist (Otolaryngologist) / General Physician",
        "specialist_icon": "👂",
        "specialist_reason": (
            "An ENT specialist can perform nasal endoscopy, evaluate sinus drainage pathways, "
            "and prescribe targeted nasal steroids, decongestants, or antibiotics."
        ),
        "urgency_level": "Moderate (Consult within 2-3 days)",
        "symptoms": {
            "facial_pressure": 3.0,
            "nasal_congestion": 2.5,
            "headache": 2.2,
            "runny_nose": 1.8,
            "sore_throat": 1.3,
            "cough": 1.2,
            "fever": 1.0,
            "fatigue": 1.1,
            "loss_of_taste_smell": 1.5,
        },
        "tests": [
            "Diagnostic Nasal Endoscopy",
            "CT Scan of Paranasal Sinuses (PNS) if chronic or recurrent",
            "Sinus culture / nasal swab"
        ],
        "precautions": [
            "Perform warm saline sinus rinses (Neti pot / squeeze bottle).",
            "Apply warm compresses over the nose, cheeks, and eyes.",
            "Inhale steam twice daily and use a room humidifier.",
            "Drink plenty of fluids to thin nasal secretions."
        ],
        "red_flags": [
            "Swelling, redness, or tenderness around the eye or eyelid",
            "Double vision or sudden visual changes",
            "Severe forehead headache with high fever and neck stiffness"
        ]
    },

    "Pulmonary Tuberculosis": {
        "name": "Pulmonary Tuberculosis (TB)",
        "icon": "🫁",
        "category": "Infectious Mycobacterial Disease",
        "is_pulmonary": True,
        "base_severity": "High / Chronic Urgent",
        "summary": (
            "A contagious bacterial infection caused by Mycobacterium tuberculosis that primarily "
            "attacks the lungs. Characterized by long-standing cough, night sweats, and weight loss."
        ),
        "specialist": "Pulmonologist / Chest Specialist / Infectious Disease Expert",
        "specialist_icon": "🫁",
        "specialist_reason": (
            "Requires specialized anti-tubercular therapy (ATT / DOTS), sputum monitoring, "
            "and serial chest radiologic evaluation."
        ),
        "urgency_level": "High Priority (Seek formal evaluation within 24-48 hours)",
        "symptoms": {
            "chronic_cough": 3.0,
            "coughing_blood": 3.0,
            "unexplained_weight_loss": 2.8,
            "night_sweats": 2.8,
            "fever": 1.8,
            "chest_pain": 1.8,
            "sharp_chest_pain": 2.0,
            "fatigue": 2.0,
            "loss_of_appetite": 2.2,
            "shortness_of_breath": 1.6,
        },
        "tests": [
            "Chest X-ray (PA View) — [Screening available in this AI Portal]",
            "Sputum Smear for Acid-Fast Bacilli (AFB)",
            "GeneXpert MTB/RIF (Nucleic Acid Amplification)",
            "Tuberculin Skin Test (Mantoux) or IGRA Blood Test (QuantiFERON-TB)",
            "High-Resolution CT Chest"
        ],
        "precautions": [
            "Cover mouth and nose with a tissue when coughing or sneezing.",
            "Ensure good ventilation in living spaces with open windows.",
            "Never stop or miss prescribed anti-TB medications once started.",
            "Eat a high-protein, nutrient-rich diet to regain strength."
        ],
        "red_flags": [
            "Coughing up significant amounts of fresh bright-red blood",
            "Rapid progressive weight loss and profound weakness",
            "Severe breathing difficulty and high persistent fever"
        ]
    },

    "Pleurisy / Pleural Effusion": {
        "name": "Pleurisy / Pleural Inflammation",
        "icon": "🫁",
        "category": "Pleural Space Disorder",
        "is_pulmonary": True,
        "base_severity": "High",
        "summary": (
            "Inflammation of the pleural sheets that cover the lungs. It causes sharp chest "
            "pain that worsens noticeably with breathing, coughing, or sneezing."
        ),
        "specialist": "Pulmonologist / Thoracic Physician",
        "specialist_icon": "🫁",
        "specialist_reason": (
            "Specialized in diagnosing pleural rubs, performing pleural fluid analysis (thoracentesis), "
            "and resolving underlying causes (infection, autoimmune, or pulmonary embolism)."
        ),
        "urgency_level": "High Priority (Consult within 12-24 hours)",
        "symptoms": {
            "sharp_chest_pain": 3.5,
            "chest_pain": 2.5,
            "shortness_of_breath": 2.4,
            "rapid_breathing": 2.2,
            "cough": 1.6,
            "fever": 1.5,
            "chills": 1.3,
        },
        "tests": [
            "Chest X-ray (Erect PA & Lateral Decubitus views)",
            "Thoracic Ultrasound / Contrast CT Chest",
            "Diagnostic Thoracentesis and Pleural Fluid Analysis",
            "Complete Blood Count (CBC) and D-Dimer test"
        ],
        "precautions": [
            "Rest and avoid rapid, strenuous deep breathing until evaluated.",
            "Lying on the painful side may sometimes reduce mechanical friction pain.",
            "Do not ignore worsening breathlessness."
        ],
        "red_flags": [
            "Sudden severe stabbing chest pain accompanied by intense breathlessness",
            "Coughing blood or rapid heart rate with low blood pressure",
            "Blue discoloration of lips or fingers"
        ]
    },

    "Acute Gastroenteritis": {
        "name": "Acute Gastroenteritis / Stomach Infection",
        "icon": "🤢",
        "category": "Gastrointestinal Infection",
        "is_pulmonary": False,
        "base_severity": "Moderate",
        "summary": (
            "An inflammation of the stomach and intestines typically caused by a viral or bacterial "
            "infection, leading to stomach cramps, nausea, vomiting, and watery diarrhea."
        ),
        "specialist": "Gastroenterologist / General Physician",
        "specialist_icon": "🩺",
        "specialist_reason": (
            "Manages hydration therapy, electrolyte balance, antiemetics, and targeted antimicrobial "
            "treatment if bacterial infection is confirmed."
        ),
        "urgency_level": "Moderate (Consult doctor if vomiting prevents fluid retention)",
        "symptoms": {
            "nausea": 2.8,
            "vomiting": 2.8,
            "abdominal_pain": 2.8,
            "diarrhea": 3.0,
            "fever": 1.4,
            "chills": 1.2,
            "body_ache": 1.2,
            "fatigue": 1.6,
            "dizziness": 1.8,
        },
        "tests": [
            "Stool Routine Examination & Culture",
            "Serum Electrolytes (Sodium, Potassium, Chloride, Bicarbonate)",
            "Complete Blood Count (CBC) and Renal Function Test (BUN/Creatinine)"
        ],
        "precautions": [
            "Sip Oral Rehydration Salt (ORS) solution continuously in small quantities.",
            "Stick to the BRAT diet (Bananas, Rice, Applesauce, Toast).",
            "Avoid dairy products, caffeine, greasy food, and high-sugar drinks.",
            "Wash hands thoroughly with soap to prevent transmission."
        ],
        "red_flags": [
            "Signs of severe dehydration (extreme thirst, dry mouth, little/no urination, dizziness)",
            "Inability to keep liquids down for more than 12-24 hours",
            "High fever (> 102°F) or bloody/black bowel movements"
        ]
    }
})

# =========================================================
# 4. NLP SYMPTOM EXTRACTION
# =========================================================

NEGATION_PATTERNS = [
    r"\bno\s+([a-zA-Z\s]+)",
    r"\bnot\s+([a-zA-Z\s]+)",
    r"\bwithout\s+([a-zA-Z\s]+)",
    r"\bdenies\s+([a-zA-Z\s]+)",
    r"\bnever\s+had\s+([a-zA-Z\s]+)",
    r"\bfree\s+of\s+([a-zA-Z\s]+)",
]


def extract_symptoms_from_text(text: str) -> List[str]:
    """
    Extracts canonical symptom keys from natural language patient text.
    Handles negation detection and multi-word phrase prioritization.
    """
    if not text or not text.strip():
        return []

    cleaned = text.lower().strip()
    
    # Identify negated clauses/phrases
    negated_spans = []
    for pattern in NEGATION_PATTERNS:
        for match in re.finditer(pattern, cleaned):
            start = match.start()
            end = min(match.end() + 15, len(cleaned))
            negated_spans.append((start, end))

    def is_in_negation(start_pos, end_pos):
        for n_start, n_end in negated_spans:
            if start_pos >= n_start and end_pos <= n_end:
                return True
        return False

    matched_symptoms = set()

    # Sort aliases by length descending so longer specific phrases match first
    sorted_aliases = sorted(SYMPTOM_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)

    for alias, canonical_key in sorted_aliases:
        # Regex word boundary search
        pattern = r"\b" + re.escape(alias) + r"\b"
        for match in re.finditer(pattern, cleaned):
            if not is_in_negation(match.start(), match.end()):
                matched_symptoms.add(canonical_key)

    return list(matched_symptoms)


# =========================================================
# 5. CORE SYMPTOM ANALYSIS ENGINE
# =========================================================

def analyze_symptoms(
    symptoms: List[str],
    duration_days: int = 1,
    severity_rating: int = 5
) -> Dict[str, Any]:
    """
    Performs comprehensive multi-factor clinical matching, confidence calculation,
    severity assessment, and doctor specialization recommendation.
    """
    if not symptoms:
        return {
            "success": False,
            "message": "No symptoms provided for analysis.",
            "predictions": [],
            "top_prediction": None,
        }

    user_symptoms_set = set(symptoms)
    
    # Check for critical red-flag emergency symptoms
    detected_red_flags = []
    for red_key, red_desc in RED_FLAG_SYMPTOMS.items():
        if red_key in user_symptoms_set:
            detected_red_flags.append(red_desc)

    scored_diseases = []

    for disease_id, disease_data in DISEASE_KNOWLEDGE_BASE.items():
        disease_symptoms = disease_data["symptoms"]
        total_possible_weight = sum(disease_symptoms.values())
        
        matched_weight = 0.0
        matched_keys = []

        for sym_key, weight in disease_symptoms.items():
            if sym_key in user_symptoms_set:
                matched_weight += weight
                matched_keys.append(sym_key)

        if not matched_keys:
            continue

        # Weighted match score
        raw_score = matched_weight / total_possible_weight
        
        # User symptom coverage ratio
        coverage = len(matched_keys) / max(len(user_symptoms_set), 1)

        # Combined confidence metric
        confidence_metric = (raw_score * 0.65) + (coverage * 0.35)
        confidence_pct = min(max(confidence_metric * 100, 15.0), 96.5)

        # Duration & Severity weighting bonus
        if disease_data["is_pulmonary"] and duration_days > 4:
            confidence_pct = min(confidence_pct + 4.0, 98.0)
        if "fever" in user_symptoms_set and "productive_cough" in user_symptoms_set and disease_id == "Pneumonia":
            confidence_pct = min(confidence_pct + 8.0, 98.5)
        if "crushing_chest_pain" in user_symptoms_set and disease_id == "Acute Coronary Syndrome / Angina":
            confidence_pct = min(confidence_pct + 12.0, 99.0)

        # Determine severity badge
        base_sev = disease_data["base_severity"]
        final_sev = base_sev
        if detected_red_flags or severity_rating >= 8:
            if "Emergency" not in final_sev:
                final_sev = f"High / Urgent (Rating {severity_rating}/10)"

        scored_diseases.append({
            "id": disease_id,
            "name": disease_data["name"],
            "icon": disease_data["icon"],
            "category": disease_data["category"],
            "summary": disease_data["summary"],
            "confidence": round(confidence_pct, 1),
            "matched_symptoms": matched_keys,
            "matched_count": len(matched_keys),
            "specialist": disease_data["specialist"],
            "specialist_icon": disease_data["specialist_icon"],
            "specialist_reason": disease_data["specialist_reason"],
            "urgency_level": disease_data["urgency_level"],
            "severity": final_sev,
            "is_pulmonary": disease_data["is_pulmonary"],
            "tests": disease_data["tests"],
            "precautions": disease_data["precautions"],
            "red_flags": disease_data["red_flags"],
        })

    # Sort diseases by confidence score descending
    scored_diseases.sort(key=lambda x: (x["confidence"], x["matched_count"]), reverse=True)

    if not scored_diseases:
        # Fallback for unknown symptom combination
        fallback_pred = {
            "id": "General Consultation Required",
            "name": "General Clinical Evaluation Advised",
            "icon": "🩺",
            "category": "Unspecified Clinical Presentation",
            "summary": "The symptoms entered do not clearly align with a single specific respiratory or organ syndrome.",
            "confidence": 45.0,
            "matched_symptoms": list(user_symptoms_set),
            "matched_count": len(user_symptoms_set),
            "specialist": "General Physician / Internal Medicine",
            "specialist_icon": "🩺",
            "specialist_reason": "A General Physician can conduct physical examination, take vital signs, and order baseline diagnostic tests.",
            "urgency_level": "Routine consultation within 24-48 hours",
            "severity": "Moderate" if severity_rating > 5 else "Mild",
            "is_pulmonary": False,
            "tests": ["Complete Blood Count (CBC)", "Vital Signs Monitoring", "Physician Physical Examination"],
            "precautions": ["Rest adequately and maintain proper hydration.", "Seek immediate care if condition worsens."],
            "red_flags": ["High fever persisting > 3 days", "Severe sudden onset pain or breathlessness"],
        }
        scored_diseases.append(fallback_pred)

    top = scored_diseases[0]
    differentials = scored_diseases[1:4]

    human_symptom_names = [SYMPTOM_LABELS_DICT.get(k, k.replace("_", " ").title()) for k in user_symptoms_set]

    return {
        "success": True,
        "input_symptoms": list(user_symptoms_set),
        "input_symptom_labels": human_symptom_names,
        "duration_days": duration_days,
        "severity_rating": severity_rating,
        "red_flags_detected": detected_red_flags,
        "has_emergency": len(detected_red_flags) > 0 or "Emergency" in top["urgency_level"],
        "top_prediction": top,
        "differential_diagnoses": differentials,
        "all_predictions": scored_diseases,
    }
