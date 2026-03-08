"""
=============================================================================
DAY 6A: DOWNLOAD REAL PUBMED MEDICAL DATA
=============================================================================
Author: Jathin | Healthcare LLM Project

Downloads real medical abstracts from PubMed using the NCBI E-utilities API.
This is FREE and doesn't require registration.

How to run:
    pip install requests
    python 06a_download_pubmed.py

Output: pubmed_abstracts.txt (real medical text for training)
=============================================================================
"""

import os
import time
import json

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
# PUBMED API DOWNLOADER
# =============================================================================

class PubMedDownloader:
    """
    Downloads abstracts from PubMed using NCBI E-utilities.

    E-utilities is a FREE API provided by the National Library of Medicine.
    No API key required for basic use (limited to 3 requests/second).
    With an API key: 10 requests/second.

    How it works:
    1. esearch: Search PubMed → get list of article IDs
    2. efetch: Fetch article details (title, abstract) by ID
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.all_abstracts = []

    def search(self, query, max_results=500):
        """Search PubMed and return article IDs."""
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            ids = data.get('esearchresult', {}).get('idlist', [])
            return ids
        except Exception as e:
            print(f"    Search error for '{query}': {e}")
            return []

    def fetch_abstracts(self, ids):
        """Fetch abstracts for a list of PubMed IDs."""
        if not ids:
            return []

        # Fetch in batches of 100
        abstracts = []
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            url = f"{self.BASE_URL}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'rettype': 'abstract',
                'retmode': 'text'
            }

            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                text = response.text

                # Split into individual abstracts
                parts = text.split('\n\n\n')
                for part in parts:
                    cleaned = part.strip()
                    if len(cleaned) > 200:  # Only keep substantial abstracts
                        abstracts.append(cleaned)

                time.sleep(0.4)  # Rate limit: max 3 requests/second

            except Exception as e:
                print(f"    Fetch error: {e}")
                continue

        return abstracts

    def download_topic(self, query, max_results=500):
        """Download abstracts for a specific medical topic."""
        print(f"    Searching: '{query}' (max {max_results})...")
        ids = self.search(query, max_results)
        print(f"    Found {len(ids)} articles. Downloading abstracts...")
        abstracts = self.fetch_abstracts(ids)
        print(f"    Got {len(abstracts)} abstracts")
        self.all_abstracts.extend(abstracts)
        return len(abstracts)


# =============================================================================
# MEDICAL TOPICS TO DOWNLOAD
# =============================================================================

MEDICAL_TOPICS = [
    # Core conditions (your BCBS expertise)
    ("type 2 diabetes mellitus treatment", 500),
    ("heart failure management guidelines", 500),
    ("hypertension clinical outcomes", 500),
    ("chronic kidney disease progression", 400),
    ("COPD exacerbation treatment", 400),
    ("atrial fibrillation anticoagulation", 400),
    ("acute myocardial infarction intervention", 400),

    # Clinical decision making
    ("clinical decision making emergency", 300),
    ("medical diagnosis differential", 300),
    ("patient assessment clinical notes", 300),

    # Healthcare operations (relevant to your work)
    ("health insurance claims processing", 200),
    ("medical coding ICD accuracy", 200),
    ("healthcare quality improvement", 200),
    ("clinical documentation improvement", 200),

    # Medications
    ("metformin diabetes outcomes", 300),
    ("statin therapy cardiovascular", 300),
    ("antibiotic prescribing guidelines", 200),

    # Lab and diagnostics
    ("laboratory testing clinical interpretation", 200),
    ("hemoglobin A1c diabetes monitoring", 200),
    ("ECG interpretation cardiac", 200),
]


# =============================================================================
# FALLBACK: Generate comprehensive medical text if download fails
# =============================================================================

def generate_fallback_data():
    """Generate a large, realistic medical dataset if PubMed download fails."""
    print("\n  Generating comprehensive medical text dataset...")

    conditions = [
        {
            'name': 'Type 2 Diabetes Mellitus',
            'icd': 'E11',
            'symptoms': ['polyuria', 'polydipsia', 'fatigue', 'blurred vision', 'weight loss', 'numbness in extremities'],
            'labs': ['HbA1c', 'fasting glucose', 'oral glucose tolerance test', 'C-peptide', 'lipid panel', 'urine microalbumin', 'serum creatinine'],
            'meds': ['metformin', 'glipizide', 'sitagliptin', 'empagliflozin', 'liraglutide', 'insulin glargine', 'pioglitazone'],
            'complications': ['diabetic nephropathy', 'diabetic retinopathy', 'peripheral neuropathy', 'cardiovascular disease', 'diabetic foot ulcer'],
        },
        {
            'name': 'Congestive Heart Failure',
            'icd': 'I50',
            'symptoms': ['dyspnea on exertion', 'orthopnea', 'paroxysmal nocturnal dyspnea', 'lower extremity edema', 'fatigue', 'weight gain'],
            'labs': ['BNP', 'NT-proBNP', 'troponin', 'BMP', 'CBC', 'chest X-ray', 'echocardiogram'],
            'meds': ['lisinopril', 'carvedilol', 'furosemide', 'spironolactone', 'sacubitril-valsartan', 'digoxin', 'hydralazine'],
            'complications': ['cardiogenic shock', 'pulmonary edema', 'renal failure', 'arrhythmia', 'thromboembolism'],
        },
        {
            'name': 'Essential Hypertension',
            'icd': 'I10',
            'symptoms': ['headache', 'dizziness', 'visual changes', 'chest pain', 'shortness of breath', 'often asymptomatic'],
            'labs': ['blood pressure monitoring', 'BMP', 'urinalysis', 'lipid panel', 'ECG', 'echocardiogram', 'renal ultrasound'],
            'meds': ['lisinopril', 'amlodipine', 'hydrochlorothiazide', 'losartan', 'metoprolol', 'chlorthalidone', 'valsartan'],
            'complications': ['stroke', 'myocardial infarction', 'heart failure', 'chronic kidney disease', 'retinopathy', 'aortic dissection'],
        },
        {
            'name': 'Chronic Obstructive Pulmonary Disease',
            'icd': 'J44',
            'symptoms': ['chronic cough', 'sputum production', 'dyspnea', 'wheezing', 'chest tightness', 'exercise intolerance'],
            'labs': ['spirometry', 'FEV1/FVC ratio', 'chest X-ray', 'CT scan', 'arterial blood gas', 'alpha-1 antitrypsin', 'pulse oximetry'],
            'meds': ['albuterol', 'ipratropium', 'tiotropium', 'fluticasone', 'budesonide', 'roflumilast', 'azithromycin'],
            'complications': ['acute exacerbation', 'pneumonia', 'respiratory failure', 'cor pulmonale', 'pneumothorax'],
        },
        {
            'name': 'Chronic Kidney Disease',
            'icd': 'N18',
            'symptoms': ['fatigue', 'decreased urine output', 'edema', 'nausea', 'pruritus', 'muscle cramps', 'confusion'],
            'labs': ['serum creatinine', 'eGFR', 'BUN', 'urine albumin-to-creatinine ratio', 'CBC', 'calcium', 'phosphorus', 'PTH'],
            'meds': ['lisinopril', 'losartan', 'dapagliflozin', 'erythropoietin', 'sevelamer', 'calcitriol', 'sodium bicarbonate'],
            'complications': ['end-stage renal disease', 'hyperkalemia', 'metabolic acidosis', 'anemia', 'mineral bone disorder', 'cardiovascular disease'],
        },
        {
            'name': 'Atrial Fibrillation',
            'icd': 'I48',
            'symptoms': ['palpitations', 'irregular heartbeat', 'dyspnea', 'fatigue', 'dizziness', 'syncope', 'chest discomfort'],
            'labs': ['ECG', 'echocardiogram', 'Holter monitor', 'TSH', 'CBC', 'BMP', 'coagulation studies'],
            'meds': ['apixaban', 'rivaroxaban', 'warfarin', 'metoprolol', 'diltiazem', 'amiodarone', 'flecainide'],
            'complications': ['stroke', 'systemic embolism', 'heart failure', 'cognitive decline', 'reduced quality of life'],
        },
        {
            'name': 'Acute Coronary Syndrome',
            'icd': 'I21',
            'symptoms': ['chest pain', 'diaphoresis', 'nausea', 'dyspnea', 'arm pain', 'jaw pain', 'epigastric discomfort'],
            'labs': ['troponin', 'ECG', 'CK-MB', 'BNP', 'chest X-ray', 'coronary angiography', 'echocardiogram'],
            'meds': ['aspirin', 'ticagrelor', 'heparin', 'nitroglycerin', 'metoprolol', 'atorvastatin', 'lisinopril'],
            'complications': ['cardiogenic shock', 'arrhythmia', 'heart failure', 'pericarditis', 'ventricular septal rupture'],
        },
        {
            'name': 'Major Depressive Disorder',
            'icd': 'F33',
            'symptoms': ['depressed mood', 'anhedonia', 'insomnia', 'fatigue', 'poor concentration', 'appetite changes', 'psychomotor retardation'],
            'labs': ['PHQ-9 screening', 'TSH', 'CBC', 'vitamin D', 'B12', 'folate', 'comprehensive metabolic panel'],
            'meds': ['sertraline', 'escitalopram', 'fluoxetine', 'venlafaxine', 'bupropion', 'duloxetine', 'mirtazapine'],
            'complications': ['suicidal ideation', 'functional impairment', 'substance abuse', 'cardiovascular disease', 'cognitive decline'],
        },
    ]

    texts = []

    for cond in conditions:
        # Generate clinical notes
        for i in range(15):
            age = 35 + (i * 3) % 45
            gender = 'male' if i % 2 == 0 else 'female'
            symp = cond['symptoms']
            labs = cond['labs']
            meds = cond['meds']
            comps = cond['complications']

            note = f"""CLINICAL NOTE
Patient: {age}-year-old {gender}
Diagnosis: {cond['name']} (ICD-10: {cond['icd']})

CHIEF COMPLAINT:
Patient presents with {symp[i % len(symp)]} and {symp[(i+1) % len(symp)]}.

HISTORY OF PRESENT ILLNESS:
This is a {age}-year-old {gender} with a known history of {cond['name']} who presents for evaluation of worsening {symp[i % len(symp)]}. The patient reports symptoms have been progressive over the past {i % 4 + 1} weeks. The patient denies any recent changes in medication compliance. Current medications include {meds[i % len(meds)]} and {meds[(i+1) % len(meds)]}.

PAST MEDICAL HISTORY:
1. {cond['name']} (ICD-10: {cond['icd']}) - diagnosed {2 + i % 8} years ago
2. Essential Hypertension (ICD-10: I10)
3. Hyperlipidemia (ICD-10: E78.5)

PHYSICAL EXAMINATION:
Vitals: Blood pressure {120 + i % 40}/{70 + i % 20} mmHg, Heart rate {60 + i % 30} bpm, Temperature 98.{i % 9}F, SpO2 {94 + i % 5}% on room air.
General: Alert and oriented, {['no acute distress', 'mild distress', 'appears comfortable'][i % 3]}.

DIAGNOSTIC WORKUP:
{labs[i % len(labs)]}: Results reviewed and interpreted in clinical context.
{labs[(i+1) % len(labs)]}: Obtained and pending final interpretation.
Additional studies ordered as clinically indicated.

ASSESSMENT AND PLAN:

1. {cond['name']} (ICD-10: {cond['icd']})
   - Current presentation consistent with {['stable disease', 'disease progression', 'acute exacerbation'][i % 3]}
   - {'Continue' if i % 3 == 0 else 'Adjust'} {meds[i % len(meds)]}
   - {'Add' if i % 2 == 0 else 'Consider adding'} {meds[(i+2) % len(meds)]}
   - Monitor for {comps[i % len(comps)]}
   - Follow-up in {[2, 4, 6, 8, 12][i % 5]} weeks
   - Patient counseled on lifestyle modifications including diet, exercise, and medication adherence

2. Preventive Care
   - Age-appropriate cancer screening discussed
   - Immunizations reviewed and updated as indicated
   - Smoking cessation counseling {'provided' if i % 3 == 0 else 'reinforced'}

"""
            texts.append(note)

        # Generate medical knowledge paragraphs
        knowledge = f"""{cond['name']} is a significant clinical condition classified under ICD-10 code {cond['icd']}. Common presenting symptoms include {', '.join(cond['symptoms'][:4])}. Diagnostic evaluation typically involves {', '.join(cond['labs'][:4])}. First-line pharmacotherapy includes {cond['meds'][0]}, with alternatives such as {cond['meds'][1]} and {cond['meds'][2]}. Important complications to monitor include {', '.join(cond['complications'][:3])}. Regular follow-up and monitoring are essential for optimal disease management and prevention of adverse outcomes.

The pathophysiology of {cond['name']} involves complex interactions between genetic predisposition, environmental factors, and pathological processes. Evidence-based guidelines recommend a multidisciplinary approach to management, incorporating pharmacologic and non-pharmacologic interventions. Patient education regarding disease self-management, medication adherence, and lifestyle modifications plays a crucial role in achieving favorable clinical outcomes.

Current research in {cond['name']} focuses on novel therapeutic targets, biomarker discovery for early detection, and precision medicine approaches to individualize treatment strategies. Clinical trials evaluating {cond['meds'][3]} and {cond['meds'][4]} have demonstrated promising results in improving patient outcomes and reducing disease-related complications.

"""
        for _ in range(8):
            texts.append(knowledge)

    return '\n'.join(texts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  DAY 6A: DOWNLOAD REAL PUBMED MEDICAL DATA")
    print("  Healthcare LLM Project — Phase 2")
    print("=" * 65)

    output_file = 'pubmed_medical_data.txt'
    all_text = ""
    total_abstracts = 0

    if HAS_REQUESTS:
        print("\n  Downloading from PubMed (NCBI E-utilities API)...")
        print("  This is FREE — no registration needed.")
        print("  Downloading may take 5-10 minutes...\n")

        downloader = PubMedDownloader()

        for topic, count in MEDICAL_TOPICS:
            num = downloader.download_topic(topic, count)
            total_abstracts += num
            time.sleep(1)  # Be nice to the API

        if downloader.all_abstracts:
            all_text = '\n\n'.join(downloader.all_abstracts)
            print(f"\n  Downloaded {total_abstracts} abstracts from PubMed!")
        else:
            print("\n  PubMed download returned no results. Using fallback data...")
            all_text = generate_fallback_data()
    else:
        print("\n  'requests' library not found.")
        print("  Install it with: pip install requests")
        print("  Or continuing with generated medical data...\n")
        all_text = generate_fallback_data()

    # If PubMed data is too small, supplement with generated data
    if len(all_text) < 500000:
        print("  Supplementing with additional medical text...")
        fallback = generate_fallback_data()
        all_text = all_text + '\n\n' + fallback

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text)

    # Stats
    chars = len(all_text)
    words = len(all_text.split())
    lines = all_text.count('\n')

    print(f"\n  Dataset saved: {output_file}")
    print(f"  Size:        {chars:,} characters ({chars/1024/1024:.1f} MB)")
    print(f"  Words:       {words:,}")
    print(f"  Lines:       {lines:,}")

    # Compare with previous datasets
    print(f"\n  Dataset Comparison:")
    print(f"  {'Dataset':<25s} | {'Characters':>15s} | {'Words':>12s}")
    print(f"  {'-'*58}")

    if os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            shk = f.read()
        print(f"  {'Shakespeare (Day 1)':<25s} | {len(shk):>15,} | {len(shk.split()):>12,}")

    if os.path.exists('medical_text.txt'):
        with open('medical_text.txt', 'r', encoding='utf-8') as f:
            med = f.read()
        print(f"  {'Medical sample (Day 4)':<25s} | {len(med):>15,} | {len(med.split()):>12,}")

    print(f"  {'PubMed data (Day 6)':<25s} | {chars:>15,} | {words:>12,}")

    print(f"""
  Next step:
    python 06b_prepare_data.py

  This will clean and analyze the data, then prepare it for training.
""")


if __name__ == '__main__':
    main()