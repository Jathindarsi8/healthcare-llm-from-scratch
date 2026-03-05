"""
=============================================================================
DAY 4A: MEDICAL DATA PREPARATION
=============================================================================
Author: Jathin | Healthcare LLM Project

Creates a medical text dataset for training your healthcare LLM.
This includes clinical notes, discharge summaries, medical Q&A,
and PubMed-style abstracts.

How to run:
    python 04a_medical_data.py

Output: medical_text.txt (ready for training)
=============================================================================
"""

import os


def create_medical_dataset():
    """
    Create a medical text dataset from multiple healthcare sources.

    In a real project, you'd download from PubMed, MIMIC, etc.
    This sample dataset demonstrates the patterns your model will learn.

    The text covers:
    - Clinical notes (SOAP format)
    - Discharge summaries
    - Medical Q&A
    - PubMed-style abstracts
    - ICD-10 and CPT documentation
    - Medication information
    - Lab result interpretations
    """

    clinical_notes = """
CLINICAL NOTE
Patient: John Smith
Date: 2024-01-15
Provider: Dr. Sarah Johnson

CHIEF COMPLAINT:
Patient presents with shortness of breath and chest pain for the past two days.

HISTORY OF PRESENT ILLNESS:
This is a 62-year-old male with a past medical history significant for type 2 diabetes mellitus,
hypertension, and hyperlipidemia who presents to the emergency department with progressive
dyspnea on exertion and substernal chest pressure. The patient reports that symptoms began
approximately 48 hours ago and have been gradually worsening. He denies any fever, cough,
or recent illness. He reports compliance with his medications including metformin, lisinopril,
and atorvastatin. He denies any recent changes in diet or activity level.

PAST MEDICAL HISTORY:
1. Type 2 Diabetes Mellitus (ICD-10: E11.65) - diagnosed 2015
2. Essential Hypertension (ICD-10: I10) - diagnosed 2010
3. Hyperlipidemia (ICD-10: E78.5) - diagnosed 2012
4. Obesity (ICD-10: E66.01) - BMI 34.2
5. Osteoarthritis of bilateral knees (ICD-10: M17.0)

MEDICATIONS:
1. Metformin 1000mg PO BID
2. Lisinopril 20mg PO daily
3. Atorvastatin 40mg PO QHS
4. Aspirin 81mg PO daily
5. Acetaminophen 500mg PO PRN for pain

ALLERGIES: Penicillin (rash), Sulfa drugs (hives)

SOCIAL HISTORY:
Former smoker, quit 5 years ago. Denies alcohol or illicit drug use.
Retired teacher. Lives with wife. Independent in all activities of daily living.

FAMILY HISTORY:
Father: Myocardial infarction at age 58, deceased
Mother: Type 2 diabetes, hypertension, alive at age 85
Brother: Coronary artery disease, status post CABG at age 55

REVIEW OF SYSTEMS:
Constitutional: No fever, chills, or weight loss
HEENT: No headache, vision changes, or sore throat
Cardiovascular: Chest pressure, dyspnea on exertion, no palpitations
Respiratory: Shortness of breath, no cough, no hemoptysis
Gastrointestinal: No nausea, vomiting, or abdominal pain
Musculoskeletal: Chronic bilateral knee pain, stable
Neurological: No dizziness, syncope, or focal weakness

PHYSICAL EXAMINATION:
Vitals: BP 158/92, HR 88, RR 22, Temp 98.6F, SpO2 94% on room air
General: Alert, oriented, mild respiratory distress
HEENT: Normocephalic, atraumatic. PERRL. Oropharynx clear.
Neck: No JVD. No lymphadenopathy. No carotid bruits.
Cardiovascular: Regular rate and rhythm. S1 S2 normal. No murmurs, rubs, or gallops.
Respiratory: Bilateral basilar crackles. No wheezing. Decreased breath sounds at bases.
Abdomen: Soft, non-tender, non-distended. Normal bowel sounds.
Extremities: Trace bilateral lower extremity edema. No cyanosis. Pulses 2+ bilaterally.
Neurological: Alert and oriented x3. Cranial nerves intact. Motor strength 5/5 throughout.

DIAGNOSTIC WORKUP:
1. ECG: Normal sinus rhythm, no acute ST changes
2. Chest X-ray: Bilateral pleural effusions, mild cardiomegaly
3. BNP: 890 pg/mL (elevated, normal less than 100)
4. Troponin I: 0.02 ng/mL (normal, less than 0.04)
5. CBC: WBC 8.2, Hgb 13.1, Plt 245
6. BMP: Na 138, K 4.2, Cr 1.4, BUN 28, Glucose 186
7. HbA1c: 8.2% (elevated, target less than 7%)

ASSESSMENT AND PLAN:

1. Acute decompensated heart failure (ICD-10: I50.21)
   - IV furosemide 40mg now, then 20mg IV BID
   - Strict I&O monitoring
   - Daily weights
   - Fluid restriction 1.5L per day
   - Echocardiogram ordered
   - Cardiology consultation

2. Type 2 Diabetes Mellitus, poorly controlled (ICD-10: E11.65)
   - HbA1c 8.2%, above target
   - Continue metformin 1000mg BID
   - Add glipizide 5mg PO daily
   - Diabetic diet education
   - Endocrinology follow-up recommended

3. Hypertension, uncontrolled (ICD-10: I10)
   - BP 158/92, above goal
   - Increase lisinopril to 40mg PO daily
   - Add amlodipine 5mg PO daily
   - Low sodium diet

4. Acute kidney injury on chronic kidney disease (ICD-10: N17.9)
   - Creatinine 1.4, baseline 1.1
   - Hold NSAIDs
   - Monitor renal function daily
   - Nephrology consultation if worsening

Disposition: Admit to telemetry floor
Code Status: Full code
DVT Prophylaxis: Heparin 5000 units SQ TID

CLINICAL NOTE
Patient: Maria Garcia
Date: 2024-01-16
Provider: Dr. Michael Chen

CHIEF COMPLAINT:
Routine follow-up for diabetes management.

HISTORY OF PRESENT ILLNESS:
This is a 55-year-old female with a history of type 2 diabetes mellitus diagnosed
eight years ago, presenting for routine quarterly follow-up. Patient reports good
compliance with medications and dietary modifications. She has been checking her
blood glucose at home and reports fasting levels between 110 and 140 mg/dL. She
denies any episodes of hypoglycemia, polyuria, polydipsia, or blurred vision.
She reports mild numbness in bilateral feet that has been stable for the past year.

CURRENT MEDICATIONS:
1. Metformin 500mg PO BID
2. Sitagliptin 100mg PO daily
3. Lisinopril 10mg PO daily
4. Atorvastatin 20mg PO QHS

PHYSICAL EXAMINATION:
Vitals: BP 132/78, HR 72, RR 16, Temp 98.4F, BMI 29.8
General: Well-appearing, no acute distress
Extremities: No edema. Monofilament testing decreased bilateral feet.
Skin: No ulcers or lesions on feet bilaterally.

LABORATORY RESULTS:
HbA1c: 7.4% (improved from 8.1% three months ago)
Fasting glucose: 128 mg/dL
Lipid panel: Total cholesterol 195, LDL 110, HDL 48, Triglycerides 165
Creatinine: 0.9, eGFR greater than 60
Urine microalbumin: 45 mg/L (mildly elevated)

ASSESSMENT AND PLAN:

1. Type 2 Diabetes Mellitus (ICD-10: E11.65)
   - HbA1c improved to 7.4% from 8.1%
   - Continue current regimen
   - Goal HbA1c less than 7%
   - Continue home glucose monitoring
   - Reinforce dietary counseling

2. Diabetic peripheral neuropathy (ICD-10: E11.42)
   - Stable numbness bilateral feet
   - Start gabapentin 300mg PO QHS
   - Annual podiatry referral
   - Foot care education provided

3. Microalbuminuria (ICD-10: E11.21)
   - Urine microalbumin 45 mg/L
   - Continue lisinopril for renal protection
   - Recheck in 3 months
   - Avoid nephrotoxic agents

Follow-up: 3 months
Return precautions: Hypoglycemia symptoms, worsening neuropathy, foot wounds

CLINICAL NOTE
Patient: Robert Williams
Date: 2024-01-17
Provider: Dr. Lisa Park

CHIEF COMPLAINT:
New onset atrial fibrillation detected on routine ECG.

HISTORY OF PRESENT ILLNESS:
This is a 71-year-old male with a history of coronary artery disease status post
percutaneous coronary intervention to the left anterior descending artery two years ago,
hypertension, and chronic obstructive pulmonary disease who presents for routine cardiology
follow-up. An ECG performed today shows new onset atrial fibrillation with a ventricular
rate of 92 beats per minute. The patient reports occasional palpitations over the past
month but denies chest pain, syncope, or worsening dyspnea. He has been compliant with
his medications including aspirin, clopidogrel, metoprolol, and atorvastatin.

PHYSICAL EXAMINATION:
Vitals: BP 142/86, HR 92 irregular, RR 18, SpO2 96% on room air
Cardiovascular: Irregularly irregular rhythm, no murmurs
Respiratory: Mild bilateral wheezing, no crackles

DIAGNOSTIC RESULTS:
ECG: Atrial fibrillation, ventricular rate 92, no acute ST-T changes
Echocardiogram: LVEF 45%, mild left atrial enlargement, no significant valvular disease
TSH: 2.4 (normal)
CBC: Within normal limits
BMP: Within normal limits

ASSESSMENT AND PLAN:

1. New onset atrial fibrillation (ICD-10: I48.91)
   - CHA2DS2-VASc score: 4 (age, hypertension, vascular disease, male)
   - Start apixaban 5mg PO BID for stroke prevention
   - Increase metoprolol to 50mg PO BID for rate control
   - Target heart rate less than 80 at rest
   - Holter monitor for 48 hours
   - Discuss rhythm versus rate control strategy

2. Coronary artery disease (ICD-10: I25.10)
   - Stable, no anginal symptoms
   - Continue dual antiplatelet therapy
   - Will transition from clopidogrel to apixaban management
   - Continue atorvastatin 80mg

3. COPD, stable (ICD-10: J44.1)
   - Continue tiotropium and albuterol PRN
   - Spirometry stable
   - Annual influenza and pneumococcal vaccination up to date

DISCHARGE SUMMARY
Patient: James Anderson
Admission Date: 2024-01-10
Discharge Date: 2024-01-14
Attending: Dr. Amanda Torres

PRINCIPAL DIAGNOSIS:
Acute ST-elevation myocardial infarction of the anterior wall (ICD-10: I21.09)

SECONDARY DIAGNOSES:
1. Type 2 Diabetes Mellitus (ICD-10: E11.9)
2. Essential Hypertension (ICD-10: I10)
3. Tobacco use disorder (ICD-10: F17.210)
4. Acute kidney injury, resolved (ICD-10: N17.9)

HOSPITAL COURSE:
The patient is a 58-year-old male who presented to the emergency department with
acute onset severe substernal chest pain radiating to the left arm with diaphoresis.
Initial ECG showed ST elevation in leads V1 through V4 consistent with anterior
STEMI. The patient was taken emergently to the cardiac catheterization laboratory
where angiography revealed a 99 percent occlusion of the proximal left anterior
descending artery. Successful percutaneous coronary intervention was performed with
placement of a drug-eluting stent with restoration of TIMI 3 flow.

Post-procedure course was uncomplicated. Troponin peaked at 45.2 ng/mL.
Echocardiogram showed LVEF of 40 percent with anterior wall hypokinesis.
The patient was started on guideline-directed medical therapy including dual
antiplatelet therapy, beta-blocker, ACE inhibitor, and high-intensity statin.
Blood glucose levels were elevated during admission with a HbA1c of 9.1 percent
indicating undiagnosed diabetes mellitus. Endocrinology was consulted and the
patient was started on metformin with plans for outpatient diabetes management.

DISCHARGE MEDICATIONS:
1. Aspirin 81mg PO daily
2. Ticagrelor 90mg PO BID
3. Metoprolol succinate 25mg PO daily
4. Lisinopril 5mg PO daily
5. Atorvastatin 80mg PO QHS
6. Metformin 500mg PO BID
7. Nitroglycerin 0.4mg SL PRN chest pain

DISCHARGE INSTRUCTIONS:
1. Cardiac rehabilitation referral provided
2. Follow-up with cardiology in 2 weeks
3. Follow-up with endocrinology in 4 weeks
4. Smoking cessation counseling provided
5. Low sodium, diabetic diet
6. Avoid heavy lifting greater than 10 pounds for 2 weeks
7. Return to ED for chest pain, shortness of breath, or arm weakness

CONDITION AT DISCHARGE: Stable, improved

"""

    medical_knowledge = """
MEDICAL KNOWLEDGE BASE

Type 2 Diabetes Mellitus is a chronic metabolic disorder characterized by insulin resistance
and relative insulin deficiency resulting in hyperglycemia. The condition affects approximately
37 million Americans and is a leading cause of cardiovascular disease, kidney failure,
blindness, and lower limb amputation. Risk factors include obesity, physical inactivity,
family history, and advancing age. The diagnosis is established by fasting plasma glucose
greater than or equal to 126 mg/dL, HbA1c greater than or equal to 6.5 percent, or
two-hour plasma glucose greater than or equal to 200 mg/dL during an oral glucose
tolerance test. First-line treatment includes lifestyle modifications and metformin therapy.
Additional pharmacologic agents include sulfonylureas, DPP-4 inhibitors, GLP-1 receptor
agonists, SGLT2 inhibitors, and insulin. The American Diabetes Association recommends
an HbA1c target of less than 7 percent for most adults. Regular monitoring should include
quarterly HbA1c measurements, annual dilated eye examinations, annual foot examinations,
and periodic assessment of renal function and urine albumin excretion.

Heart failure is a clinical syndrome resulting from structural or functional impairment
of ventricular filling or ejection of blood. It affects approximately 6.2 million adults
in the United States. Heart failure is classified by left ventricular ejection fraction
into heart failure with reduced ejection fraction (HFrEF, LVEF less than or equal to
40 percent), heart failure with mildly reduced ejection fraction (HFmrEF, LVEF 41 to
49 percent), and heart failure with preserved ejection fraction (HFpEF, LVEF greater
than or equal to 50 percent). Common etiologies include coronary artery disease,
hypertension, valvular heart disease, and cardiomyopathies. Symptoms include dyspnea,
orthopnea, paroxysmal nocturnal dyspnea, fatigue, and lower extremity edema. Diagnosis
involves clinical assessment, BNP or NT-proBNP measurement, echocardiography, and
chest radiography. Treatment of HFrEF includes ACE inhibitors or ARBs, beta-blockers,
mineralocorticoid receptor antagonists, and SGLT2 inhibitors as guideline-directed
medical therapy. Diuretics are used for volume management. Device therapy including
implantable cardioverter-defibrillators and cardiac resynchronization therapy may be
indicated in selected patients.

Chronic obstructive pulmonary disease is a common preventable and treatable disease
characterized by persistent respiratory symptoms and airflow limitation due to airway
and alveolar abnormalities usually caused by significant exposure to noxious particles
or gases. Cigarette smoking is the most common risk factor. The diagnosis is confirmed
by spirometry showing a post-bronchodilator FEV1/FVC ratio less than 0.70. COPD is
classified by severity based on FEV1 percentage predicted. Treatment includes smoking
cessation, pulmonary rehabilitation, bronchodilators including short-acting and long-acting
beta-agonists and anticholinergics, inhaled corticosteroids for patients with frequent
exacerbations, and supplemental oxygen for patients with severe hypoxemia. Acute
exacerbations are managed with short-acting bronchodilators, systemic corticosteroids,
and antibiotics when bacterial infection is suspected.

Hypertension is defined as systolic blood pressure greater than or equal to 130 mmHg
or diastolic blood pressure greater than or equal to 80 mmHg based on the 2017
ACC/AHA guidelines. It is a major risk factor for cardiovascular disease, stroke,
chronic kidney disease, and heart failure. Evaluation includes confirming elevated
blood pressure on multiple occasions, assessing for target organ damage, screening
for secondary causes when indicated, and cardiovascular risk assessment. First-line
pharmacologic treatment includes thiazide diuretics, calcium channel blockers,
ACE inhibitors, and angiotensin receptor blockers. Most patients require two or more
antihypertensive agents to achieve blood pressure goals. Lifestyle modifications
including sodium restriction, regular physical activity, weight management, and
moderation of alcohol intake are recommended for all patients.

Atrial fibrillation is the most common sustained cardiac arrhythmia, affecting
approximately 2.7 million Americans. It is characterized by rapid and irregular
activation of the atria leading to an irregularly irregular ventricular response.
Risk factors include advancing age, hypertension, heart failure, valvular heart
disease, obesity, and obstructive sleep apnea. Atrial fibrillation significantly
increases the risk of stroke. Stroke risk is assessed using the CHA2DS2-VASc
scoring system. Anticoagulation with direct oral anticoagulants or warfarin is
recommended for patients with a CHA2DS2-VASc score of 2 or greater in men and
3 or greater in women. Rate control with beta-blockers or calcium channel blockers
is the initial strategy for most patients. Rhythm control with antiarrhythmic drugs
or catheter ablation may be considered for patients with persistent symptoms despite
adequate rate control.

Acute myocardial infarction occurs when coronary blood flow is abruptly reduced
or completely occluded, resulting in myocardial necrosis. ST-elevation myocardial
infarction represents complete coronary occlusion requiring emergent reperfusion
therapy. The diagnosis is based on clinical presentation including chest pain,
ECG findings showing ST-segment elevation, and elevated cardiac biomarkers
including troponin. Primary percutaneous coronary intervention is the preferred
reperfusion strategy when available within 120 minutes of first medical contact.
Fibrinolytic therapy is an alternative when PCI is not available in a timely manner.
Adjunctive therapy includes dual antiplatelet therapy with aspirin and a P2Y12
inhibitor, anticoagulation, beta-blockers, ACE inhibitors, and high-intensity
statin therapy. Cardiac rehabilitation is recommended for all patients following
acute myocardial infarction.

Chronic kidney disease is defined as abnormalities of kidney structure or function
present for greater than 3 months with implications for health. It is classified
by cause, GFR category, and albuminuria category. The GFR categories range from
G1 with normal or high GFR greater than 90 to G5 with GFR less than 15 indicating
kidney failure. Risk factors include diabetes mellitus, hypertension, cardiovascular
disease, obesity, and family history. Management focuses on treating the underlying
cause, slowing progression with ACE inhibitors or ARBs and SGLT2 inhibitors,
managing complications including anemia, mineral and bone disorder, metabolic acidosis,
and hyperkalemia, and preparing for renal replacement therapy when indicated.

The assessment of a patient in the emergency department begins with the primary survey
evaluating airway, breathing, circulation, disability, and exposure. Vital signs
including blood pressure, heart rate, respiratory rate, temperature, and oxygen
saturation are obtained. A focused history and physical examination are performed
based on the chief complaint. Diagnostic workup may include laboratory studies,
imaging, and electrocardiography. Risk stratification tools are used to guide
disposition decisions. Critical diagnoses that must be considered include acute
coronary syndrome, pulmonary embolism, aortic dissection, tension pneumothorax,
and stroke. Time-sensitive interventions are prioritized based on clinical urgency.

Clinical documentation must include the chief complaint, history of present illness,
past medical history, medications, allergies, social history, family history, review
of systems, physical examination findings, diagnostic results, assessment, and plan.
The assessment should include a differential diagnosis and clinical reasoning supporting
the working diagnosis. The plan should address each problem identified and include
specific orders for medications, monitoring, consultations, and follow-up. Medical
decision making complexity is determined by the number and complexity of problems
addressed, the amount and complexity of data reviewed and analyzed, and the risk
of complications, morbidity, and mortality of patient management decisions.

"""

    # Repeat and combine to create a larger training corpus
    full_text = ""
    for i in range(8):
        full_text += clinical_notes
    for i in range(12):
        full_text += medical_knowledge

    return full_text


def main():
    print("=" * 65)
    print("  DAY 4A: MEDICAL DATA PREPARATION")
    print("  Healthcare LLM Project")
    print("=" * 65)

    print("\n  Creating medical text dataset...")
    text = create_medical_dataset()

    # Save to file
    output_path = 'medical_text.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Stats
    chars = len(text)
    words = len(text.split())
    lines = text.count('\n')
    unique_chars = len(set(text))

    print(f"\n  Dataset created: {output_path}")
    print(f"  Characters:  {chars:,}")
    print(f"  Words:       {words:,}")
    print(f"  Lines:       {lines:,}")
    print(f"  Unique chars: {unique_chars}")

    # Show sample
    print(f"\n  First 300 characters:")
    print(f"  {text[:300]}")

    # Compare with Shakespeare
    if os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            shakespeare = f.read()
        print(f"\n  Comparison:")
        print(f"  {'':20s} | {'Shakespeare':>15s} | {'Medical':>15s}")
        print(f"  {'-'*55}")
        print(f"  {'Characters':<20s} | {len(shakespeare):>15,} | {chars:>15,}")
        print(f"  {'Words':<20s} | {len(shakespeare.split()):>15,} | {words:>15,}")
        print(f"  {'Unique chars':<20s} | {len(set(shakespeare)):>15} | {unique_chars:>15}")

    print(f"""
  Dataset ready! Now run:
    python 04b_medical_gpt.py

  This will train your GPT on medical text and compare
  the output with your Shakespeare-trained model.
""")


if __name__ == '__main__':
    main()