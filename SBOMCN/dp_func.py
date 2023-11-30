


import pandas as pd
import numpy as np

# preprocessing tools



# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# warnings
import warnings
warnings.filterwarnings('ignore')

#saving model 
import pickle
import statistics
#List of Diseases is listed in list disease.

disease=['Immune thrombocytopenia','Fabri Disease','Cystic Fibrosis','Wilson\'s Disease','Pulmonary Fibrosis',
         'MNGIE(Mitochondrial Neurogastrointestinal Encephalomyopathy)','Dengue','Chicken pox','Lung Cancer','Tuberculosis']
#List of the symptoms is listed here in list l1.

l1=['Easy_or_excessive_Bruising','Easy_or_excessive_Bleeding','Petechiae(small_red_or_purple_dots_on_the_skin)',
    'Fatigue','Enlarged_spleen','Pain_tingling_burning_sensations_in_the_hands_and_feet',
    'Skin_rash','Gastrointestinal_symptoms((diarrhea_and(nausea_or_vomiting))_or_(any_symptom_except_diarrhea))',
    'Eye_problems(such_as_cloudiness_or_a_corneal_opacity)',
    'Hearing_loss_or_Heart_problems_or_Kidney_problems',
    'Only_Prolonged_diarrhea','Wet_cough','Frequent_lung(or_sinus)infections','Poor_growth_and_failure_to_gain_weight',
    'Male_infertility','Shortness_of_breath','Loss_of_appetite_&_abdominal_pain',
    'Yellowing_of_the_skin(or_the_whites_of_the_eye)(jaundice)','Fluid_buildup_in_the_legs_or_abdomen',
    'Golden_brown_eye_discoloration(Kayser-Fleischer_rings)','Uncontrolled_movements_or_muscle_stiffness_or_tremors',
    'Problems_with_speech,swallowing_or_physical_coordination','Dry_cough','Weight_loss','Aching_muscles_&_joints',
    'Widening_rounding_of_the_tips_of_fingers/toes(clubbing)','Weight_and_muscle_loss',
    'Drooping_of_the_eyelids_and_tingling_sensations_in_the_limbs','Digestive_problems','Encephalopathy(brain_damage)',
    'Progressive_difficulty_speaking_and_swallowing','High_fever_104F(40C)','Severe_Headache','Muscle,bone_or_joint_pain',
    'Nausea_or_Vomiting(fever)','Pain_behind_the_eyes','Swollen_glands','Itchy(blister_rash)','Fever','Loss_of_appetite',
    'Headache','Chronic_cough_or_Hoarseness','Chest_pain','Unexplained_weight_loss','Cough_with_blood','Night_sweats_or_chills']




v1 = ["Easy_or_excessive_Bruising","Easy_or_excessive_Bleeding","Petechiae(small_red_or_purple_dots_on_the_skin)","Fatigue","Enlarged_spleen"]
#print(v1)


# selection or core function
def select_best(t):
    #print('prev',t)
    d={}
    l=[]
    for i in t:
        if i not in d:
            d[i]=1
            l.append(i)
        else:
            d[i]+=1
    l1=[]
    for i in l:
        if d[i]==1:
            d.pop(i)
        else:
            l1.append(i)

    #print('end',l1)
    return l1


# Main inner function
def pred_dis(val,final_models):
    t=[]
    for k, v in final_models.items():
        t.append(int(v.predict(val)))
    best=select_best(t)
    l=[]
    for i in range(len(disease)):
        for j in best:
            if (i==j):
                l.append(disease[i])            
    if(len(l)==1):
        return "Mostly: "+l[0]
    elif(len(l)==0):
        return "Miscelleneous symptoms... disease can not be predicted"
    elif(len(l)==2):
        return "Maybe : "+" , ".join(l)
    else:
        return "Maybe : "+" , ".join(l)

# encode inputs
def encode_value(v1):
    data={}
    for i in l1:
        if i in v1:
            data[i]=1
        else:
            data[i]=0
    data_list = [data]
    df_values = pd.DataFrame(data_list)
    return df_values

# Main outer function
def predict_disease(symptoms,m):
    s1 = symptoms.split(",")
    s2=encode_value(s1)
    s3=pred_dis(s2,m)
    return s3

#unit test 1
s="Easy_or_excessive_Bruising,Easy_or_excessive_Bleeding,Petechiae(small_red_or_purple_dots_on_the_skin),Fatigue,Enlarged_spleen"
with open('final_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
print('---------------------------------------------------------------------------------------------')
print()
m=model_dict
print("UNIT TEST1   --> dp_func.py")
print("Symptoms:",s)
print("Output:",predict_disease(s,m))
print('---------------------------------------------------------------------------------------------')
print()
