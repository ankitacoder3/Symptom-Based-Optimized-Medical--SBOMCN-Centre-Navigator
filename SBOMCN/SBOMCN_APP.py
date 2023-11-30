from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
#import cap
#from cap import recommend_hospitals     #MCN-CR
import pandas as pd

import dp_func #dp   #


from loc_hos_recommendation import recommend_hospital  #loc
from cost_rating import recommend_hospitals

app = Flask(__name__)

# Load your hospital dataset
data = pd.read_csv("hospital_data_new_merged_15_11.csv")  # Replace with the actual file path

# Load the recommendation function from the pickle file
with open('Cost_rating.pkl', 'rb') as f:
    recommend_hospitals = pickle.load(f)





#----------------------------------------------COMMON------------------------------------------------------------
print('---------------------------------------------------------------------------------------------')
print()

@app.route('/home.html', methods=['GET'])
def home_page():                 
        print('1- RENDERING HOME.HTML')
        return render_template('home.html')
@app.route('/home1.html', methods=['GET'])
def home_page1():                 
        print('1- RENDERING HOME.HTML1')
        return render_template('home1.html')

@app.route('/login.html', methods=['GET'])
def login_1():                 
        print('1- RENDERING LOGIN.HTML')
        return render_template('login.html')

print('FLASK')
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict.html', methods=['GET'])
def predict_page():                 #dropdown
        print('1- RENDERING PREDICT.HTML')
        return render_template('predict.html')

d2=["Immune thrombocytopenia","Fabri Disease","Cystic Fibrosis","Wilson's disease","Pulmonary Fibrosis","Mitochondrial Neurogastrointestinal Encephalomyopathy",
            "Dengue","Chicken pox","Lung cancer","Tuberculosis"]
d1=['Immune thrombocytopenia','Fabri Disease','Cystic Fibrosis','Wilson\'s Disease','Pulmonary Fibrosis',
         'MNGIE(Mitochondrial Neurogastrointestinal Encephalomyopathy)','Dengue','Chicken pox','Lung Cancer','Tuberculosis']
sync_d={}
for i in range(len(d1)):
     sync_d[d1[i]]=d2[i]

@app.route('/findhospital.html', methods=['GET'])
def recommend_page():                 
        print('1- RENDERING RECOMMEND.HTML')
        predicted_class = request.args.get('predicted_class')
        #opl=predicted_class = request.args.get('opl')
        print('i/p-> ',predicted_class)
        c=0
        if(predicted_class!='None'and predicted_class!= None):
            #print(predicted_class,c)
            l=predicted_class.split(': ')
            if(len(l)!=1):
                #print(l)
                l.pop(0)
                l=(l[0]).split(' , ')
                #print(l)
                c=len(l)
                print(l,c)
            c=1
        else:
            l=['']



        if(c>0):
            for i in range(len(l)):
                 if(l[i] not in d2):
                      l[i]=sync_d[l[i]]
    
        return render_template('findhospital.html', p_c=l[0],p_cs=l,c=c)
#----------------------------------------------------DP ----------------------------------------------------------------

with open('final_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

print('MODELS LOADED -- DP')
m=model_dict
print('No. of models:',len(m))
#print("Model dict" ,model_dict) #

#trial- #how to call or use

def trial_dp():
    print("ALL Models")
    for k, v in m.items():
            print(k,v)
    print()

    print('---------------------------------------------------------------------------------------------')
    print()
    print('UNIT TEST2           ---> SBOMCN_APP.py')
    s1="Easy_or_excessive_Bruising,Easy_or_excessive_Bleeding,Petechiae(small_red_or_purple_dots_on_the_skin),Fatigue,Enlarged_spleen"
    print('Symptoms:',s1)
    op=dp_func.predict_disease(s1,m)
    print('Output:',op)

    print('---------------------------------------------------------------------------------------------')
    print()

trial_dp() #

#-----------------------------------------------------------MCN-LOC----------------------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    print('1- At recommend ...')
    disease_name = request.form['diseaseName']
    filter_app= request.form['filter']
    print('1a- Inputs --> Disease:', disease_name,'\t Filter:',filter_app)
    if filter_app=='nearby':
        print("2- procesing for loc...")
        
        Cur=-1

        Cur=request.form['useCurrentLocation']

        option='0'
        if(Cur=="true"):
              option='1'
        elif(Cur=='false'):
              option='2'
        else:
              option='-1'

        print('3- CHOICE BOX APPEARED')
        print('\n4- Inputs2 --> Current Location Setting:',Cur,'\t Option:',option)

        
        if option == '1': #current loc
                result = recommend_hospital(disease_name, option)
                return render_template('loc_output.html', result=result,p1="Bangalore",p2="560100",D=disease_name)
        elif option == '2':
                place_name = request.form['place_name']
                pincode = request.form['pincode']
                result = recommend_hospital(disease_name, option, place_name, pincode)
                print(result,pincode,place_name)
                print("")
                return render_template('loc_output.html', result=result,p1=place_name,p2=pincode,D=disease_name)
        else:
                print('ERROR: NO OPTION SELECTED!!')
                result = []
        return render_template('loc_output.html', result=result,D=disease_name) 
    else: 
         print("error--- wrong filter")

    filter_by = request.form['filter'] 

    if filter_by == 'review':
        result = recommend_hospitals(data, disease_name, 'Rating')
        return render_template('rating_cost.html', result=result, filter_type='RATING',D=disease_name)
    elif filter_by == 'cost':
        result = recommend_hospitals(data, disease_name, 'Cost')
        return render_template('rating_cost.html', result=result, filter_type='COST',D=disease_name)
    else:
        return "Filter type not supported."






#----------------------------------------------------DP ----------------------------------------------------------------

#dict for symptom mapping
print('SYMPTOMS LOADED')
d={}
def init_dp(d):
    d["option1_value1"]="Easy_or_excessive_Bruising"
    d["option2_value1"]="Easy_or_excessive_Bleeding"
    d["option3_value1"]="Petechiae(small_red_or_purple_dots_on_the_skin)"
    d["option4_value1"]="Fatigue"
    d["option5_value1"]="Enlarged_spleen"
    d["option1_value2"]="Pain_tingling_burning_sensations_in_the_hands_and_feet"
    d["option2_value2"]="Skin_rash"
    d["option3_value2"]="Gastrointestinal_symptoms((diarrhea_and(nausea_or_vomiting))_or_(any_symptom_except_diarrhea))"
    d["option4_value2"]="Eye_problems(such_as_cloudiness_or_a_corneal_opacity)"
    d["option5_value2"]="Hearing_loss_or_Heart_problems_or_Kidney_problems"
    d["option3_value3"]="Only_Prolonged_diarrhea"
    d["option1_value3"]="Wet_cough"
    d["option2_value3"]="Frequent_lung(or_sinus)infections"
    d["option4_value3"]="Poor_growth_and_failure_to_gain_weight"
    d["option5_value3"]="Male_infertility"
    d["option1_value5"]="Shortness_of_breath"
    d["option1_value4"]="Loss_of_appetite_&_abdominal_pain"
    d["option2_value4"]="Yellowing_of_the_skin(or_the_whites_of_the_eye)(jaundice)"
    d["option3_value4"]="Fluid_buildup_in_the_legs_or_abdomen"
    d["option4_value4"]="Golden_brown_eye_discoloration(Kayser-Fleischer_rings)"
    d["option5_value4"]="Uncontrolled_movements_or_muscle_stiffness_or_tremors"
    d["option2_value5"]="Dry_cough"
    d["option5_value5"]="Aching_muscles_&_joints"
    d["option1_value6"]="Weight_and_muscle_loss"
    d["option1_value7"]="Severe_Headache"
    d["option1_value8"]="Itchy(blister_rash)"
    d["option1_value9"]="Chronic_cough_or_Hoarseness"
    d["option2_value6"]="Drooping_of_the_eyelids_and_tingling_sensations_in_the_limbs"
    d["option2_value7"]="High_fever_104F(40C)"
    d["option2_value8"]="Fever"
    d["option2_value9"]="Chest_pain"
    d["option3_value6"]="Digestive_problems"
    d["option3_value7"]="Muscle,bone_or_joint_pain"
    d["option3_value9"]="Unexplained_weight_loss"
    d["option4_value5"]="Weight_loss"
    d["option4_value6"]= "Progressive_difficulty_speaking_and_swallowing"
    d["option4_value7"]="Nausea_or_Vomiting(fever)"
    d["option4_value8"]="Loss_of_appetite"
    d["option4_value9"]="Cough_with_blood"
    d["option5_value6"]="Encephalopathy(brain_damage)"
    d["option5_value7"]="Pain_behind_the_eyes"
    d["option5_value8"]="Headache"
    d["option3_value8"]="Night_sweats_or_chill"
    d["option5_value9"]="Problems_with_speech,swallowing_or_physical_coordination"
    d["option3_value5"]="Widening_rounding_of_the_tips_of_fingers/toes(clubbing)"
    d["option5_value10"]="Swollen_glands"

    #print(d,'\n',len(d))
    print(len(d))

    print('---------------------------------------------------------------------------------------------')
    print()
    return d

def web_dis():
    d={}
    d["Easy or excessive Bruising"]="Easy_or_excessive_Bruising"
    d["Easy or excessive Bleeding"]="Easy_or_excessive_Bleeding"
    d["Petechiae"]="Petechiae(small_red_or_purple_dots_on_the_skin)"
    d["Fatigue"]="Fatigue"
    d["Enlarged spleen"]="Enlarged_spleen"
    d["Pain, tingling, burning sensations in the hands and feet"]="Pain_tingling_burning_sensations_in_the_hands_and_feet"
    d["Skin rash"]="Skin_rash"
    d["Gastrointestinal symptoms"]="Gastrointestinal_symptoms((diarrhea_and(nausea_or_vomiting))_or_(any_symptom_except_diarrhea))"
    d["Eye problems"]="Eye_problems(such_as_cloudiness_or_a_corneal_opacity)"
    d["Hearing loss or Heart problems or Kidney problems"]="Hearing_loss_or_Heart_problems_or_Kidney_problems"
    d["Only Prolonged diarrhea"]="Only_Prolonged_diarrhea"
    d["Wet cough"]="Wet_cough"
    d["Frequent lung (or sinus) infections"]="Frequent_lung(or_sinus)infections"
    d["Poor growth and failure to gain weight"]="Poor_growth_and_failure_to_gain_weight"
    d["Male infertility"]="Male_infertility"
    d["Shortness of breath"]="Shortness_of_breath"
    d["Loss of appetite & abdominal pain"]="Loss_of_appetite_&_abdominal_pain"
    d["Yellowing of the skin"]="Yellowing_of_the_skin(or_the_whites_of_the_eye)(jaundice)"
    d["Fluid buildup in the legs or abdomen"]="Fluid_buildup_in_the_legs_or_abdomen"
    d["Golden-brown eye discoloration"]="Golden_brown_eye_discoloration(Kayser-Fleischer_rings)"
    d["Uncontrolled movements or muscle stiffness or tremors"]="Uncontrolled_movements_or_muscle_stiffness_or_tremors"
    d["Dry cough"]="Dry_cough"
    d["Aching muscles & joints"]="Aching_muscles_&_joints"
    d["Weight and muscle loss"]="Weight_and_muscle_loss"
    d["Severe Headache"]="Severe_Headache"
    d["Itchy, blister rash"]="Itchy(blister_rash)"
    d["Chronic cough or Hoarseness"]="Chronic_cough_or_Hoarseness"
    d["Drooping of the eyelids and tingling sensations in the limbs"]="Drooping_of_the_eyelids_and_tingling_sensations_in_the_limbs"
    d["High fever 104 F (40 C)"]="High_fever_104F(40C)"
    d["Fever"]="Fever"
    d["Chest pain"]="Chest_pain"
    d["Digestive problems"]="Digestive_problems"
    d["Muscle, bone or joint pain"]="Muscle,bone_or_joint_pain"
    d["Unexplained weight loss"]="Unexplained_weight_loss"
    d["Weight loss"]="Weight_loss"
    d["Progressive difficulty speaking and swallowing"]= "Progressive_difficulty_speaking_and_swallowing"
    d["Nausea or Vomiting (and fever)"]="Nausea_or_Vomiting(fever)"
    d["Loss of appetite"]="Loss_of_appetite"
    d["Cough with blood"]="Cough_with_blood"
    d["Encephalopathy (brain damage)"]="Encephalopathy(brain_damage)"
    d["Pain behind the eyes"]="Pain_behind_the_eyes"
    d["Headache"]="Headache"
    d["Night sweat or chills"]="Night_sweats_or_chill"
    d["Problems with Speech, swallowing, Physical coordination"]="Problems_with_speech,swallowing_or_physical_coordination"
    d["Widening rounding of the tips of fingers/toes"]="Widening_rounding_of_the_tips_of_fingers/toes(clubbing)"
    d["Swollen glands"]="Swollen_glands"

    l={}
    for i in d:
         l[d[i]]=i
    return l

d=init_dp(d)
webb=web_dis()
print('WEB NAMES:',len(webb))





@app.route('/predict', methods=['POST'])
def predict_output():           #home
    print('2- FORM DATA')
    data1 = request.form['option1']
    data2 = request.form['option2']
    data3 = request.form['option3']
    data4 = request.form['option4']
    data5 = request.form['option5']

    #print('2-a- values')
    #print(data1,data2,data3,data4,data5)

    web33=[data1,data2,data3,data4,data5]
    #print(web1)
    
    
    # Assuming d is your dictionary for symptom mapping
    data1 = d.get(data1, -1)  # If not found, default to 0 or some other value
    data2 = d.get(data2, -1)
    data3 = d.get(data3, -1)
    data4 = d.get(data4, -1)
    data5 = d.get(data5, -1)
    #print('2-b-  mapped values')
    #print(data1,data2,data3,data4,data5)

    
    l3=[data1, data2, data3, data4, data5]
    web1=[data1, data2, data3, data4, data5]

    #count of empty symptoms
    c=0
    for i in range(len(l3)):
        if l3[i] ==-1:
              l3[i]=''
              web1[i]='-'
        else:
            web1[i]=webb[web1[i]]
            c+=1


    #edgecase
    if(c==0):
         P='None'
         T='WARNING: Zero symptom given, cannot predict disease! Kindly give atleast 3 symptoms for accurate results.'
         return render_template('dp_output.html', predicted_class=P, additional = T,flag=1,S1=web1[0],S2=web1[1],S3=web1[2],S4=web1[3],S5=web1[4])
              
    if(c==1):
         P='None'
         T='WARNING: Only one symptom given, cannot predict disease with 1 symptom! Kindly give atleast 3 symptoms for accurate results.'
         return render_template('dp_output.html', predicted_class=P, additional = T,flag=1,S1=web1[0],S2=web1[1],S3=web1[2],S4=web1[3],S5=web1[4])
              

    #print('2-c- list values')
    #print(l3)

    S5=''
    for i in range(len(l3)):
          S5+= str(l3[i]) 
          S5+= ','
    #print('2-c- string values')
    #print(S5)

    P=dp_func.predict_disease(S5,m)
    #print('2After calling')
    #print('output:',P)

    #integration testing
    print()
    print('---------------------------------------------------------------------------------------------')
    print()
    print('INTEGRATION TEST1           ---> SBOMCN_APP.py')
    print('Web_input:',web1)
    print('Symptoms_mapped:',l3)
    print('Output:',P)
    print('---------------------------------------------------------------------------------------------')
    print()

    #print('datas:',data1,data2,data3,data4,data5)

    #render output on web
    if(c==2):
        T='Kindly give 3 or 4 symptoms to get accurate results.'
        return render_template('dp_output.html', predicted_class=P, additional=T,flag=0,S1=web1[0],S2=web1[1],S3=web1[2],S4=web1[3],S5=web1[4])
    T='Go to \'Find Hospital\' tab or \'Next\' button, to navigate your preferred Medical Center.'
    return render_template('dp_output.html', predicted_class=P,additional=T , flag=2,S1=web1[0],S2=web1[1],S3=web1[2],S4=web1[3],S5=web1[4])

#------------------------------------------------------MAIN------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)



