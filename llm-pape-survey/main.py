import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

#loading data
df = pd.read_csv("papers.csv")

#preprocess
df['info'] = 'The Document Title is "' + df['Document Title'] + '". The Abstract is "' + df['Abstract'] + '". The IEEE Terms keywords are "' + df['IEEE Terms'] + '".'

#model
llm = Ollama(model="llama3") #, temperature=0.9, num_predict=5)

#prompt function
def output(input):

    prompt_medical = PromptTemplate.from_template(
    "Just answer whether the topic is related to medical diagnosis or not strictly among the {options} only, used in the following paper with its description as follows: {message}?"
    )
    medical = llm.invoke(prompt_medical.format(options = ["Yes", "No"], message=input))

    prompt_modality = PromptTemplate.from_template(
    "Just answer the name in a SINGLE WORD of the image modality strictly among the {options} only, used in the following paper with its description as follows: {message}?"
    )
    modality = llm.invoke(prompt_modality.format(options = ["CT", "MRI", "X-Ray", "None"], message=input))

    prompt_disease = PromptTemplate.from_template(
    "Only write answer the name of the disease in {message}? Donot write a full sentence. Only give the name."
    )
    disease = llm.invoke(prompt_disease.format(message=input))

    prompt_model = PromptTemplate.from_template(
    "Just answer whether a new model is built or not strictly among the {options} only, used in the following paper with its description as follows: {message}?"
    )
    model = llm.invoke(prompt_model.format(options = ["Yes", "No"], message=input))

    prompt_explainibility = PromptTemplate.from_template(
    "Just answer whether there is explainaibility and interpretability of the model among the {options} only, used in the following paper with its description as follows: {message}?"
    )
    explainibility = llm.invoke(prompt_explainibility.format(options = ["Yes", "No"], message=input))

    return medical, modality, disease, model, explainibility

#result
medical_list, modality_list, disease_list, model_list, explainibility_list = [], [], [], [], []
for index, row in df.iterrows():  # Process only the first 10 rows
    input_text = row['info']
    medical, modality, disease, model, explainibility = output(input_text)
    
    medical_list.append(medical)
    modality_list.append(modality)
    disease_list.append(disease)
    model_list.append(model)
    explainibility_list.append(explainibility)

df['Medical'] = medical_list
df['Modality'] = modality_list
df['Disease'] = disease_list
df['Model'] = model_list
df['Explainability'] = explainibility_list

# Save the updated DataFrame to a new CSV file if needed
df.to_csv('updated_papers.csv', index=False)

# Optionally, print the updated DataFrame
print(df)
