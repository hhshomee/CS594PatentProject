from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "hf_token_" #Please change this lie with your hugging-face access token 

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    use_auth_token=hf_token  
)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
df = pd.read_csv('data/data_A61.csv')




patent_classes = {
    "A61B": "DIAGNOSIS; SURGERY; IDENTIFICATION",
    "A61F": "FILTERS IMPLANTABLE INTO BLOOD VESSELS",
    "A61K": "PREPARATIONS FOR MEDICAL, DENTAL OR TOILETRY PURPOSES",
    "A61L": "METHODS OR APPARATUS FOR STERILISING MATERIALS OR OBJECTS IN GENERAL",
    "A61M": "DEVICES FOR INTRODUCING MEDIA INTO, OR ONTO, THE BODY",
    "A61N": "ELECTROTHERAPY; MAGNETOTHERAPY; RADIATION THERAPY; ULTRASOUND THERAPY",
    "A61P": "SPECIFIC THERAPEUTIC ACTIVITY OF CHEMICAL COMPOUNDS OR MEDICINAL PREPARATIONS",
    "A61Q": "SPECIFIC USE OF COSMETICS OR SIMILAR TOILETRY PREPARATIONS"
}


if "predicted_class" not in df.columns:
    df["predicted_class"] = ""
if "explanation" not in df.columns:
    df["explanation"] = ""
if "output" not in df.columns:
    df["output"] = ""

save_path = os.path.join("output", "A61_explanation.csv")
for idx, row in df.iterrows():

    abstract = row["abstract"]


    prompt = f''' 
              You are an expert on US Patent field, and have knowledge on the difference between different type of patent classes. Your goal is to complete the following task:
              Task: You will use the patent abstract, and use it to generate explanation of the abstract and classify the patent into one of the following classes. The classification must be chosen from the following list:

              {chr(10).join([f"{key}: {value}" for key, value in patent_classes.items()])}
              Patent Abstract:
              \"\"\"{abstract}\"\"\"

              Respond in this exact format: 
              Class: <one of {list(patent_classes.keys())}> 
              Explanation: <brief reasoning under 100 words>

              Below is an example: 
              Class: A61K
              Explanation:  The abstract describes vaccine compositions for treating and/or preventing infections, which is a medical preparation for a specific purpose. The composition comprises bacteria of the Chlamydiaceae family, which have been previously treated by at least one peptidoglycan inhibitor, or extracts of said treated bacteria. This falls under the category of "preparations for medical, dental or toiletry purposes" as described in A61K.


    '''


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)


    output_text = decoded.split(prompt)[-1].strip()
    class_line = [line for line in output_text.split("\n") if "Class:" in line]
    explanation_line = [line for line in output_text.split("\n") if "Explanation:" in line]

    predicted_class = class_line[0].replace("Class:", "").strip() if class_line else ""
    explanation = explanation_line[0].replace("Explanation:", "").strip() if explanation_line else output_text
     

    df.at[idx, "predicted_class"] = predicted_class
    df.at[idx, "explanation"] = explanation
    df.at[idx, "output"] = output_text



    df.to_csv(save_path, index=False)
    print(f"[{idx+1}/{len(df)}]  Saved class: {predicted_class}")

print("All explanation written back to CSV.")

