from openai import OpenAI
import argparse
kimi_url = "https://api.moonshot.cn/v1"
kimi_key = "sk-Llg1MqZy0oucHMnNwQYgfutkzAXn6S4fKrL212GE4vs1FF81"
kimi_id = 'moonshot-128k'

deepseek_url = "https://api.deepseek.com/v1"
deepseek_key = "sk-88614777ad0a452fb0f2e0e4dff29716"
deepseek_id = 'deepseek-reasoner'

openai_url_en = "https://api.openai.com/v1"
openai_url = "http://rerverseapi.workergpt.cn/v1"
openai_id = 'gpt-4o'

openai_key1 = "sk-f8Pi0ldBJaVSbFzI51187aD5541f443384E1E702933a6a4f"
openai_key2 = "sk-ooj0BpGiP6eRcosaDf6c2bFcE86440Fa809dB9773dF7377f"
openai_key3 = "sk-CxlcVbbbl9qqEexB162f2077CcCb421bA12280585d1eD0F9"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ds")
args = parser.parse_args()
model = args.model

if model == 'ds':
    api_key = deepseek_key
    base_url = deepseek_url
    model_id = deepseek_id
elif model == 'kimi':
    api_key = kimi_key
    base_url = kimi_url
    model_id = kimi_id
elif model == 'openai':
    api_key = openai_key1
    base_url = openai_url
    model_id = openai_id

client = OpenAI(
    api_key = api_key,
    base_url = base_url,
)

question = input("请提出一个开放而深刻的艺术、科学或文学问题：🥰\n")
while True:
    completion = client.chat.completions.create(
        model = model_id,
        messages = [
            {"role": "system", "content": "你是一名思考家，你需要根据上一步思考，自由延展你的思考深度和广度，最后再提出一个相关拓展问题，注意不要局限在某一方面，要广泛思考。"},
            {"role": "user", "content": f"{question}"},
        ],
        temperature = 1,
    )
    question = completion.choices[0].message.content
    print(completion.choices[0].message.content)
    # with open("test.txt", "a") as f:    
    #     f.write(question + "\n")
        