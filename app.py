from flask import Flask, request, render_template, redirect, url_for, session
import json
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import random

def bildgenerierung(pipe, prompt):
    g = torch.Generator(device="cuda")
    g.manual_seed(1337)
    # Bildgenerierung
    prompt = ""
    negative_prompt = ""
    image = pipe(prompt, generator=g).images[0]

    # Speichern des generierten Bildes
    number = random.randint(0, 100000)
    image.save("static/images/image" + str(number) + ".png")


    return str(number)

app = Flask(__name__)

pipeline = pipeline(task="text-generation", model="Orenguteng/Llama-3-8B-Lexi-Uncensored", torch_dtype=torch.bfloat16, device_map="auto")

model_path = "juggernaut.safetensors"
pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
    )
pipe.to("cuda" if torch.cuda.is_available() else "cpu")


@app.route('/', methods=['GET', 'POST'])
def chat():  # put application's code here
    with open('chat_history.json', 'r', encoding='utf-8') as f:
        chat_h = json.load(f)

    with open('chatmodel_history.json', 'r', encoding='utf-8') as f:
        chatmodel_h = json.load(f)

    # Öffnet eine Datei zum Schreiben ('w')
    #with open('chat_history.json', 'w', encoding='utf-8') as f:
        # Schreibt das Dictionary im JSON-Format in die Datei
        #json.dump(mein_dict, f, ensure_ascii=False, indent=4)

    if request.method == 'POST':
        if request.form.get("message").startswith("image please"):
            numb = bildgenerierung(pipe, "placeholder")
            chat_h['messages'].append({"message": "image" + numb + ".png", "sender": "image"})
        else:
            chat_h['messages'].append({"message": request.form.get("message"), "sender": "andre"})

            chatmodel_h['messages'].append(
                {"role": "user", "content": request.form.get("message")}
             )

            response = pipeline(chatmodel_h['messages'], max_new_tokens=512)
            chat_h['messages'].append({"message": response[0]["generated_text"][-1]["content"], "sender": "other"})
            chatmodel_h['messages'] = response[0]["generated_text"]

        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(chat_h, f, ensure_ascii=False, indent=4)

        with open('chatmodel_history.json', 'w', encoding='utf-8') as f:
            json.dump(chatmodel_h, f, ensure_ascii=False, indent=4)


    return render_template('room.html', user="andre", messages=chat_h['messages'])


if __name__ == '__main__':
    app.run()
