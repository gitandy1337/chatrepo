from flask import Flask, render_template
from flask_socketio import SocketIO, send
from diffusers import StableDiffusionXLPipeline
import torch
from transformers import pipeline
import json
import random

def bildgenerierung(pipe, prompt_out):
    g = torch.Generator(device="cuda")
    g.manual_seed(1337)
    with open('magicnumber.txt', 'r', encoding='utf-8') as datei:
        i = int(datei.read())
        print(i)
    # Bildgenerierung
    prompt = "<prompt>" + prompt_out
    negative_prompt = "weird anatomy, camera, phone, other persons"

    image = pipe(prompt, negative_prompt=negative_prompt, generator=g).images[0]

    # Speichern des generierten Bildes
    number = random.randint(0, 100000)
    image.save("static/images/image" + str(number) + ".png")

    with open('magicnumber.txt', 'w', encoding='utf-8') as datei:
        datei.write(str(i))

    return str(number)

# Initialisierung der Flask-App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dein-super-geheimer-schluessel'  # Ändere das in der Produktion!
pipeline = pipeline(task="text-generation", model="<llm>",
                    torch_dtype=torch.bfloat16, device_map="auto")

# Initialisierung von SocketIO
socketio = SocketIO(app)

model_path = "juggernaut.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
    )
pipe.to("cuda" if torch.cuda.is_available() else "cpu")


# Route, um die Hauptseite (das Chat-Interface) zu laden
@app.route('/')
def index():
    """Liefert die HTML-Datei für den Chat."""
    return render_template('index.html')

def antwort(message):
    imagerequest = False

    with open('chat.json', 'r', encoding='utf-8') as f:
            chat_json = json.load(f)

    chat_json['messages'].append(
        {"role": "user", "content": message}
    )

    response = pipeline(chat_json['messages'], max_new_tokens=512)

    if response[0]["generated_text"][-1]["content"].startswith('<imagerequest>'):
        imagerequest = True
        print("war ein request")
    
    chat_json['messages'] = response[0]["generated_text"]

    if imagerequest:
        send({"sender":"Susanne", "message":response[0]["generated_text"][-1]["content"][14:]}, broadcast=True)
    else:
        send({"sender":"Susanne", "message":response[0]["generated_text"][-1]["content"]}, broadcast=True)

    with open('chat.json', 'w', encoding='utf-8') as f:
        json.dump(chat_json, f, ensure_ascii=False, indent=4)

    if imagerequest:
        mess = "describe your situation as you would like to give it to an text2image model, put the description in [ and ]."
        chat_json['messages'].append(
            {"role": "user", "content": mess}
        )
        response = pipeline(chat_json['messages'], max_new_tokens=512)
        print(response[0]["generated_text"][-1]["content"])
        prompi = response[0]["generated_text"][-1]["content"].split("[")[1].split("]")[0]
        print(prompi)

        num = bildgenerierung(pipe, prompi)
        send({"sender": "image", "message": num})


# Event-Handler für eingehende Nachrichten vom Client
@socketio.on('message')
def handle_message(msg):
    """Empfängt eine Nachricht und sendet sie an alle Clients zurück."""
    print(f"Nachricht empfangen: {msg}")

    msg_dict = {'sender': 'Peter', 'message': msg}
    send(msg_dict, broadcast=True)

    antwort(msg)

# Startet die Anwendung
if __name__ == '__main__':
    # '0.0.0.0' macht den Server im lokalen Netzwerk erreichbar
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
