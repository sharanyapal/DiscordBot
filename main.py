import discord
import os
import requests
import json
import random
import cv2
import numpy as np
import urllib.request
from dotenv import load_dotenv
from keep_alive import keep_alive 

#os.environ['TOKEN'] = 'MTA2MTU0MTQ4NjQzODY2MjIwNA.G4VPG9.QLzobaAbm9zJtp-YcJJmyRqnbvioNvVUlslia8'

intents = discord.Intents().all()
client = discord.Client(intents=intents)

sad_words = ["sad", "depressed", "unhappy", "angry", "miserable", "depressing"]

starter_encouragements = [
  "Cheer up!",
  "Hang in there.",
  "You are a great person / bot!"
]

db_file = "data.json"

if os.path.exists(db_file):
    with open(db_file, "r") as file:
        db = json.load(file)
else:
    db = {
        "responding": True,
        "encouragements": []
    }
    with open(db_file, "w") as file:
        json.dump(db, file)

def get_quote():
  response = requests.get("https://zenquotes.io/api/random")
  json_data = json.loads(response.text)
  quote = json_data[0]['q'] + " -" + json_data[0]['a']
  return(quote)

def update_encouragements(encouraging_message):
  if "encouragements" in db.keys():
    encouragements = db["encouragements"]
    encouragements.append(encouraging_message)
    db["encouragements"] = encouragements
  else:
    db["encouragements"] = [encouraging_message]
  with open(db_file, "w") as file:
    json.dump(db, file)

def delete_encouragement(index):
  encouragements = db["encouragements"]
  if len(encouragements) > index:
    del encouragements[index]
    db["encouragements"] = encouragements
  with open(db_file, "w") as file:
    json.dump(db, file)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = message.content

    if msg.startswith('!hello'):
      await message.channel.send('Hello!')

    if msg.startswith('!ping'):
      await message.channel.send('Ping!')

    if msg.startswith('!react'):
      await message.add_reaction('👍')
      await message.add_reaction('👎')

    if msg.startswith('$inspire'):
      quote = get_quote()
      await message.channel.send(quote)

    if db["responding"]:
      options = starter_encouragements
      if "encouragements" in db.keys():
        options = options + db["encouragements"]

      if any(word in msg for word in sad_words):
        await message.channel.send(random.choice(options))

    if msg.startswith("$new"):
      encouraging_message = msg.split("$new ",1)[1]
      update_encouragements(encouraging_message)
      await message.channel.send("New encouraging message added.")

    if msg.startswith("$del"):
      encouragements = []
      if "encouragements" in db.keys():
        index = int(msg.split("$del",1)[1])
        delete_encouragment(index)
        encouragements = db["encouragements"]
      await message.channel.send(encouragements)

    if msg.startswith("$list"):
      encouragements = []
      if "encouragements" in db.keys():
        encouragements = db["encouragements"]
      await message.channel.send(encouragements)

    if msg.startswith("$responding"):
      value = msg.split("$responding ",1)[1]

      if value.lower() == "true":
        db["responding"] = True
        await message.channel.send("Responding is on.")
      else:
        db["responding"] = False
        await message.channel.send("Responding is off.")
      
    if msg.startswith('!face'):
        image_url = msg.split(' ')[1]
        image = np.array(bytearray(urllib.request.urlopen(image_url).read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        file = discord.File(fp=jpeg.tobytes(), filename='faces.jpg')
        await message.channel.send("Detected faces:", file=file)

#client.run('MTA2MTU0MTQ4NjQzODY2MjIwNA.G4VPG9.QLzobaAbm9zJtp-YcJJmyRqnbvioNvVUlslia8')
keep_alive()
load_dotenv()
client.run(os.getenv('TOKEN'))