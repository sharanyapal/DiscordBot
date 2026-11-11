# DiscordBot

**A Python Discord bot with quote retrieval, encouragement responses and a local JSON data store.**

`Discord.py · Python · Requests · JSON` · `Bot experiment` · Repository reference

---

## Project overview

The bot's Python entry point contains message and encouragement logic, an external quote API call, and a JSON-backed store. Additional JavaScript files explore face and eye tracking; they should be treated as separate experiments rather than required bot capabilities.

## Run or inspect locally

Use a Python virtual environment, install packages required by `main.py` (including Discord.py, Requests and python-dotenv), and review `keep_alive.py` and the bot permissions. Start the bot with:

```bash
python main.py
```

Provide a new bot token using local environment configuration. **Never commit or share bot credentials.**

## Repository contents

| Item | What it represents |
|:--|:--|
| [`README.md`](./README.md) | Project documentation |
| [`data.json`](./data.json) | Project file or module |
| [`eye_tracker.js`](./eye_tracker.js) | Application code |
| [`face_detector.js`](./face_detector.js) | Application code |
| [`face_landmarks.js`](./face_landmarks.js) | Application code |
| [`face_spoofing.js`](./face_spoofing.js) | Application code |
| [`keep_alive.py`](./keep_alive.py) | Application code |
| [`main.py`](./main.py) | Application code |
| [`proctor.js`](./proctor.js) | Application code |

## How to evaluate

- Start from the program entry point or interface.
- Test with non-sensitive sample data and review failure cases.
- Inspect any dependency, permission, token or environment-variable requirements before execution.

## Scope

Do not use the repository's historical credential values. Rotate any active tokens before running or publishing an instance.


This README reflects files currently present in the repository. It does not claim that all external services, hosting links or dependencies are operational. Never publish credentials or user data when testing it.

<details><summary><strong>Original README / historical project notes</strong></summary>

# DiscordBot
Creating a discord bot and make it work.

</details>

---

<sub>Maintained with the [Sharanya Pal GitHub project collection](https://github.com/sharanyapal?tab=repositories).</sub>
