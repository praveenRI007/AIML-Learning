<h1 align="center">ATS Friendly Resume Maker</h1>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg"
       alt="Gemini Logo"
       width="160"/>
</p>

<p align="center">
  Powered by Google's Gemini API
</p>

<br><br>


- Building a clean resume from scratch
- Tailoring it for every job description without rewriting from zero
- Watching the ATS score change in real time as you edit
- Downloading a polished PDF the moment you're done

- No BS ads !
- No BS SaaS fees !


using your own **free gemini** api key !

<br><br>
## Demo


*(GIF placeholder)*

<br><br>
## How it works

- Upload your resume → Gemini parses it into an editable form with a **live preview**.
- Paste a job description → keywords get extracted and scored against your CV.
- Click **Build ATS Version** → Gemini rewrites your bullets using the JD's vocabulary, but only keywords your CV already supports the claim.
- Live preview will **highlight** the **keywords** in **yellow** and keywords chosen if **not valid** can be **removed** and ATS score will update accordingly .
- Toggle between **original** and **tailored**, view respective ATS scores and **edit** either, **download** as **PDF** .

**Note :** _Be mindful when using the application since behind the scene AI (gemini) is infusing content into your cv , manual review is definitely required :D !_

<br><br>
## How to Run it ?
<br><br>
### Option 1 — Prebuilt executable

1. Get a Gemini key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Download the executable from the artifact folder in repository.
3. Create a `.env` file next to `resume-builder.exe`:
```
   GEMINI_API_KEY=AIzaSy...your-key-here...
```
3. Double-click `resume-builder.exe`. Browser opens at `http://localhost:8000`.

<br><br>
### Option 2 — From source

For modifying prompts, scoring, layout, etc.

```bash
git clone [repo](https://github.com/praveenRI007/AIML-Learning.git)
cd <repo>
pip install -r requirements.txt
echo "GEMINI_API_KEY=AIzaSy..." > .env
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`.

and voila you have web app running in your local :D !

<br><br>

### Developer Note

Its a simple prototype that i have developed so that it can be useful , if you feel its useful and really have ideas to improve around this app 

feel free to open an issue, start a discussion, or submit a PR. :D !


## License

MIT
