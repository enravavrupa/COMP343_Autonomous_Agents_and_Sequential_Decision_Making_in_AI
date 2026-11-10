# Loading the OpenAI API Key from `.env`

Your `LLMClient` (Task 2, `agent.py`) needs an OpenAI API key. Don't hardcode
it in `agent.py` and don't paste it into chat/commit messages -- keep it in a
local `.env` file that never gets committed, and load it at runtime.

## 1. Install `python-dotenv`

Already pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Create your `.env` file

Copy the template and fill in your real key:

```bash
cp .env.example .env
```

`.env` should look like:

```
OPENAI_API_KEY=sk-your-real-key-here
```

`.env` is listed in `.gitignore` -- it will not be committed. `.env.example`
(no real key inside) is the one that *should* be committed, so anyone
cloning the repo knows which variable to set.

## 3. Load it in `agent.py`

At the top of `agent.py`, before you construct the OpenAI client:

```python
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ, if present

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment automatically
```

`load_dotenv()` is a no-op if `.env` doesn't exist or the variable is
already set some other way (e.g. `export OPENAI_API_KEY=...` in your shell,
which still works and takes priority over `.env`) -- so this is safe to call
unconditionally.

## 4. Sanity-check it worked

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.environ.get('OPENAI_API_KEY')))"
```

Should print `True`. If it prints `False`, either `.env` doesn't exist yet,
it's not in the directory you're running from, or the line inside it isn't
exactly `OPENAI_API_KEY=...` (no quotes needed, no spaces around `=`).

## Common mistakes

- **Committing `.env`.** Check `git status` before every commit while this
  file exists locally -- if `.env` shows up as untracked/staged, don't add
  it. If you ever do commit a real key, treat it as leaked: revoke it on
  the OpenAI dashboard and generate a new one, don't just delete the file
  in a later commit (it's still in git history).
- **Calling `load_dotenv()` after constructing the client.** The client
  reads the environment at construction time -- load the `.env` file first.
- **Running from a different working directory.** `load_dotenv()` looks for
  `.env` in the current directory by default. If you run `agent.py` from
  somewhere else, pass the path explicitly: `load_dotenv(dotenv_path="/full/path/to/.env")`.
