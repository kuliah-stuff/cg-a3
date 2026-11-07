# Virtual environment guide

A virtual environment (venv) lives in `.venv/` in this folder. It holds its own
copies of numpy, numba and pillow, so they don't clash with the packages
installed globally on your machine (the global numpy 2.5 / numba mismatch is
why we made it).

Installed in the venv: numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, pillow 13.0.0.

## Activate it

Open a terminal in this folder (`Assignment3-starter`), then:

**PowerShell** (also the VS Code terminal)
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, run this once, then try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**cmd.exe**
```bat
.venv\Scripts\activate.bat
```

**Git Bash**
```bash
source .venv/Scripts/activate
```

When active, your prompt starts with `(.venv)`. From then on, `python` and
`pip` refer to the venv's copies.

## Use it

```powershell
python check.py          # run all public checks
python check.py 1        # run only Task 1
python run.py --scene cornell --size 128 --spp 16 --out cornell.png
```

Leave the venv with:
```powershell
deactivate
```

### Without activating

You can call the venv's Python directly:
```powershell
.\.venv\Scripts\python.exe run.py --scene cornell --size 64 --spp 4
```

### VS Code

Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, then pick
`.venv\Scripts\python.exe`. New terminals will then activate it automatically.

## Check that it works

```powershell
python -c "import numpy, numba, PIL; print(numpy.__version__, numba.__version__)"
```
Expected: `2.5.4 0.68.1`.

## Install or change packages

Always do this with the venv active (or via `.\.venv\Scripts\python.exe -m pip`):
```powershell
python -m pip install -r requirements.txt
python -m pip list
```

## Recreate from scratch

If the venv breaks or you move the folder (venvs hold absolute paths, so
moving or copying the folder breaks them):
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Numba needs NumPy 2.4 or less` | You are using the global Python, not the venv. Activate `.venv` first. |
| `getaddrinfo failed` during install | No internet or DNS. Check your connection or VPN and retry. |
| `python` not found after activating | Delete and recreate the venv (see above). |
| Activate script is blocked | Run the `Set-ExecutionPolicy` command above. |

Don't submit or share `.venv/`. It is large and machine-specific.
