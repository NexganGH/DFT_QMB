# ✅ Reproducible Setup Guide: WSL + Ubuntu + GPAW + PyCharm

This guide describes **every step required** to:

1. Install **WSL** (Windows Subsystem for Linux)  
2. Install **Ubuntu**  
3. Create a **Python virtual environment** in WSL  
4. Install **ASE** and **GPAW** (Linux-only packages)  
5. Configure **PyCharm** to use the WSL interpreter  
6. Run DFT/GPAW code from PyCharm

Every command is fully reproducible.

---

# 1. Install and enable WSL

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

If the system asks you to reboot, restart your machine and run the same command again.

To explicitly install Ubuntu:

```powershell
wsl --install -d Ubuntu
```

This installs:

- WSL2  
- Ubuntu as the default Linux distribution  

After installation, a terminal window will open asking you to:

- Create a **Linux username**
- Create a **Linux password**

---

# 2. Verify WSL installation

Open PowerShell and run:

```powershell
wsl --status
wsl --list --verbose
```

Expected output example:

```
NAME      STATE     VERSION
Ubuntu    Running   2
```

---

# 3. Open Ubuntu (WSL)

From PowerShell or the Start Menu:

```powershell
wsl
```

You should now see a Linux prompt similar to:

```
yourname@YOURPC:~$
```

This means WSL is working.

---

# 4. Install required Linux packages

Inside the Ubuntu terminal, run:

```bash
sudo apt update
sudo apt install -y python3-pip python3-dev python3-venv build-essential \
    libxc-dev libfftw3-dev libopenblas-dev gfortran
```

These are required dependencies for GPAW and scientific Python.

---

# 5. Create a virtual environment for GPAW

Inside Ubuntu:

```bash
cd ~
python3 -m venv gpaw-env
source gpaw-env/bin/activate
```

Your prompt should now show:

```
(gpaw-env) yourname@yourpc:~$
```

---

# 6. Install ASE and GPAW (Linux only)

Inside the virtual environment:

```bash
pip install --upgrade pip
pip install ase
pip install gpaw
```

This will successfully compile/install GPAW inside WSL.

---

# 7. (Optional) Test GPAW installation

```python
python3 - << 'EOF'
import gpaw
import ase
print("GPAW installed:", gpaw.__version__)
print("ASE installed:", ase.__version__)
EOF
```

---

# 8. Configure PyCharm to use the WSL interpreter

### 1. Open PyCharm → *Settings*  
`File → Settings → Project → Python Interpreter`

### 2. Click the **⚙️ gear** → *Add Interpreter…*

### 3. Choose **“WSL”**

### 4. Select the Python executable from your venv:

```
/home/<your-username>/gpaw-env/bin/python
```

(You may browse manually if PyCharm does not auto-detect it.)

### 5. Click **OK** and wait for indexing.

PyCharm will now execute all Python code inside Ubuntu.

---

# 9. Run your DFT/GPAW scripts in PyCharm

Use PyCharm normally.  
All scripts will run in the Linux environment and will work with:

- GPAW  
- ASE  
- NumPy/SciPy  
- MPI (optional)

This is now a complete, reproducible DFT environment.

---

# 10. (Optional) Install Pythonk3-tk to visualize Ase structures

On WSL
```
sudo apt install python3-tk
```

---

# 🎉 Done!

You now have:

✔ Fully working WSL + Ubuntu  
✔ Proper virtual environment  
✔ GPAW + ASE installed  
✔ PyCharm connected to WSL interpreter  
✔ Reproducible setup for your DFT_QMB project  

You can safely share this file with others to reproduce your environment exactly.
