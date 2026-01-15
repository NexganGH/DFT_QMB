# Setup: WSL + Ubuntu + GPAW + PyCharm

---

**Note**: this is mainly for Windows users. In MacOS you do not need WSL. Just run the commands starting from Step. 4.

# 1. Install and enable WSL

Open PowerShell as Administrator and run:

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

# 6. Install ASE and GPAW

Inside the virtual environment:

```bash
pip install --upgrade pip
pip install ase
pip install gpaw
```

This will successfully compile/install GPAW inside WSL.

---

# 7. Configure PyCharm to use the WSL interpreter
For this project we used the Pycharm IDE. You might as well simply run the commands on terminal. Here we show how to connect Pycharm to your Virtual Environment on WSL. If you are on MacOS/Linux, you need to select the virtual environment in the Pycharm settings.

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

Now you can simply run Pycharm

---

Here is a **clean Step 10** you can drop in, written to match the style and level of the rest of the file. I’ll also fix the numbering so it’s consistent.

---

# 10. Install project dependencies from `pyproject.toml`

After cloning the repository, move to the **main project directory** (the one containing `pyproject.toml`):

```bash
cd path/to/your/project
```

Make sure your GPAW virtual environment is active:

```bash
source ~/gpaw-env/bin/activate
```

Then install the project dependencies specified in `pyproject.toml`:

```bash
pip install .
```

If you want an editable (development) install:
```bash
pip install -e .
```

This ensures that all required Python dependencies for the project are installed consistently.

---

# 11.  Install missing dependencies
Not all Python dependencies are currently listed in pyproject.toml. If a script fails due to a missing package, install it as needed using:
```
pip install <package-name>
```

When using PyCharm, missing dependencies may also be detected automatically and can be installed directly when prompted. PyCharm will use the selected WSL-based interpreter.

# 12. (Optional) Install python3-tk to visualize ASE structures**

On WSL:

```bash
sudo apt install python3-tk
```
