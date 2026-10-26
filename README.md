# ECG Noise Reduction & Heart Rhythm Anomaly Detection (PCA)

We take real heartbeats, add real motion noise, and use linear algebra (covariance, eigenvectors, orthogonal projection) to clean the signal, compress it, and catch abnormal beats.

## What's in the folder

| File | What it does |
|---|---|
| `linalg.py` | The maths: centering, covariance, our own eigen-solver (power iteration), projection matrix, and proof checks |
| `apps.py` | Uses the maths: denoising, compression, Eckart–Young check, anomaly scores (SPE, T²), accuracy metrics |
| `demo.py` | Runs everything in the terminal and saves a picture of the results |
| `main.py` | The backend server the website talks to |
| `index.html` | The website: 6 animated steps, 3-D views and the final anomaly verdict |
| `data/` | The ECG recordings (if one is missing, it downloads automatically from PhysioNet's AWS mirror the first time) |
| `requirements.txt` | The list of Python libraries needed |

## Step 1: Install Python

You need **Python 3.10 or newer**. Check by opening a terminal and typing:

```
python --version
```

If it says "not found", install it from python.org. On Windows, tick **"Add Python to PATH"** during install.

## Step 2: Open a terminal inside this folder

Unzip the project. Then either:
- **VS Code:** File → Open Folder → pick `mathsmini`, then Terminal → New Terminal
- **Windows:** open the folder, click the address bar, type `cmd`, press Enter

## Step 3: Install the libraries (only once)

```
pip install -r requirements.txt
```

This reads `requirements.txt` and installs everything in it. It takes a minute or two.

## Step 4: Run the demo

```
python demo.py
```

This prints every maths stage with its proof checks, then saves the charts as `demo_119.png`. Open that image to see them.

You can also choose a different patient, noise type and noise level:

```
python demo.py 208 ma 12
```

- `208` = which patient recording (try 100, 105, 119, 200, 208, 233)
- `ma` = noise type: `em` electrode motion, `ma` muscle, `bw` baseline wander
- `12` = noise level in dB (higher number = less noise)

## Step 5: Open the website

```
uvicorn main:app --reload --port 8000
```

This starts the server. **Keep this terminal open.** Closing it stops the server.

Then open **http://127.0.0.1:8000** in your browser. The ECG Lab website opens and runs the maths automatically.

- Pick a patient and noise type on the left, then press **Load data & run maths**.
- Go through **steps 1 to 6** (or use the **Next** button at the bottom of each step).
- Every step has a **See the maths happen** panel that animates the matrix operation, plus **proof checks** that test the theorems live.
- Step 6 ends with the **final verdict**: anomaly detected or not. Drag the alarm-line slider and watch the verdict, the 3-D view and the numbers update.

Want the raw API instead? It's at **http://127.0.0.1:8000/docs**.

To stop the server, click in its terminal and press **Ctrl + C**.

## The 6 stages

1. **Data matrix:** each heartbeat becomes one row, giving a 1987 × 504 matrix.
2. **Covariance:** C = X̃ᵀX̃ / (M−1). It shows how every sample point moves with every other.
3. **Eigenbasis:** our own power-iteration solver finds the main "shapes" of a heartbeat (eigen-beats).
4. **Projection:** P = V_k V_kᵀ keeps the top k shapes and throws away the rest, which is mostly noise.
5. **Denoise + compress:** gives the SNR gain and compression ratio, and checks the Eckart–Young theorem holds exactly.
6. **Anomaly detection:** abnormal beats don't fit the "normal" shapes, so their leftover error (SPE) spikes.

## Something not working?

| Problem | Fix |
|---|---|
| `python` not found | Try `py` instead of `python`, or reinstall Python with "Add to PATH" ticked |
| `pip` not found | Use `python -m pip install -r requirements.txt` |
| Stuck on "downloading..." | It tries the fast AWS mirror first, then PhysioNet. If both are slow, get the `data/` folder from a teammate |
| Browser says "can't reach this site" | The server isn't running. Do Step 5 again and keep that terminal open |
| `call POST /fit first` | The server restarted. Press **Load data & run maths** on the website again |
