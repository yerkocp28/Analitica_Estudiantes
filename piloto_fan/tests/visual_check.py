"""Comprobación visual manual con Edge; solo escribe en artifacts/ y no usa perfiles personales."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time

import requests
import websocket

HOME = Path(__file__).resolve().parents[1]
OUT = HOME / "artifacts/visual"
OUT.mkdir(exist_ok=True)
profile = tempfile.mkdtemp(prefix="edge-check-", dir=OUT)
browser = subprocess.Popen([
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--remote-debugging-port=0", "--remote-allow-origins=*", "--user-data-dir=" + profile, "about:blank"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
try:
    endpoint = Path(profile) / "DevToolsActivePort"
    deadline = time.monotonic() + 30
    while not endpoint.exists() and time.monotonic() < deadline:
        time.sleep(.2)
    port = int(endpoint.read_text().splitlines()[0])
    pages = requests.get(f"http://127.0.0.1:{port}/json", timeout=5).json()
    ws = websocket.create_connection(next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page"), timeout=15)
    counter = 0

    def call(method, params=None):
        global counter
        counter += 1
        ws.send(json.dumps({"id": counter, "method": method, "params": params or {}}))
        while True:
            result = json.loads(ws.recv())
            if result.get("id") == counter:
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result.get("result", {})

    def evaluate(expression):
        return call("Runtime.evaluate", {"expression": expression, "returnByValue": True}).get("result", {}).get("value")

    call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
    call("Page.navigate", {"url": "http://127.0.0.1:8502"})
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if evaluate("document.querySelectorAll('[data-testid=stMetric]').length"):
            break
        time.sleep(.3)
    for label, filename in [("Resumen", "resumen.png"), ("Continuidad", "continuidad.png"), ("Perfiles de ingreso", "perfiles.png")]:
        if label != "Resumen":
            evaluate("[...document.querySelectorAll('label')].find(x=>x.innerText.trim()===" + json.dumps(label) + ")?.click()")
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if evaluate("document.body.innerText.includes(" + json.dumps('Casos evaluados en la selección' if label == 'Continuidad' else 'Perfiles de ingreso de la facultad') + ")"):
                    break
                time.sleep(.3)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if not evaluate("document.body.innerText.includes('Stop')"):
                break
            time.sleep(.3)
        time.sleep(3)
        screenshot = call("Page.captureScreenshot", {"format": "png"})["data"]
        (OUT / filename).write_bytes(base64.b64decode(screenshot))
        (OUT / (filename + ".txt")).write_text(evaluate("document.body.innerText") or "", encoding="utf-8")
        print(filename, flush=True)
    ws.close()
finally:
    browser.terminate()
    browser.wait(timeout=10)
