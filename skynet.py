import numpy as np
import requests
import gzip
import os
from PIL import Image, ImageOps

# Konfigurasi Arsitektur
input_nodes = 784
hidden_nodes = 128
output_nodes = 10
learn_rate = 0.1
MODEL_FILE = "ai_brain.npz"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- FUNGSI AUTO-CENTERING ---
def praporses_nuklir(path):
    img = Image.open(path).convert('L')
    img_array = np.array(img)

    if np.mean(img_array) > 127:
        img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    w, h = img.size
    max_side = max(w, h)

    new_img = Image.new('L', (int(max_side * 1.4), int(max_side * 1.4)), (0))
    offset = (
        int((max_side * 1.4 - w) / 2),
        int((max_side * 1.4 - h) / 2)
    )
    new_img.paste(img, offset)

    img_final = new_img.resize((28, 28), Image.Resampling.LANCZOS)
    return np.array(img_final).reshape(784, 1) / 255.0

# --- DATA AUGMENTATION ---
def augment(x):
    img = x.reshape(28, 28)
    shift = np.random.randint(-1, 2, 2)
    img = np.roll(img, shift[0], axis=0)
    img = np.roll(img, shift[1], axis=1)
    return img.reshape(784, 1)

def train_nuklir():
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"

    if not os.path.exists("mnist.pkl.gz"):
        print("📥 Mendownload data MNIST asli...")
        r = requests.get(url)
        with open("mnist.pkl.gz", "wb") as f:
            f.write(r.content)

    import pickle
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        tr, va, te = pickle.load(f, encoding='latin1')

    inputs, targets = tr

    wih = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
    who = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
    bh = np.zeros((hidden_nodes, 1))
    bo = np.zeros((output_nodes, 1))

    epochs = 20
    print(f"🚀 Memulai Training {epochs} Epoch dengan Augmentasi...")

    for e in range(epochs):
        for i in range(len(inputs)):
            x = inputs[i].reshape(-1, 1)

            if np.random.rand() > 0.5:
                x = augment(x)

            y = np.zeros((10, 1))
            y[targets[i]] = 1

            h_out = sigmoid(np.dot(wih, x) + bh)
            o_out = sigmoid(np.dot(who, h_out) + bo)

            do = (o_out - y) * sigmoid_derivative(o_out)
            dh = np.dot(who.T, do) * sigmoid_derivative(h_out)

            who -= learn_rate * np.dot(do, h_out.T)
            bo -= learn_rate * do
            wih -= learn_rate * np.dot(dh, x.T)
            bh -= learn_rate * dh

        print(f"✅ Epoch {e + 1}/{epochs} Selesai")

    np.savez(MODEL_FILE, wih=wih, who=who, bh=bh, bo=bo)

def eksekusi_prediksi(path):
    brain = np.load(MODEL_FILE)
    input_data = praporses_nuklir(path)

    h = sigmoid(np.dot(brain['wih'], input_data) + brain['bh'])
    o = sigmoid(np.dot(brain['who'], h) + brain['bo'])

    print(f"\n🎯 HASIL AKHIR: {np.argmax(o)}")
    print(f"📊 KEYAKINAN: {np.max(o) * 100:.2f}%")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        train_nuklir()
    else:
        print("🧠 Menggunakan memori yang sudah ada. Hapus 'ai_brain.npz' untuk latihan ulang.")

    while True:
        p = input("\nPath Gambar (atau 'exit'): ")
        if p == 'exit':
            break
        try:
            eksekusi_prediksi(p)
        except Exception as e:
            print(f"Error: {e}")
