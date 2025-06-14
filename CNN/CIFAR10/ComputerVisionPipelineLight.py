# swin_quick.py  ────────────────────────────────────────────────────
import os, cv2, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, io, transforms, models

# ─── edit these ─────────────────────────────────────────────────────
ROOT   = r"D:\data\flowers"      # expects ROOT\train\class\*.jpg etc.
BATCH  = 32
EPOCHS = 3
# ────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── OpenCV pre-processing (exact spec) ─────────────────────────────
def cv2_prep(t):                          # t : uint8 CHW RGB
    img = t.permute(1, 2, 0).numpy()      # → HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, k, 1);  img = cv2.erode(img, k, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.tensor(img).permute(2, 0, 1)      # CHW uint8
# -------------------------------------------------------------------

SIZE = 224            # Swin-T default
tfm  = transforms.Compose([
    cv2_prep,
    transforms.ConvertImageDtype(torch.float32),    # 0-1
    transforms.Resize(SIZE), transforms.CenterCrop(SIZE),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

train_dir, val_dir = os.path.join(ROOT, "train"), os.path.join(ROOT, "val")
train = datasets.ImageFolder(train_dir, loader=io.read_image, transform=tfm)
val   = datasets.ImageFolder(val_dir,   loader=io.read_image, transform=tfm)

train_loader = DataLoader(train, BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val,   BATCH, shuffle=False, num_workers=2, pin_memory=True)

nclass = len(train.classes)
print("Classes:", train.classes)

# ─── build Swin-T, replace head ─────────────────────────────────────
model = models.swin_t(weights="IMAGENET1K_V1")
model.head = nn.Linear(model.head.in_features, nclass)
model.to(device)

# ─── quick train ----------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
opt     = optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad();  loss_fn(model(x), y).backward();  opt.step()

# ─── validation accuracy -------------------------------------------
model.eval(); hits = tots = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        hits += (model(x).argmax(1) == y).sum().item();  tots += y.size(0)
print(f"Swin-T val-acc: {hits / tots:.3f}")

