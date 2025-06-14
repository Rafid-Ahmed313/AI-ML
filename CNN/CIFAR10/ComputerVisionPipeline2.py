# quick_pipeline_dynamic.py
import os, cv2, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, io, transforms, models

# ─── tweak only these ────────────────────────────────────────────────
ROOT   = r"D:\data\flowers"    # expect ROOT\train\class\*.jpg etc.
BATCH  = 32
EPOCHS = 3
# ─────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── OpenCV pre-process (exactly your steps) ────────────────────────
def cv2_prep(t):
    img = t.permute(1, 2, 0).numpy()                 # HWC RGB uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (0, 0), fx=4, fy=4,
                     interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    k   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, k, 1);  img = cv2.erode(img, k, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.tensor(img).permute(2, 0, 1)        # CHW uint8
# --------------------------------------------------------------------

SIZE = 299
tfm  = transforms.Compose([
    cv2_prep,
    transforms.ConvertImageDtype(torch.float32),
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

# ─── pick the backbones you want – comment out to skip ───────────────
models4 = [
    models.resnet50(weights="IMAGENET1K_V2"),
    models.swin_t(weights="IMAGENET1K_V1"),
    models.vit_b_16(weights="IMAGENET1K_V1"),
    models.inception_v3(weights="IMAGENET1K_V1", aux_logits=False)
]
# --------------------------------------------------------------------

# replace classifier layers automatically
for m in models4:
    if isinstance(m, models.ResNet):
        m.fc = nn.Linear(m.fc.in_features, nclass)
    elif isinstance(m, models.SwinTransformer):
        m.head = nn.Linear(m.head.in_features, nclass)
    elif isinstance(m, models.VisionTransformer):
        m.heads.head = nn.Linear(m.heads.head.in_features, nclass)
    elif isinstance(m, models.Inception3):
        m.fc = nn.Linear(m.fc.in_features, nclass)
    m.to(device)

# ─── mini training loop ─────────────────────────────────────────────
loss_fn = nn.CrossEntropyLoss()
for m in models4:
    opt = optim.AdamW(m.parameters(), lr=3e-4)
    for _ in range(EPOCHS):
        m.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad();   loss_fn(m(x), y).backward();   opt.step()

# ─── soft-vote ensemble ─────────────────────────────────────────────
class Ensemble(nn.Module):
    def __init__(self, nets):
        super().__init__();   self.nets = nn.ModuleList(nets)
        for n in self.nets: n.eval()
    def forward(self, x):
        return torch.stack([torch.softmax(n(x), 1) for n in self.nets]).mean(0)

ens = Ensemble(models4).to(device)

def accuracy(model):
    model.eval(); hit = tot = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            hit += (model(x).argmax(1) == y).sum().item();  tot += y.size(0)
    return hit / tot

print(f"Ensemble val-acc: {accuracy(ens):.3f}")

