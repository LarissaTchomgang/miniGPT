import torch
import torch.nn as nn
import os
import glob
import gc
import random
import time
import math
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import FastBPETokenizer

# ===================== مسارات الملفات =====================
DATA_DIR = "/content/drive/MyDrive/ai/miniGPT/data/pps"
SAVE_DIR = "/content/drive/MyDrive/ai/miniGPT/models/MiniGPT_LC_2"
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_NAME = "checkpoint.pt"
FINAL_NAME = "MiniGPT_LC.pt"
CKPT_PATH = os.path.join(SAVE_DIR, CKPT_NAME)
FINAL_PATH = os.path.join(SAVE_DIR, FINAL_NAME)
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer.json")

# ===================== الجهاز =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 الجهاز المستخدم: {device}")

# ===================== إعدادات التدريب =====================
batch_size = 32  
epochs = 140
lr = 3e-4  
warmup_steps = 250
save_every = 10
max_seq_len = 256  # طول التسلسل
gradient_accumulation_steps = 1  # تراكم التدرجات

# ===================== إعدادات النموذج =====================
TARGET_EMBED = 256
TARGET_HEADS = 4
TARGET_LAYERS = 4
TARGET_VOCAB = 10700
NUM_LAYERS_TO_TRAIN = None

# ===================== فئات البيانات =====================
class TextChunkDataset(Dataset):
    """مجموعة بيانات فعالة في استخدام الذاكرة"""
    def __init__(self, files, tokenizer, seq_len=128, chunks_per_file=100):
        self.seq_len = seq_len
        self.chunks = []
        self._prepare_chunks(files, tokenizer, chunks_per_file)
    
    def _prepare_chunks(self, files, tokenizer, chunks_per_file):
        print(f"📊 Preparing data sets...")
        
        for file_idx, file_path in enumerate(files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                    if len(text) < 1000:  # تجاهل الملفات الصغيرة جداً
                        continue
                    
                    # تفكيك النص
                    encoded = tokenizer.encode(text)
                    
                    # إنشاء مجموعات من النص
                    for i in range(0, len(encoded) - self.seq_len, self.seq_len):
                        if len(self.chunks) >= len(files) * chunks_per_file:
                            break
                        
                        chunk = encoded[i:i + self.seq_len]
                        if len(chunk) == self.seq_len:
                            self.chunks.append(chunk)
                    
                    if (file_idx + 1) % 5 == 0:
                        print(f" handler {file_idx + 1}/{len(files)} - sets: {len(self.chunks)}")
                        
            except Exception as e:
               print(f"⚠️ Error in {file_path}: {e}")
        
        print(f"✅ {len(self.chunks)} set prepared")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# ===================== دوال المساعدة =====================
def clear_memory():
    """تنظيف الذاكرة"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def expand_model(old_model, new_cfg):
    """توسيع النموذج"""
    new_model = MiniGPT(**new_cfg).to(device)
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()

    for k in new_sd:
        if k in old_sd:
            old, new = old_sd[k], new_sd[k]
            if old.shape == new.shape:
                new_sd[k] = old
            elif len(old.shape) == len(new.shape):
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old.shape, new.shape))
                new_sd[k][slices] = old[slices]

    new_model.load_state_dict(new_sd)
    return new_model

def freeze_layers(model, n_last):
    """تجميد الطبقات"""
    total_params = sum(p.numel() for p in model.parameters())
    
    if n_last is None:
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = total_params
        print("🔓 Training for all classes")
    else:
        for p in model.parameters():
            p.requires_grad = False
        
        for block in model.blocks[-n_last:]:
            for p in block.parameters():
                p.requires_grad = True
        
        for p in model.ln_f.parameters():
            p.requires_grad = True
        for p in model.head.parameters():
            p.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔒 Another training {n_last} layers only")
    
    print(f"📊 Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")

def get_lr(it, warmup_steps=1000, max_lr=3e-4, min_lr=3e-5):
    """مجدول معدل التعلم مع تسخين"""
    if it < warmup_steps:
        return max_lr * (it / warmup_steps)
    else:
        # تباطؤ جيب التمام
        progress = (it - warmup_steps) / (epochs * 1000 - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ===================== الرئيسي =====================
def main():
    print("=" * 50)
    print("🚀 بدء تدريب MiniGPT")
    print("=" * 50)
    
    # 1. تحميل التوكينيزر
    print("\n🔹 Loading tokenizer...")
    
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"📁 Number of files: {len(files)}")
    
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = FastBPETokenizer()
        tokenizer.load(TOKENIZER_PATH)
        vocab_size = tokenizer.tokenizer.get_vocab_size()
        print(f"✅ Tokenizer loaded (vocab: {vocab_size})")
    else:
        print("🔹 New Tokenizer Training...")
        tokenizer = FastBPETokenizer(vocab_size=TARGET_VOCAB)
        tokenizer.train(files[:min(10, len(files))])
        tokenizer.save(TOKENIZER_PATH)
        vocab_size = tokenizer.tokenizer.get_vocab_size()
        print(f"✅ Training completed (vocab: {vocab_size})")
    
    # 2. تحضير البيانات
    print("\n🔹 Data preparation...")
    start_time = time.time()
    
    # استخدام جزء من الملفات للتدريب السريع
    train_files = files[:min(15, len(files))]
    dataset = TextChunkDataset(train_files, tokenizer, seq_len=max_seq_len, chunks_per_file=500)
    
    # محمل البيانات مع معالجة متوازية
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=True if device == "cuda" else False,
        drop_last=True
    )
    

    print(f"⏱️ Data preparation time: {time.time() - start_time:.2f} seconds")

    print(f"📊 Training sets: {len(dataset):,}")
    
    print(f"📦 Batch size: {batch_size}")
    
    # 3. تحميل أو إنشاء النموذج
    print("\n🔹 Preparing the form...")
    
    if os.path.exists(CKPT_PATH):
        print("🔹 Load checkpoint...")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        
        old_cfg = ckpt["config"]
        old_model = MiniGPT(**old_cfg).to(device)
        old_model.load_state_dict(ckpt["model"])
        
        new_cfg = {
            "vocab_size": vocab_size,
            "embed_dim": TARGET_EMBED,
            "num_heads": TARGET_HEADS,
            "num_layers": TARGET_LAYERS,
            "max_seq_len": max_seq_len
        }
        
        model = expand_model(old_model, new_cfg)
        start_epoch = ckpt["epoch"] + 1
        print(f"✅ The model was loaded from epoch {start_epoch-1}")
    else:
        print("🔹 Create a new form...")
        model = MiniGPT(
            vocab_size=vocab_size,
            embed_dim=TARGET_EMBED,
            num_heads=TARGET_HEADS,
            num_layers=TARGET_LAYERS,
            max_seq_len=max_seq_len
        ).to(device)
        start_epoch = 0
    
    # 4. تجميد الطبقات
    freeze_layers(model, NUM_LAYERS_TO_TRAIN)
    
    # 5. تحضير المحسن ومعيار الخسارة
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 6. استخدام Mixed Precision إن أمكن
    use_amp = device == "cuda"
    if use_amp:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("🎯 Mixed Precision Training has been activated")
    
    # 7. التدريب
    print("\n" + "=" * 50)
    print("🚀 Start training...")
    print("=" * 50)
    
    model.train()
    global_step = start_epoch * len(dataloader)
    total_steps = epochs * len(dataloader)
    
    start_training_time = time.time()
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0
            epoch_start_time = time.time()
            
            for batch_idx, (x, y) in enumerate(dataloader):
                # تحديث معدل التعلم
                current_lr = get_lr(global_step, warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # نقل البيانات إلى الجهاز
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # التدريب مع Mixed Precision
                if use_amp:
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # عرض التقدم
                if batch_idx % 10 == 0:
                    current_loss = loss.item()
                    progress = (epoch * len(dataloader) + batch_idx) / total_steps * 100
                    print(f"\rEpoch {epoch}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {current_loss:.4f} | LR: {current_lr:.6f} | "
                          f"Progress: {progress:.1f}%", end="")
            
            # إحصائيات الـepoch
            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            
            print(f"\rEpoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | "
                  f"LR: {current_lr:.6f}                 ")
            
            # حفظ checkpoint
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "embed_dim": TARGET_EMBED,
                        "num_heads": TARGET_HEADS,
                        "num_layers": TARGET_LAYERS,
                        "max_seq_len": max_seq_len
                    },
                    "loss": avg_loss
                }
                
                torch.save(checkpoint, CKPT_PATH)
                print(f"💾 Checkpoint saved (Epoch {epoch})")
            
            # تنظيف الذاكرة
            if epoch % 10 == 0:
                clear_memory()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Training paused by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 8. الحفظ النهائي
    print("\n🔹 Final Save...")
    
    final_package = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_type": "MiniGPT",
            "vocab_size": vocab_size,
            "embed_dim": TARGET_EMBED,
            "num_heads": TARGET_HEADS,
            "num_layers": TARGET_LAYERS,
            "max_seq_len": max_seq_len
        },
        "tokenizer_path": "tokenizer.json"
    }
    
    torch.save(final_package, FINAL_PATH)
    
    total_time = time.time() - start_training_time
    print(f"\n✅ Training complete!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📁 Final Form: {FINAL_PATH}")
    
    # 9. تقرير الأداء
    print("\n" + "=" * 50)
    print("📊 Final Performance Report")
    print("=" * 50)
    
    if device == "cuda":
        print(f"🎯 GPU user: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print(f"📈 Final Average Loss: {avg_loss:.4f}")
    print(f"🚀 Average speed: {total_steps/total_time:.2f} steps/second")

if __name__ == "__main__":
    # فحص الذاكرة الأولية
    if device == "cuda":
        print(f"💾 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # تشغيل التدريب
    main()
